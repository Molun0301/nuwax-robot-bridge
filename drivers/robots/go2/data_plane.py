from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
import importlib
import logging
import math
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from contracts.geometry import FrameTree, Pose, Quaternion, Transform, Vector3
from contracts.maps import CostMap, OccupancyGrid, SemanticMap, SemanticRegion
from contracts.navigation import ExplorationState, ExplorationStatus, ExploreAreaRequest, NavigationGoal, NavigationState, NavigationStatus

if TYPE_CHECKING:
    from drivers.robots.go2.settings import Go2DataPlaneConfig


LOGGER = logging.getLogger("nuwax_robot_bridge.go2.data_plane")

_KNOWN_GRID_MAP_LAYERS = {
    "elevation",
    "variance",
    "traversability",
    "normal_x",
    "normal_y",
    "normal_z",
    "upper_bound",
    "time",
    "min_filter",
    "smooth",
    "inpaint",
    "erosion",
    "rgb",
}


@dataclass
class ManagedRos2Process:
    """受控的 ROS2 子进程。"""

    name: str
    command: str
    setup_script: str = ""
    process: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)

    def start(self) -> None:
        """启动子进程。"""

        if self.process is not None and self.process.poll() is None:
            return
        shell_parts: List[str] = []
        if self.setup_script.strip():
            shell_parts.append(f"source {self.setup_script}")
        shell_parts.append(self.command)
        shell_command = " && ".join(shell_parts)
        LOGGER.info("启动 ROS2 子进程 name=%s command=%s", self.name, shell_command)
        self.process = subprocess.Popen(
            ["bash", "-lc", shell_command],
            start_new_session=True,
            stdout=None,
            stderr=None,
        )

    def stop(self, timeout_sec: float = 5.0) -> None:
        """停止子进程。"""

        if self.process is None or self.process.poll() is not None:
            return
        LOGGER.info("停止 ROS2 子进程 name=%s pid=%s", self.name, self.process.pid)
        self.process.terminate()
        try:
            self.process.wait(timeout=max(0.1, timeout_sec))
        except subprocess.TimeoutExpired:
            LOGGER.warning("ROS2 子进程超时未退出，强制杀死 name=%s pid=%s", self.name, self.process.pid)
            self.process.kill()
            self.process.wait(timeout=2.0)

    def is_running(self) -> bool:
        """返回子进程是否仍在运行。"""

        return self.process is not None and self.process.poll() is None


class RclpyGo2RosBridge:
    """基于 rclpy 的 Go2 ROS2 / Nav2 数据面桥。"""

    def __init__(self, config: "Go2DataPlaneConfig") -> None:
        self.config = config
        self._lock = threading.RLock()
        self._started = False
        self._owns_rclpy_context = False
        self._rclpy = None
        self._node = None
        self._executor = None
        self._executor_thread: Optional[threading.Thread] = None
        self._action_client = None
        self._goal_handle = None
        self._current_goal: Optional[NavigationGoal] = None
        self._latest_pose: Optional[Pose] = None
        self._tf_transforms: Dict[Tuple[str, str], Transform] = {}
        self._latest_occupancy_direct: Optional[OccupancyGrid] = None
        self._latest_occupancy_from_grid: Optional[OccupancyGrid] = None
        self._latest_cost_direct: Optional[CostMap] = None
        self._latest_cost_from_grid: Optional[CostMap] = None
        self._latest_semantic_map: Optional[SemanticMap] = None
        self._latest_navigation_state = NavigationState(status=NavigationStatus.IDLE)
        self._goal_reached = False
        self._last_error: Optional[str] = None

    def start(self) -> None:
        """启动 ROS2 桥。"""

        with self._lock:
            if self._started:
                return
            modules = self._load_ros_modules()
            if modules is None:
                message = "当前环境不可导入 rclpy 或 ROS2 消息包。"
                self._last_error = message
                if self.config.require_ros2:
                    raise RuntimeError(message)
                LOGGER.warning("%s Go2 ROS2 数据面保持未启用。", message)
                return

            self._rclpy = modules["rclpy"]
            if not self._rclpy.ok():
                self._rclpy.init(args=None)
                self._owns_rclpy_context = True
            node_class = modules["Node"]
            executor_class = modules["MultiThreadedExecutor"]
            self._node = node_class("nuwax_go2_data_plane")
            self._executor = executor_class(num_threads=2)
            self._executor.add_node(self._node)

            self._create_subscriptions(modules)
            self._create_nav2_client(modules)

            self._executor_thread = threading.Thread(
                target=self._executor.spin,
                name="nuwax_go2_ros2_executor",
                daemon=True,
            )
            self._executor_thread.start()
            self._started = True
            self._last_error = None
            LOGGER.info("Go2 ROS2 数据面桥已启动。")

    def stop(self) -> None:
        """停止 ROS2 桥。"""

        with self._lock:
            if not self._started:
                return
            try:
                if self._executor is not None:
                    self._executor.shutdown(timeout_sec=1.0)
            except Exception:
                LOGGER.exception("关闭 ROS2 executor 失败。")
            try:
                if self._node is not None:
                    self._node.destroy_node()
            except Exception:
                LOGGER.exception("销毁 ROS2 节点失败。")
            if self._executor_thread is not None:
                self._executor_thread.join(timeout=2.0)
            if self._owns_rclpy_context and self._rclpy is not None:
                try:
                    self._rclpy.shutdown()
                except Exception:
                    LOGGER.exception("关闭 rclpy 上下文失败。")
            self._executor = None
            self._node = None
            self._executor_thread = None
            self._action_client = None
            self._goal_handle = None
            self._started = False
            LOGGER.info("Go2 ROS2 数据面桥已停止。")

    def is_running(self) -> bool:
        """返回桥是否已启动。"""

        return self._started

    def is_localization_available(self) -> bool:
        """返回定位数据是否可用。"""

        return self._latest_pose is not None or bool(self._tf_transforms)

    def is_map_available(self) -> bool:
        """返回地图数据是否可用。"""

        return (
            self._latest_occupancy_direct is not None
            or self._latest_occupancy_from_grid is not None
            or self._latest_cost_direct is not None
            or self._latest_cost_from_grid is not None
            or self._latest_semantic_map is not None
        )

    def is_navigation_available(self) -> bool:
        """返回 Nav2 导航能力是否可用。"""

        return self._action_client is not None and self._started

    def get_current_pose(self) -> Optional[Pose]:
        """读取当前位姿。"""

        with self._lock:
            return self._latest_pose.model_copy(deep=True) if self._latest_pose is not None else None

    def get_frame_tree(self) -> Optional[FrameTree]:
        """读取当前 TF 树快照。"""

        with self._lock:
            transforms = [item.model_copy(deep=True) for item in self._tf_transforms.values()]
            if self._latest_pose is not None:
                base_frame = self.config.map_synthesis.base_frame
                parent_frame = self._latest_pose.frame_id
                synthetic_key = (parent_frame, base_frame)
                if synthetic_key not in self._tf_transforms:
                    transforms.append(
                        Transform(
                            parent_frame_id=parent_frame,
                            child_frame_id=base_frame,
                            translation=self._latest_pose.position,
                            rotation=self._latest_pose.orientation,
                            authority="odom_pose",
                        )
                    )
            if not transforms:
                return None
            return FrameTree(
                root_frame_id=self.config.map_synthesis.map_frame,
                transforms=transforms,
            )

    def get_occupancy_grid(self) -> Optional[OccupancyGrid]:
        """读取占据栅格地图。"""

        with self._lock:
            grid = self._latest_occupancy_direct or self._latest_occupancy_from_grid
            return grid.model_copy(deep=True) if grid is not None else None

    def get_cost_map(self) -> Optional[CostMap]:
        """读取代价地图。"""

        with self._lock:
            cost_map = self._latest_cost_direct or self._latest_cost_from_grid
            return cost_map.model_copy(deep=True) if cost_map is not None else None

    def get_semantic_map(self) -> Optional[SemanticMap]:
        """读取语义地图。"""

        with self._lock:
            return self._latest_semantic_map.model_copy(deep=True) if self._latest_semantic_map is not None else None

    def set_goal(self, goal: NavigationGoal) -> bool:
        """提交 Nav2 导航目标。"""

        if not self.is_navigation_available() or goal.target_pose is None:
            return False
        assert self._action_client is not None
        timeout = max(0.1, self.config.nav2.action_wait_timeout_sec)
        if not self._action_client.wait_for_server(timeout_sec=timeout):
            self._latest_navigation_state = NavigationState(
                current_goal_id=goal.goal_id,
                status=NavigationStatus.FAILED,
                current_pose=self.get_current_pose(),
                message="Nav2 action 服务未就绪。",
            )
            self._goal_reached = False
            return False

        goal_msg = self._build_nav2_goal(goal)
        self._current_goal = goal
        self._goal_reached = False
        self._latest_navigation_state = NavigationState(
            current_goal_id=goal.goal_id,
            status=NavigationStatus.PLANNING,
            current_pose=self.get_current_pose(),
            message="导航目标已提交到 Nav2。",
        )
        future = self._action_client.send_goal_async(goal_msg, feedback_callback=self._on_navigation_feedback)
        future.add_done_callback(self._on_navigation_goal_response)
        return True

    def cancel_goal(self) -> bool:
        """取消当前导航目标。"""

        with self._lock:
            goal_handle = self._goal_handle
            current_goal = self._current_goal
        if goal_handle is None:
            self._latest_navigation_state = NavigationState(
                current_goal_id=current_goal.goal_id if current_goal is not None else None,
                status=NavigationStatus.CANCELLED,
                current_pose=self.get_current_pose(),
                message="当前没有活动中的 Nav2 目标。",
            )
            return False

        future = goal_handle.cancel_goal_async()
        future.add_done_callback(self._on_navigation_cancel_response)
        return True

    def get_navigation_state(self) -> NavigationState:
        """读取导航状态。"""

        with self._lock:
            return self._latest_navigation_state.model_copy(deep=True)

    def is_goal_reached(self) -> bool:
        """判断导航目标是否已到达。"""

        with self._lock:
            return self._goal_reached

    def get_status(self) -> Dict[str, object]:
        """返回当前桥状态。"""

        with self._lock:
            return {
                "started": self._started,
                "localization_available": self.is_localization_available(),
                "map_available": self.is_map_available(),
                "navigation_available": self.is_navigation_available(),
                "last_error": self._last_error,
            }

    def _load_ros_modules(self) -> Optional[Dict[str, object]]:
        try:
            modules = {
                "rclpy": importlib.import_module("rclpy"),
                "Node": getattr(importlib.import_module("rclpy.node"), "Node"),
                "MultiThreadedExecutor": getattr(importlib.import_module("rclpy.executors"), "MultiThreadedExecutor"),
                "ActionClient": getattr(importlib.import_module("rclpy.action"), "ActionClient"),
                "Odometry": getattr(importlib.import_module("nav_msgs.msg"), "Odometry"),
                "RosOccupancyGrid": getattr(importlib.import_module("nav_msgs.msg"), "OccupancyGrid"),
                "TFMessage": getattr(importlib.import_module("tf2_msgs.msg"), "TFMessage"),
                "PoseStamped": getattr(importlib.import_module("geometry_msgs.msg"), "PoseStamped"),
            }
            try:
                modules["GridMap"] = getattr(importlib.import_module("grid_map_msgs.msg"), "GridMap")
            except Exception:
                modules["GridMap"] = None
            try:
                modules["NavigateToPose"] = getattr(importlib.import_module("nav2_msgs.action"), "NavigateToPose")
            except Exception:
                modules["NavigateToPose"] = None
            try:
                modules["Nav2Costmap"] = getattr(importlib.import_module("nav2_msgs.msg"), "Costmap")
            except Exception:
                modules["Nav2Costmap"] = None
            return modules
        except Exception:
            LOGGER.exception("导入 ROS2 Python 模块失败。")
            return None

    def _create_subscriptions(self, modules: Dict[str, object]) -> None:
        assert self._node is not None
        topics = self.config.topics
        self._node.create_subscription(modules["Odometry"], topics.odom_topic, self._on_odom, 20)
        self._node.create_subscription(modules["TFMessage"], topics.tf_topic, self._on_tf_message, 50)
        self._node.create_subscription(modules["TFMessage"], topics.tf_static_topic, self._on_tf_message, 10)
        self._node.create_subscription(modules["RosOccupancyGrid"], topics.occupancy_topic, self._on_occupancy_grid, 10)
        if modules.get("GridMap") is not None and topics.grid_map_topic.strip():
            self._node.create_subscription(modules["GridMap"], topics.grid_map_topic, self._on_grid_map, 10)
        if modules.get("Nav2Costmap") is not None and topics.cost_map_topic.strip():
            self._node.create_subscription(modules["Nav2Costmap"], topics.cost_map_topic, self._on_cost_map, 10)

    def _create_nav2_client(self, modules: Dict[str, object]) -> None:
        if not self.config.nav2.enabled or modules.get("NavigateToPose") is None:
            return
        assert self._node is not None
        action_client_cls = modules["ActionClient"]
        self._action_client = action_client_cls(
            self._node,
            modules["NavigateToPose"],
            self.config.topics.navigate_to_pose_action,
        )

    def _on_odom(self, message: object) -> None:
        pose = self._build_pose_from_odom(message)
        with self._lock:
            self._latest_pose = pose

    def _on_tf_message(self, message: object) -> None:
        transforms = getattr(message, "transforms", [])
        with self._lock:
            for transform in transforms:
                parent_frame = str(getattr(getattr(transform, "header", None), "frame_id", "") or "").strip()
                child_frame = str(getattr(transform, "child_frame_id", "") or "").strip()
                if not parent_frame or not child_frame:
                    continue
                self._tf_transforms[(parent_frame, child_frame)] = self._build_transform_from_ros(transform)

    def _on_occupancy_grid(self, message: object) -> None:
        with self._lock:
            self._latest_occupancy_direct = self._build_contract_occupancy_grid(message)

    def _on_cost_map(self, message: object) -> None:
        with self._lock:
            self._latest_cost_direct = self._build_contract_cost_map(message)

    def _on_grid_map(self, message: object) -> None:
        occupancy = self._build_occupancy_from_grid_map(message)
        cost_map = self._build_cost_map_from_grid_map(message)
        semantic_map = self._build_semantic_map_from_grid_map(message)
        with self._lock:
            self._latest_occupancy_from_grid = occupancy
            self._latest_cost_from_grid = cost_map
            self._latest_semantic_map = semantic_map

    def _build_pose_from_odom(self, message: object) -> Pose:
        header = getattr(message, "header", None)
        pose_with_cov = getattr(message, "pose", None)
        pose_obj = getattr(pose_with_cov, "pose", None)
        frame_id = str(getattr(header, "frame_id", "") or self.config.map_synthesis.map_frame)
        return Pose(
            frame_id=frame_id,
            position=Vector3(
                x=float(getattr(getattr(pose_obj, "position", None), "x", 0.0)),
                y=float(getattr(getattr(pose_obj, "position", None), "y", 0.0)),
                z=float(getattr(getattr(pose_obj, "position", None), "z", 0.0)),
            ),
            orientation=Quaternion(
                x=float(getattr(getattr(pose_obj, "orientation", None), "x", 0.0)),
                y=float(getattr(getattr(pose_obj, "orientation", None), "y", 0.0)),
                z=float(getattr(getattr(pose_obj, "orientation", None), "z", 0.0)),
                w=float(getattr(getattr(pose_obj, "orientation", None), "w", 1.0)),
            ),
        )

    def _build_transform_from_ros(self, message: object) -> Transform:
        transform = getattr(message, "transform", None)
        header = getattr(message, "header", None)
        return Transform(
            parent_frame_id=str(getattr(header, "frame_id", "") or self.config.map_synthesis.map_frame),
            child_frame_id=str(getattr(message, "child_frame_id", "") or self.config.map_synthesis.base_frame),
            translation=Vector3(
                x=float(getattr(getattr(transform, "translation", None), "x", 0.0)),
                y=float(getattr(getattr(transform, "translation", None), "y", 0.0)),
                z=float(getattr(getattr(transform, "translation", None), "z", 0.0)),
            ),
            rotation=Quaternion(
                x=float(getattr(getattr(transform, "rotation", None), "x", 0.0)),
                y=float(getattr(getattr(transform, "rotation", None), "y", 0.0)),
                z=float(getattr(getattr(transform, "rotation", None), "z", 0.0)),
                w=float(getattr(getattr(transform, "rotation", None), "w", 1.0)),
            ),
            authority="ros2_tf",
        )

    def _build_contract_occupancy_grid(self, message: object) -> OccupancyGrid:
        info = getattr(message, "info", None)
        header = getattr(message, "header", None)
        frame_id = str(getattr(header, "frame_id", "") or self.config.map_synthesis.map_frame)
        return OccupancyGrid(
            map_id="go2_occupancy_map",
            frame_id=frame_id,
            width=int(getattr(info, "width", 1)),
            height=int(getattr(info, "height", 1)),
            resolution_m=float(getattr(info, "resolution", 0.1) or 0.1),
            origin=self._build_pose_from_ros_pose(getattr(info, "origin", None), frame_id=frame_id),
            data=[int(value) for value in getattr(message, "data", [])],
        )

    def _build_contract_cost_map(self, message: object) -> CostMap:
        metadata = getattr(message, "metadata", None)
        frame_id = str(getattr(metadata, "layer", "") or self.config.map_synthesis.map_frame)
        if metadata is not None and getattr(metadata, "map_load_time", None) is not None:
            frame_id = str(getattr(getattr(message, "header", None), "frame_id", "") or self.config.map_synthesis.map_frame)
        width = int(getattr(metadata, "size_x", getattr(message, "width", 1)))
        height = int(getattr(metadata, "size_y", getattr(message, "height", 1)))
        resolution = float(getattr(metadata, "resolution", getattr(message, "resolution", 0.1)) or 0.1)
        origin_pose = getattr(metadata, "origin", getattr(message, "origin", None))
        raw_data = getattr(message, "data", [])
        return CostMap(
            map_id="go2_nav2_cost_map",
            frame_id=frame_id,
            width=width,
            height=height,
            resolution_m=resolution,
            origin=self._build_pose_from_ros_pose(origin_pose, frame_id=frame_id),
            data=[float(value) for value in raw_data],
        )

    def _build_pose_from_ros_pose(self, pose_obj: object, *, frame_id: str) -> Pose:
        return Pose(
            frame_id=frame_id,
            position=Vector3(
                x=float(getattr(getattr(pose_obj, "position", None), "x", 0.0)),
                y=float(getattr(getattr(pose_obj, "position", None), "y", 0.0)),
                z=float(getattr(getattr(pose_obj, "position", None), "z", 0.0)),
            ),
            orientation=Quaternion(
                x=float(getattr(getattr(pose_obj, "orientation", None), "x", 0.0)),
                y=float(getattr(getattr(pose_obj, "orientation", None), "y", 0.0)),
                z=float(getattr(getattr(pose_obj, "orientation", None), "z", 0.0)),
                w=float(getattr(getattr(pose_obj, "orientation", None), "w", 1.0)),
            ),
        )

    def _grid_map_dimensions(self, message: object) -> Tuple[int, int]:
        info = getattr(message, "info", None)
        resolution = float(getattr(info, "resolution", 0.1) or 0.1)
        length_x = float(getattr(info, "length_x", resolution))
        length_y = float(getattr(info, "length_y", resolution))
        width = max(1, int(round(length_x / resolution)))
        height = max(1, int(round(length_y / resolution)))
        return width, height

    def _grid_map_origin_pose(self, message: object, *, frame_id: str) -> Pose:
        info = getattr(message, "info", None)
        pose_obj = getattr(info, "pose", None)
        length_x = float(getattr(info, "length_x", 0.0))
        length_y = float(getattr(info, "length_y", 0.0))
        center_x = float(getattr(getattr(pose_obj, "position", None), "x", 0.0))
        center_y = float(getattr(getattr(pose_obj, "position", None), "y", 0.0))
        center_z = float(getattr(getattr(pose_obj, "position", None), "z", 0.0))
        return Pose(
            frame_id=frame_id,
            position=Vector3(
                x=center_x - length_x / 2.0,
                y=center_y - length_y / 2.0,
                z=center_z,
            ),
            orientation=Quaternion(
                x=float(getattr(getattr(pose_obj, "orientation", None), "x", 0.0)),
                y=float(getattr(getattr(pose_obj, "orientation", None), "y", 0.0)),
                z=float(getattr(getattr(pose_obj, "orientation", None), "z", 0.0)),
                w=float(getattr(getattr(pose_obj, "orientation", None), "w", 1.0)),
            ),
        )

    def _extract_grid_layer(self, message: object, layer_name: str) -> Optional[np.ndarray]:
        layers = list(getattr(message, "layers", []))
        if layer_name not in layers:
            return None
        layer_index = layers.index(layer_name)
        data_entries = list(getattr(message, "data", []))
        if layer_index >= len(data_entries):
            return None
        raw_data = np.asarray(list(getattr(data_entries[layer_index], "data", [])), dtype=np.float32)
        width, height = self._grid_map_dimensions(message)
        if raw_data.size != width * height:
            return None
        matrix = raw_data.reshape(height, width)
        outer_start_index = int(getattr(message, "outer_start_index", 0))
        inner_start_index = int(getattr(message, "inner_start_index", 0))
        if outer_start_index:
            matrix = np.roll(matrix, -outer_start_index, axis=0)
        if inner_start_index:
            matrix = np.roll(matrix, -inner_start_index, axis=1)
        return matrix

    def _build_occupancy_from_grid_map(self, message: object) -> Optional[OccupancyGrid]:
        traversability = self._extract_grid_layer(message, self.config.map_synthesis.traversability_layer)
        if traversability is None:
            return None
        validity = None
        if self.config.map_synthesis.validity_layer.strip():
            validity = self._extract_grid_layer(message, self.config.map_synthesis.validity_layer)
        occupancy = np.clip(np.rint((1.0 - traversability) * 100.0), 0, 100).astype(np.int32)
        occupancy = np.where(traversability >= self.config.map_synthesis.free_threshold, 0, occupancy)
        occupancy = np.where(traversability <= self.config.map_synthesis.lethal_threshold, 100, occupancy)
        if validity is not None:
            occupancy = np.where(validity <= 0.0, -1, occupancy)
        width, height = self._grid_map_dimensions(message)
        frame_id = str(getattr(getattr(message, "header", None), "frame_id", "") or self.config.map_synthesis.map_frame)
        return OccupancyGrid(
            map_id="go2_gridmap_occupancy",
            frame_id=frame_id,
            width=width,
            height=height,
            resolution_m=float(getattr(getattr(message, "info", None), "resolution", 0.1) or 0.1),
            origin=self._grid_map_origin_pose(message, frame_id=frame_id),
            data=[int(value) for value in occupancy.flatten().tolist()],
        )

    def _build_cost_map_from_grid_map(self, message: object) -> Optional[CostMap]:
        traversability = self._extract_grid_layer(message, self.config.map_synthesis.traversability_layer)
        if traversability is None:
            return None
        validity = None
        if self.config.map_synthesis.validity_layer.strip():
            validity = self._extract_grid_layer(message, self.config.map_synthesis.validity_layer)
        cost = np.clip((1.0 - traversability) * 100.0, 0.0, 100.0)
        if validity is not None:
            cost = np.where(validity <= 0.0, 100.0, cost)
        width, height = self._grid_map_dimensions(message)
        frame_id = str(getattr(getattr(message, "header", None), "frame_id", "") or self.config.map_synthesis.map_frame)
        return CostMap(
            map_id="go2_gridmap_cost",
            frame_id=frame_id,
            width=width,
            height=height,
            resolution_m=float(getattr(getattr(message, "info", None), "resolution", 0.1) or 0.1),
            origin=self._grid_map_origin_pose(message, frame_id=frame_id),
            data=[float(value) for value in cost.flatten().tolist()],
        )

    def _build_semantic_map_from_grid_map(self, message: object) -> Optional[SemanticMap]:
        frame_id = str(getattr(getattr(message, "header", None), "frame_id", "") or self.config.map_synthesis.map_frame)
        width, height = self._grid_map_dimensions(message)
        origin = self._grid_map_origin_pose(message, frame_id=frame_id)
        resolution = float(getattr(getattr(message, "info", None), "resolution", 0.1) or 0.1)
        semantic_layers = list(self.config.map_synthesis.semantic_layers)
        if not semantic_layers:
            semantic_layers = [
                layer
                for layer in list(getattr(message, "layers", []))
                if layer not in _KNOWN_GRID_MAP_LAYERS
            ]
        regions: List[SemanticRegion] = []
        for layer_name in semantic_layers:
            mask_source = self._extract_grid_layer(message, layer_name)
            if mask_source is None:
                continue
            mask = np.asarray(mask_source >= self.config.map_synthesis.semantic_threshold, dtype=np.uint8)
            regions.extend(
                self._regions_from_mask(
                    mask=mask,
                    score_source=mask_source,
                    label=layer_name,
                    frame_id=frame_id,
                    width=width,
                    height=height,
                    origin=origin,
                    resolution=resolution,
                )
            )
        if not regions:
            traversability = self._extract_grid_layer(message, self.config.map_synthesis.traversability_layer)
            if traversability is not None:
                regions.extend(
                    self._regions_from_mask(
                        mask=np.asarray(traversability >= self.config.map_synthesis.free_threshold, dtype=np.uint8),
                        score_source=traversability,
                        label="traversable",
                        frame_id=frame_id,
                        width=width,
                        height=height,
                        origin=origin,
                        resolution=resolution,
                    )
                )
                regions.extend(
                    self._regions_from_mask(
                        mask=np.asarray(traversability <= self.config.map_synthesis.lethal_threshold, dtype=np.uint8),
                        score_source=1.0 - traversability,
                        label="hazard",
                        frame_id=frame_id,
                        width=width,
                        height=height,
                        origin=origin,
                        resolution=resolution,
                    )
                )
        if not regions:
            return None
        return SemanticMap(
            map_id="go2_semantic_map",
            frame_id=frame_id,
            regions=regions,
            metadata={"source": "grid_map"},
        )

    def _regions_from_mask(
        self,
        *,
        mask: np.ndarray,
        score_source: np.ndarray,
        label: str,
        frame_id: str,
        width: int,
        height: int,
        origin: Pose,
        resolution: float,
    ) -> List[SemanticRegion]:
        if mask.size == 0:
            return []
        visited = np.zeros_like(mask, dtype=np.uint8)
        regions: List[SemanticRegion] = []
        region_index = 0
        for row in range(height):
            for col in range(width):
                if mask[row, col] == 0 or visited[row, col] == 1:
                    continue
                component = self._flood_fill(mask, visited, start_row=row, start_col=col)
                if len(component) < self.config.map_synthesis.semantic_min_cells:
                    continue
                xs: List[float] = []
                ys: List[float] = []
                scores: List[float] = []
                rows = [cell[0] for cell in component]
                cols = [cell[1] for cell in component]
                for item_row, item_col in component:
                    xs.append(origin.position.x + (item_col + 0.5) * resolution)
                    ys.append(origin.position.y + (item_row + 0.5) * resolution)
                    scores.append(float(score_source[item_row, item_col]))
                min_row = min(rows)
                max_row = max(rows)
                min_col = min(cols)
                max_col = max(cols)
                polygon = [
                    Vector3(x=origin.position.x + min_col * resolution, y=origin.position.y + min_row * resolution, z=0.0),
                    Vector3(x=origin.position.x + (max_col + 1) * resolution, y=origin.position.y + min_row * resolution, z=0.0),
                    Vector3(x=origin.position.x + (max_col + 1) * resolution, y=origin.position.y + (max_row + 1) * resolution, z=0.0),
                    Vector3(x=origin.position.x + min_col * resolution, y=origin.position.y + (max_row + 1) * resolution, z=0.0),
                ]
                region_index += 1
                regions.append(
                    SemanticRegion(
                        region_id=f"{label}_{region_index}",
                        label=label,
                        centroid=Pose(
                            frame_id=frame_id,
                            position=Vector3(
                                x=float(sum(xs) / len(xs)),
                                y=float(sum(ys) / len(ys)),
                                z=0.0,
                            ),
                            orientation=Quaternion(w=1.0),
                        ),
                        polygon_points=polygon,
                        attributes={
                            "source": "grid_map",
                            "cell_count": len(component),
                            "score_mean": float(sum(scores) / len(scores)),
                        },
                    )
                )
        return regions

    def _flood_fill(self, mask: np.ndarray, visited: np.ndarray, *, start_row: int, start_col: int) -> List[Tuple[int, int]]:
        queue = [(start_row, start_col)]
        visited[start_row, start_col] = 1
        component: List[Tuple[int, int]] = []
        height, width = mask.shape
        while queue:
            row, col = queue.pop()
            component.append((row, col))
            neighbors = (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            )
            for next_row, next_col in neighbors:
                if next_row < 0 or next_row >= height or next_col < 0 or next_col >= width:
                    continue
                if visited[next_row, next_col] == 1 or mask[next_row, next_col] == 0:
                    continue
                visited[next_row, next_col] = 1
                queue.append((next_row, next_col))
        return component

    def _build_nav2_goal(self, goal: NavigationGoal) -> object:
        assert self._node is not None
        assert self._action_client is not None
        goal_message = self._action_client._action_type.Goal()
        pose_stamped = importlib.import_module("geometry_msgs.msg").PoseStamped()
        pose_stamped.header.frame_id = goal.target_pose.frame_id
        pose_stamped.pose.position.x = goal.target_pose.position.x
        pose_stamped.pose.position.y = goal.target_pose.position.y
        pose_stamped.pose.position.z = goal.target_pose.position.z
        pose_stamped.pose.orientation.x = goal.target_pose.orientation.x
        pose_stamped.pose.orientation.y = goal.target_pose.orientation.y
        pose_stamped.pose.orientation.z = goal.target_pose.orientation.z
        pose_stamped.pose.orientation.w = goal.target_pose.orientation.w
        goal_message.pose = pose_stamped
        if hasattr(goal_message, "behavior_tree"):
            goal_message.behavior_tree = str(goal.metadata.get("behavior_tree", ""))
        return goal_message

    def _on_navigation_goal_response(self, future: object) -> None:
        try:
            goal_handle = future.result()
        except Exception:
            LOGGER.exception("Nav2 目标提交失败。")
            with self._lock:
                goal_id = self._current_goal.goal_id if self._current_goal is not None else None
                self._latest_navigation_state = NavigationState(
                    current_goal_id=goal_id,
                    status=NavigationStatus.FAILED,
                    current_pose=self.get_current_pose(),
                    message="提交 Nav2 目标失败。",
                )
            return
        with self._lock:
            if goal_handle is None or not getattr(goal_handle, "accepted", False):
                goal_id = self._current_goal.goal_id if self._current_goal is not None else None
                self._latest_navigation_state = NavigationState(
                    current_goal_id=goal_id,
                    status=NavigationStatus.FAILED,
                    current_pose=self.get_current_pose(),
                    message="Nav2 拒绝了导航目标。",
                )
                return
            self._goal_handle = goal_handle
            goal_id = self._current_goal.goal_id if self._current_goal is not None else None
            self._latest_navigation_state = NavigationState(
                current_goal_id=goal_id,
                status=NavigationStatus.ACCEPTED,
                current_pose=self.get_current_pose(),
                message="Nav2 已接受导航目标。",
            )
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_navigation_result)

    def _on_navigation_feedback(self, feedback_message: object) -> None:
        feedback = getattr(feedback_message, "feedback", feedback_message)
        current_pose_msg = getattr(feedback, "current_pose", None)
        current_pose = None
        if current_pose_msg is not None:
            current_pose = self._build_pose_from_ros_pose(current_pose_msg.pose, frame_id=current_pose_msg.header.frame_id)
        remaining_distance = getattr(feedback, "distance_remaining", None)
        eta_msg = getattr(feedback, "estimated_time_remaining", None)
        eta_sec = None
        if eta_msg is not None:
            eta_sec = float(getattr(eta_msg, "sec", 0.0)) + float(getattr(eta_msg, "nanosec", 0.0)) / 1e9
        with self._lock:
            goal_id = self._current_goal.goal_id if self._current_goal is not None else None
            self._latest_navigation_state = NavigationState(
                current_goal_id=goal_id,
                status=NavigationStatus.RUNNING,
                current_pose=current_pose or self._latest_pose,
                remaining_distance_m=float(remaining_distance) if remaining_distance is not None else None,
                goal_reached=False,
                message="Nav2 正在执行导航。",
                metadata={"estimated_time_remaining_sec": eta_sec},
            )

    def _on_navigation_result(self, future: object) -> None:
        try:
            result_wrapper = future.result()
        except Exception:
            LOGGER.exception("读取 Nav2 结果失败。")
            with self._lock:
                goal_id = self._current_goal.goal_id if self._current_goal is not None else None
                self._latest_navigation_state = NavigationState(
                    current_goal_id=goal_id,
                    status=NavigationStatus.FAILED,
                    current_pose=self.get_current_pose(),
                    message="读取 Nav2 结果失败。",
                )
                self._goal_handle = None
                self._goal_reached = False
            return
        result = getattr(result_wrapper, "result", result_wrapper)
        error_code = int(getattr(result, "error_code", 0) or 0)
        status = NavigationStatus.SUCCEEDED if error_code == 0 else NavigationStatus.FAILED
        message = "导航目标已到达。" if status == NavigationStatus.SUCCEEDED else f"Nav2 导航失败，错误码 {error_code}。"
        with self._lock:
            goal_id = self._current_goal.goal_id if self._current_goal is not None else None
            self._latest_navigation_state = NavigationState(
                current_goal_id=goal_id,
                status=status,
                current_pose=self.get_current_pose(),
                remaining_distance_m=0.0 if status == NavigationStatus.SUCCEEDED else None,
                goal_reached=status == NavigationStatus.SUCCEEDED,
                message=message,
                metadata={"error_code": error_code},
            )
            self._goal_handle = None
            self._goal_reached = status == NavigationStatus.SUCCEEDED

    def _on_navigation_cancel_response(self, future: object) -> None:
        try:
            future.result()
        except Exception:
            LOGGER.exception("取消 Nav2 目标失败。")
        with self._lock:
            goal_id = self._current_goal.goal_id if self._current_goal is not None else None
            self._latest_navigation_state = NavigationState(
                current_goal_id=goal_id,
                status=NavigationStatus.CANCELLED,
                current_pose=self.get_current_pose(),
                message="Nav2 导航目标已取消。",
            )
            self._goal_handle = None
            self._goal_reached = False


class Go2DataPlaneRuntime:
    """Go2 定位、地图、导航、探索数据面运行时。"""

    def __init__(
        self,
        config: "Go2DataPlaneConfig",
        *,
        bridge: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.bridge = bridge or RclpyGo2RosBridge(config)
        self._processes: List[ManagedRos2Process] = self._build_process_specs()
        self._exploration_lock = threading.RLock()
        self._exploration_stop_event = threading.Event()
        self._exploration_thread: Optional[threading.Thread] = None
        self._exploration_state = ExplorationState(status=ExplorationStatus.IDLE)
        self._started = False

    def start(self) -> None:
        """启动数据面运行时。"""

        if not self.config.enabled:
            LOGGER.info("Go2 数据面未启用，跳过启动。")
            return
        if self._started:
            return
        for process in self._processes:
            process.start()
            if self.config.lidar_pipeline.launch_stagger_sec > 0.0:
                time.sleep(self.config.lidar_pipeline.launch_stagger_sec)
        if self.config.nav2.auto_launch and self.config.nav2.launch_stabilize_sec > 0.0:
            time.sleep(self.config.nav2.launch_stabilize_sec)
        self.bridge.start()
        self._started = True

    def stop(self) -> None:
        """停止数据面运行时。"""

        self.stop_exploration()
        self.bridge.stop()
        for process in reversed(self._processes):
            process.stop()
        self._started = False

    def is_started(self) -> bool:
        """返回数据面是否已启动。"""

        return self._started

    def is_localization_available(self) -> bool:
        """返回定位能力是否可用。"""

        return self.config.enabled and bool(self.bridge.is_localization_available())

    def is_map_available(self) -> bool:
        """返回地图能力是否可用。"""

        return self.config.enabled and bool(self.bridge.is_map_available())

    def is_navigation_available(self) -> bool:
        """返回导航能力是否可用。"""

        return self.config.enabled and self.config.nav2.enabled and bool(self.bridge.is_navigation_available())

    def is_exploration_available(self) -> bool:
        """返回探索能力是否可用。"""

        return self.is_navigation_available() and self.config.exploration.enabled

    def get_current_pose(self) -> Optional[Pose]:
        """读取当前位姿。"""

        return self.bridge.get_current_pose()

    def get_frame_tree(self) -> Optional[FrameTree]:
        """读取当前 TF 树。"""

        return self.bridge.get_frame_tree()

    def get_occupancy_grid(self) -> Optional[OccupancyGrid]:
        """读取占据栅格地图。"""

        return self.bridge.get_occupancy_grid()

    def get_cost_map(self) -> Optional[CostMap]:
        """读取代价地图。"""

        return self.bridge.get_cost_map()

    def get_semantic_map(self) -> Optional[SemanticMap]:
        """读取语义地图。"""

        return self.bridge.get_semantic_map()

    def set_goal(self, goal: NavigationGoal) -> bool:
        """提交导航目标。"""

        return self.bridge.set_goal(goal)

    def cancel_goal(self) -> bool:
        """取消导航目标。"""

        return self.bridge.cancel_goal()

    def get_navigation_state(self) -> NavigationState:
        """读取导航状态。"""

        return self.bridge.get_navigation_state()

    def is_goal_reached(self) -> bool:
        """判断当前导航目标是否已完成。"""

        return self.bridge.is_goal_reached()

    def start_exploration(self, request: ExploreAreaRequest) -> bool:
        """启动探索任务。"""

        if not self.is_exploration_available():
            return False
        with self._exploration_lock:
            if self._exploration_thread is not None and self._exploration_thread.is_alive():
                return False
            self._exploration_stop_event.clear()
            self._exploration_state = ExplorationState(
                current_request_id=request.request_id,
                status=ExplorationStatus.ACCEPTED,
                strategy=request.strategy,
                message="探索任务已接受。",
            )
            self._exploration_thread = threading.Thread(
                target=self._run_exploration,
                args=(request,),
                name=f"go2_exploration_{request.request_id}",
                daemon=True,
            )
            self._exploration_thread.start()
        return True

    def stop_exploration(self) -> bool:
        """停止探索任务。"""

        with self._exploration_lock:
            self._exploration_stop_event.set()
            thread = self._exploration_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        with self._exploration_lock:
            current_request_id = self._exploration_state.current_request_id
            self._exploration_state = ExplorationState(
                current_request_id=current_request_id,
                status=ExplorationStatus.CANCELLED if current_request_id else ExplorationStatus.IDLE,
                strategy=self._exploration_state.strategy,
                message="探索任务已取消。" if current_request_id else None,
            )
            self._exploration_thread = None
        self.cancel_goal()
        return True

    def get_exploration_state(self) -> ExplorationState:
        """读取探索状态。"""

        with self._exploration_lock:
            return self._exploration_state.model_copy(deep=True)

    def get_status(self) -> Dict[str, object]:
        """返回当前数据面状态。"""

        return {
            "enabled": self.config.enabled,
            "started": self._started,
            "bridge": self.bridge.get_status() if hasattr(self.bridge, "get_status") else {},
            "localization_available": self.is_localization_available(),
            "map_available": self.is_map_available(),
            "navigation_available": self.is_navigation_available(),
            "exploration_available": self.is_exploration_available(),
            "managed_processes": [
                {
                    "name": process.name,
                    "command": process.command,
                    "running": process.is_running(),
                }
                for process in self._processes
            ],
        }

    def _build_process_specs(self) -> List[ManagedRos2Process]:
        processes: List[ManagedRos2Process] = []
        if self.config.lidar_pipeline.enabled and self.config.lidar_pipeline.auto_launch:
            for index, command in enumerate(self.config.lidar_pipeline.launch_commands, start=1):
                processes.append(
                    ManagedRos2Process(
                        name=f"go2_lidar_pipeline_{index}",
                        command=command,
                        setup_script=self.config.lidar_pipeline.setup_script or self.config.setup_script,
                    )
                )
        if self.config.nav2.enabled and self.config.nav2.auto_launch and self.config.nav2.launch_command.strip():
            nav2_command = self.config.nav2.launch_command
            if self.config.dds_iface.strip() and "dds_iface:=" not in nav2_command:
                nav2_command = f"{nav2_command} dds_iface:={shlex.quote(self.config.dds_iface)}"
            processes.append(
                ManagedRos2Process(
                    name="go2_nav2",
                    command=nav2_command,
                    setup_script=self.config.nav2.setup_script or self.config.setup_script,
                )
            )
        return processes

    def _run_exploration(self, request: ExploreAreaRequest) -> None:
        try:
            with self._exploration_lock:
                self._exploration_state = ExplorationState(
                    current_request_id=request.request_id,
                    status=ExplorationStatus.RUNNING,
                    strategy=request.strategy,
                    message="探索任务运行中。",
                )

            candidate_poses = self._build_exploration_candidates(request)
            if not candidate_poses:
                with self._exploration_lock:
                    self._exploration_state = ExplorationState(
                        current_request_id=request.request_id,
                        status=ExplorationStatus.FAILED,
                        strategy=request.strategy,
                        message="当前地图中未找到可用探索候选点。",
                    )
                return

            total = len(candidate_poses)
            completed = 0
            frontier_count = total
            for index, pose in enumerate(candidate_poses, start=1):
                if self._exploration_stop_event.is_set():
                    break
                goal = NavigationGoal(
                    goal_id=f"{request.request_id}_candidate_{index}",
                    target_pose=pose,
                    target_name=request.target_name,
                    metadata={"exploration_request_id": request.request_id, "candidate_index": index},
                )
                if not self.set_goal(goal):
                    continue
                success = self._wait_navigation_goal(goal_timeout_sec=self.config.exploration.goal_timeout_sec)
                if success:
                    completed += 1
                frontier_count = max(0, total - index)
                with self._exploration_lock:
                    self._exploration_state = ExplorationState(
                        current_request_id=request.request_id,
                        status=ExplorationStatus.RUNNING,
                        strategy=request.strategy,
                        covered_ratio=float(completed) / float(total),
                        frontier_count=frontier_count,
                        message=f"探索进行中，已完成 {completed}/{total} 个候选点。",
                    )
            with self._exploration_lock:
                if self._exploration_stop_event.is_set():
                    self._exploration_state = ExplorationState(
                        current_request_id=request.request_id,
                        status=ExplorationStatus.CANCELLED,
                        strategy=request.strategy,
                        covered_ratio=float(completed) / float(total),
                        frontier_count=frontier_count,
                        message="探索任务已取消。",
                    )
                else:
                    self._exploration_state = ExplorationState(
                        current_request_id=request.request_id,
                        status=ExplorationStatus.SUCCEEDED,
                        strategy=request.strategy,
                        covered_ratio=float(completed) / float(total),
                        frontier_count=0,
                        message=f"探索任务完成，共访问 {completed}/{total} 个候选点。",
                    )
        except Exception:
            LOGGER.exception("执行 Go2 探索任务失败。")
            with self._exploration_lock:
                self._exploration_state = ExplorationState(
                    current_request_id=request.request_id,
                    status=ExplorationStatus.FAILED,
                    strategy=request.strategy,
                    message="探索任务执行异常。",
                )
        finally:
            with self._exploration_lock:
                self._exploration_thread = None

    def _build_exploration_candidates(self, request: ExploreAreaRequest) -> List[Pose]:
        center_pose = request.center_pose
        if center_pose is None and request.target_name:
            semantic_map = self.get_semantic_map()
            if semantic_map is not None:
                normalized = request.target_name.strip().lower()
                for region in semantic_map.regions:
                    aliases = [str(region.attributes.get("alias", ""))]
                    aliases.extend(str(item) for item in region.attributes.get("aliases", []) if item)
                    names = {region.label.strip().lower(), region.region_id.strip().lower()}
                    names.update(item.strip().lower() for item in aliases if item.strip())
                    if normalized in names and region.centroid is not None:
                        center_pose = region.centroid
                        break
        if center_pose is None:
            center_pose = self.get_current_pose()
        if center_pose is None:
            return []

        radius = request.radius_m or self.config.exploration.sample_radius_m
        sample_count = self.config.exploration.sample_count
        cost_map = self.get_cost_map()
        candidates: List[Pose] = []
        for index in range(sample_count):
            angle = (2.0 * math.pi * float(index)) / float(sample_count)
            x = center_pose.position.x + radius * math.cos(angle)
            y = center_pose.position.y + radius * math.sin(angle)
            if cost_map is not None:
                cost = self._sample_cost(cost_map, x=x, y=y)
                if cost is None or cost > self.config.exploration.max_goal_cost:
                    continue
            candidates.append(
                Pose(
                    frame_id=center_pose.frame_id,
                    position=Vector3(x=x, y=y, z=center_pose.position.z),
                    orientation=Quaternion(
                        x=center_pose.orientation.x,
                        y=center_pose.orientation.y,
                        z=center_pose.orientation.z,
                        w=center_pose.orientation.w,
                    ),
                )
            )
        return candidates

    def _sample_cost(self, cost_map: CostMap, *, x: float, y: float) -> Optional[float]:
        resolution = cost_map.resolution_m
        origin_x = cost_map.origin.position.x
        origin_y = cost_map.origin.position.y
        col = int((x - origin_x) / resolution)
        row = int((y - origin_y) / resolution)
        if col < 0 or row < 0 or col >= cost_map.width or row >= cost_map.height:
            return None
        index = row * cost_map.width + col
        if index < 0 or index >= len(cost_map.data):
            return None
        return float(cost_map.data[index])

    def _wait_navigation_goal(self, *, goal_timeout_sec: float) -> bool:
        deadline = time.time() + max(0.1, goal_timeout_sec)
        while time.time() < deadline:
            if self._exploration_stop_event.is_set():
                self.cancel_goal()
                return False
            state = self.get_navigation_state()
            if self.is_goal_reached() or state.status == NavigationStatus.SUCCEEDED:
                return True
            if state.status in {NavigationStatus.FAILED, NavigationStatus.CANCELLED}:
                return False
            time.sleep(max(0.05, self.config.exploration.status_poll_interval_sec))
        self.cancel_goal()
        return False
