from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
import importlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from contracts.geometry import FrameTree, Pose, Quaternion, Transform, Vector3
from contracts.maps import CostMap, OccupancyGrid, SemanticMap, SemanticRegion
from contracts.navigation import ExplorationState, ExplorationStatus, ExploreAreaRequest, NavigationGoal, NavigationState, NavigationStatus
from contracts.robot_state import IMUState
from drivers.robots.go2.frontier_exploration import Go2FrontierExplorer
from drivers.robots.go2.global_map import Go2GlobalMapBuildResult, Go2SparseGlobalMapBuilder
from drivers.robots.go2.navigation_backend import Go2GridNavigationPlanner, Go2NavigationSession
from drivers.robots.go2.collision_detector import Go2CollisionDetector, Go2CollisionDetectorConfig, CollisionStatus
from drivers.robots.go2.cost_mapper import Go2CostMapper, Go2CostMapperConfig, Go2CostMapperResult
from drivers.robots.go2.voxel_mapper import Go2VoxelMapper, Go2VoxelMapperConfig, Go2VoxelMapperResult
from drivers.robots.go2.local_planner import Go2DWALocalPlanner, Go2LocalPlannerConfig, Go2VelocityCommand, LocalPlannerState

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

_POINT_FIELD_DTYPE_MAP = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


@dataclass
class LocalMapScan:
    """Go2 端侧局部地图使用的点云扫描。"""

    stamp_sec: float
    sensor_x: float
    sensor_y: float
    source_label: str
    points_xyz_world: np.ndarray


def _prepend_env_path(name: str, value: str) -> None:
    """把路径前置到指定环境变量。"""

    normalized = str(value or "").strip()
    if not normalized:
        return
    current = str(os.environ.get(name, "") or "").strip()
    entries = [item for item in current.split(":") if item.strip()]
    if normalized in entries:
        entries = [item for item in entries if item != normalized]
    os.environ[name] = ":".join([normalized, *entries]) if entries else normalized


def _prepare_go2_official_sdk_environment(config: "Go2DataPlaneConfig") -> None:
    """准备 unitree_sdk2py 导入所需的 Go2 官方运行环境。"""

    sdk_path = str(getattr(config, "sdk_path", "") or "").strip()
    if sdk_path:
        candidate = Path(sdk_path).expanduser()
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    cyclonedds_lib_dir = str(getattr(config, "cyclonedds_lib_dir", "") or "").strip()
    if cyclonedds_lib_dir:
        _prepend_env_path("LD_LIBRARY_PATH", cyclonedds_lib_dir)

    importlib.invalidate_caches()


def _is_ros_environment_active() -> bool:
    """判断当前进程是否已经进入 ROS2 环境。"""

    return any(
        str(os.environ.get(name, "") or "").strip()
        for name in ("ROS_VERSION", "AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH", "ROS_DISTRO")
    )


def _uses_sdk2_in_process(config: "Go2DataPlaneConfig") -> bool:
    """判断当前进程是否计划启用 Unitree SDK2。"""

    direct_dds_enabled = bool(getattr(getattr(config, "direct_dds", None), "enabled", False))
    official_backend_enabled = bool(getattr(getattr(config, "official", None), "enabled", False))
    return direct_dds_enabled or official_backend_enabled


def _sdk2_ros_conflict_message() -> str:
    """返回 SDK2 与 ROS2 进程冲突说明。"""

    return (
        "根据宇树官方约束，Unitree SDK2 与 ROS2 环境不能在同一进程中同时初始化；"
        "请在未进入 ROS2 环境的终端中启动宿主机网关，ROS2 能力需要独立进程侧车。"
    )


def _load_go2_official_sdk_modules(
    config: "Go2DataPlaneConfig",
    *,
    include_dds_topics: bool = False,
) -> Optional[Dict[str, object]]:
    """加载 Go2 官方 SDK 模块。"""

    _prepare_go2_official_sdk_environment(config)
    try:
        modules: Dict[str, object] = {
            "ChannelFactoryInitialize": getattr(
                importlib.import_module("unitree_sdk2py.core.channel"),
                "ChannelFactoryInitialize",
            ),
            "SportClient": getattr(
                importlib.import_module("unitree_sdk2py.go2.sport.sport_client"),
                "SportClient",
            ),
            "RobotStateClient": getattr(
                importlib.import_module("unitree_sdk2py.go2.robot_state.robot_state_client"),
                "RobotStateClient",
            ),
            "ObstaclesAvoidClient": getattr(
                importlib.import_module("unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client"),
                "ObstaclesAvoidClient",
            ),
        }
        if include_dds_topics:
            channel_module = importlib.import_module("unitree_sdk2py.core.channel")
            modules["ChannelSubscriber"] = getattr(channel_module, "ChannelSubscriber")
            modules["SportModeState"] = getattr(
                importlib.import_module("unitree_sdk2py.idl.unitree_go.msg.dds_"),
                "SportModeState_",
            )
            modules["DirectPointCloud2"] = getattr(
                importlib.import_module("unitree_sdk2py.idl.sensor_msgs.msg.dds_"),
                "PointCloud2_",
            )
        return modules
    except Exception as exc:
        LOGGER.warning("导入 Go2 官方 SDK 模块失败：%s", exc)
        return None


@dataclass
class ManagedRos2Process:
    """受控的 ROS2 子进程。"""

    name: str
    command: str
    setup_script: str = ""
    cyclonedds_lib_dir: str = ""
    process: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)

    def start(self) -> None:
        """启动子进程。"""

        if self.process is not None and self.process.poll() is None:
            return
        shell_parts: List[str] = []
        if self.setup_script.strip():
            shell_parts.append(f"source {self.setup_script}")
        if self.cyclonedds_lib_dir.strip():
            quoted_lib_dir = shlex.quote(self.cyclonedds_lib_dir)
            shell_parts.append(f"export LD_LIBRARY_PATH={quoted_lib_dir}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}")
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
    """Go2 端侧数据桥。

    优先直接消费宇树官方 DDS 位姿与点云，再用 ROS2 话题补齐 TF、地图层和兼容输入。
    """

    def __init__(self, config: "Go2DataPlaneConfig") -> None:
        self.config = config
        self._lock = threading.RLock()
        self._started = False
        self._owns_rclpy_context = False
        self._rclpy = None
        self._node = None
        self._executor = None
        self._executor_thread: Optional[threading.Thread] = None
        self._ros_started = False
        self._ros_in_process_disabled_reason = ""
        self._direct_dds_started = False
        self._direct_dds_stop_event = threading.Event()
        self._direct_dds_threads: List[threading.Thread] = []
        self._direct_pose_subscriber = None
        self._direct_point_cloud_subscriber = None
        self._action_client = None
        self._goal_handle = None
        self._current_goal: Optional[NavigationGoal] = None
        self._latest_pose: Optional[Pose] = None
        self._latest_pose_from_ros: Optional[Pose] = None
        self._latest_pose_from_dds: Optional[Pose] = None
        self._latest_pose_source = ""
        self._latest_imu_state: Optional[IMUState] = None
        self._latest_imu_state_from_ros: Optional[IMUState] = None
        self._latest_imu_state_from_dds: Optional[IMUState] = None
        self._tf_transforms: Dict[Tuple[str, str], Transform] = {}
        self._latest_occupancy_direct: Optional[OccupancyGrid] = None
        self._latest_occupancy_from_grid: Optional[OccupancyGrid] = None
        self._latest_occupancy_from_global_cloud: Optional[OccupancyGrid] = None
        self._latest_occupancy_from_local_cloud: Optional[OccupancyGrid] = None
        self._latest_cost_direct: Optional[CostMap] = None
        self._latest_cost_from_grid: Optional[CostMap] = None
        self._latest_cost_from_global_cloud: Optional[CostMap] = None
        self._latest_cost_from_local_cloud: Optional[CostMap] = None
        self._latest_semantic_map: Optional[SemanticMap] = None
        self._latest_semantic_map_from_global_cloud: Optional[SemanticMap] = None
        self._latest_semantic_map_from_local_cloud: Optional[SemanticMap] = None
        self._latest_navigation_state = NavigationState(status=NavigationStatus.IDLE)
        self._goal_reached = False
        self._last_error: Optional[str] = None
        self._ros_setup_loaded = False
        self._latest_odom_source_topic = ""
        self._latest_imu_source_topic = ""
        self._latest_point_cloud_source_topic = ""
        self._latest_global_map_source = ""
        self._latest_local_map_source = ""
        self._point_cloud_available = False
        self._direct_pose_ready = False
        self._ros_pose_ready = False
        self._direct_imu_ready = False
        self._ros_imu_ready = False
        self._direct_point_cloud_ready = False
        self._ros_point_cloud_ready = False
        self._ros_occupancy_ready = False
        self._ros_grid_map_ready = False
        self._ros_cost_map_ready = False
        self._global_map_ready = False
        self._local_map_ready = False
        self._global_map_builder = Go2SparseGlobalMapBuilder(self.config.map_synthesis)
        self._local_map_scans: Deque[LocalMapScan] = deque()
        self._last_global_map_update_monotonic = 0.0
        self._last_local_map_update_monotonic = 0.0

        self._collision_detector = Go2CollisionDetector(self.config.collision_detector)
        self._cost_mapper = Go2CostMapper(self.config.cost_mapper)
        self._voxel_mapper = Go2VoxelMapper(self.config.voxel_mapper)
        self._local_planner = Go2DWALocalPlanner(self.config.local_planning)
        self._frontier_explorer: Optional[Go2FrontierExplorer] = None
        self._last_collision_status = CollisionStatus.SAFE
        self._last_local_velocity_cmd: Optional[Go2VelocityCommand] = None

    def start(self) -> None:
        """启动 Go2 数据桥。"""

        with self._lock:
            if self._started:
                return
            if _uses_sdk2_in_process(self.config) and _is_ros_environment_active():
                message = _sdk2_ros_conflict_message()
                self._last_error = message
                raise RuntimeError(message)
            self._start_direct_dds_backend()

            modules = None
            if _uses_sdk2_in_process(self.config):
                self._ros_in_process_disabled_reason = (
                    "当前进程已启用 Unitree SDK2，已按官方约束禁用进程内 ROS2 订阅。"
                )
                if self.config.require_ros2:
                    self._last_error = self._ros_in_process_disabled_reason
                    raise RuntimeError(self._ros_in_process_disabled_reason)
                LOGGER.info(self._ros_in_process_disabled_reason)
            else:
                self._ros_in_process_disabled_reason = ""
                modules = self._load_ros_modules()

            if modules is not None:
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
                self._executor_thread = threading.Thread(
                    target=self._executor.spin,
                    name="nuwax_go2_ros2_executor",
                    daemon=True,
                )
                self._executor_thread.start()
                self._ros_started = True
            elif self.config.require_ros2 and not self._direct_dds_started:
                message = "当前环境不可导入 rclpy 或 ROS2 消息包，且 Go2 官方 DDS 直连未就绪。"
                self._last_error = message
                raise RuntimeError(message)
            elif not self._direct_dds_started:
                message = "当前环境不可导入 rclpy 或 ROS2 消息包，Go2 数据桥保持未启用。"
                self._last_error = message
                LOGGER.warning(message)
                return

            self._started = True
            self._last_error = None
            LOGGER.info(
                "Go2 数据桥已启动 direct_dds=%s ros2=%s",
                self._direct_dds_started,
                self._ros_started,
            )

    def stop(self) -> None:
        """停止 Go2 数据桥。"""

        with self._lock:
            if not self._started:
                return
            self._stop_direct_dds_backend()
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
            self._ros_started = False
            self._started = False
            LOGGER.info("Go2 数据桥已停止。")

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
            or self._latest_occupancy_from_global_cloud is not None
            or self._latest_occupancy_from_local_cloud is not None
            or self._latest_cost_direct is not None
            or self._latest_cost_from_grid is not None
            or self._latest_cost_from_global_cloud is not None
            or self._latest_cost_from_local_cloud is not None
            or self._latest_semantic_map is not None
            or self._latest_semantic_map_from_global_cloud is not None
            or self._latest_semantic_map_from_local_cloud is not None
        )

    def is_navigation_available(self) -> bool:
        """桥本身不再提供导航控制能力。"""

        return False

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

    def get_imu_state(self) -> Optional[IMUState]:
        """读取当前 IMU 状态。"""

        with self._lock:
            return self._latest_imu_state.model_copy(deep=True) if self._latest_imu_state is not None else None

    def get_occupancy_grid(self) -> Optional[OccupancyGrid]:
        """读取占据栅格地图。"""

        with self._lock:
            grid = (
                self._latest_occupancy_direct
                or self._latest_occupancy_from_grid
                or self._latest_occupancy_from_global_cloud
                or self._latest_occupancy_from_local_cloud
            )
            return grid.model_copy(deep=True) if grid is not None else None

    def get_cost_map(self) -> Optional[CostMap]:
        """读取代价地图。"""

        with self._lock:
            cost_map = (
                self._latest_cost_direct
                or self._latest_cost_from_grid
                or self._latest_cost_from_global_cloud
                or self._latest_cost_from_local_cloud
            )
            return cost_map.model_copy(deep=True) if cost_map is not None else None

    def get_collision_status(
        self,
        current_pose: Pose,
        scan_ranges: Optional[np.ndarray] = None,
    ) -> CollisionStatus:
        """实时碰撞检测。"""
        return self._collision_detector.check_collision(current_pose, scan_ranges)

    def process_pointcloud_for_costmap(
        self,
        points_xyz: np.ndarray,
        sensor_x: float,
        sensor_y: float,
    ) -> Optional[Go2CostMapperResult]:
        """将3D点云转换为2D代价地图。"""
        return self._cost_mapper.process_pointcloud(points_xyz, sensor_x, sensor_y)

    def add_pointcloud_to_voxel_map(
        self,
        points_xyz: np.ndarray,
        timestamp: float = 0.0,
    ) -> Optional[Go2VoxelMapperResult]:
        """添加一帧点云到体素地图。"""
        return self._voxel_mapper.add_frame(points_xyz, timestamp)

    def compute_local_velocity(
        self,
        current_pose: Pose,
        global_plan: List[Tuple[float, float]],
        cost_map: Optional[np.ndarray] = None,
        scan_ranges: Optional[np.ndarray] = None,
    ) -> Go2VelocityCommand:
        """用DWA计算局部动态避障速度。"""
        return self._local_planner.compute_velocity(
            current_pose, global_plan, cost_map, scan_ranges
        )

    def check_exploration_safety(
        self,
        current_pose: Pose,
        scan_ranges: Optional[np.ndarray] = None,
    ):
        """检测探索期间的安全性。"""
        if self._frontier_explorer is None:
            from drivers.robots.go2.settings import Go2ExplorationConfig
            self._frontier_explorer = Go2FrontierExplorer(Go2ExplorationConfig())
        return self._frontier_explorer.check_exploration_safety(current_pose, scan_ranges)

    def get_voxel_map_pointcloud(self) -> Optional[np.ndarray]:
        """获取体素地图生成的全局点云。"""
        return self._voxel_mapper.get_global_pointcloud()

    def get_voxel_map_count(self) -> int:
        """获取当前体素数量。"""
        return self._voxel_mapper.get_voxel_count()

    def get_local_planner_state(self) -> LocalPlannerState:
        """获取局部规划器状态。"""
        return self._local_planner.state

    def reset_local_planner(self) -> None:
        """重置局部规划器。"""
        self._local_planner.reset()

    def get_semantic_map(self) -> Optional[SemanticMap]:
        """读取语义地图。"""

        with self._lock:
            semantic_map = (
                self._latest_semantic_map
                or self._latest_semantic_map_from_global_cloud
                or self._latest_semantic_map_from_local_cloud
            )
            return semantic_map.model_copy(deep=True) if semantic_map is not None else None

    def set_goal(self, goal: NavigationGoal) -> bool:
        """桥本身不再处理导航目标。"""

        del goal
        return False

    def cancel_goal(self) -> bool:
        """桥本身不再处理导航取消。"""

        return False

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
                "ros_started": self._ros_started,
                "ros_in_process_disabled_reason": self._ros_in_process_disabled_reason,
                "direct_dds_started": self._direct_dds_started,
                "localization_available": self.is_localization_available(),
                "map_available": self.is_map_available(),
                "navigation_available": self.is_navigation_available(),
                "pose_source": self._latest_pose_source,
                "dds_pose_ready": self._direct_pose_ready,
                "ros_pose_ready": self._ros_pose_ready,
                "odom_source_topic": self._latest_odom_source_topic,
                "imu_available": self._latest_imu_state is not None,
                "dds_imu_ready": self._direct_imu_ready,
                "ros_imu_ready": self._ros_imu_ready,
                "imu_source_topic": self._latest_imu_source_topic,
                "dds_cloud_ready": self._direct_point_cloud_ready,
                "ros_point_cloud_ready": self._ros_point_cloud_ready,
                "point_cloud_source_topic": self._latest_point_cloud_source_topic,
                "point_cloud_available": self._point_cloud_available,
                "ros_map_ready": self._ros_occupancy_ready or self._ros_grid_map_ready or self._ros_cost_map_ready,
                "global_map_ready": self._global_map_ready,
                "global_map_source": self._latest_global_map_source,
                "local_map_ready": self._local_map_ready,
                "local_map_source": self._latest_local_map_source,
                "last_error": self._last_error,
            }

    def _load_ros_modules(self) -> Optional[Dict[str, object]]:
        self._ensure_ros_python_environment()
        try:
            modules = {
                "rclpy": importlib.import_module("rclpy"),
                "Node": getattr(importlib.import_module("rclpy.node"), "Node"),
                "MultiThreadedExecutor": getattr(importlib.import_module("rclpy.executors"), "MultiThreadedExecutor"),
                "Odometry": getattr(importlib.import_module("nav_msgs.msg"), "Odometry"),
                "RosOccupancyGrid": getattr(importlib.import_module("nav_msgs.msg"), "OccupancyGrid"),
                "TFMessage": getattr(importlib.import_module("tf2_msgs.msg"), "TFMessage"),
                "Imu": getattr(importlib.import_module("sensor_msgs.msg"), "Imu"),
                "PointCloud2": getattr(importlib.import_module("sensor_msgs.msg"), "PointCloud2"),
            }
            try:
                modules["GridMap"] = getattr(importlib.import_module("grid_map_msgs.msg"), "GridMap")
            except Exception:
                modules["GridMap"] = None
            try:
                modules["Nav2Costmap"] = getattr(importlib.import_module("nav2_msgs.msg"), "Costmap")
            except Exception:
                modules["Nav2Costmap"] = None
            return modules
        except Exception as exc:
            if self.config.require_ros2 or not self._direct_dds_started:
                LOGGER.exception("导入 ROS2 Python 模块失败。")
            else:
                LOGGER.warning("Go2 端侧 ROS2 Python 环境当前不可用，继续使用官方 DDS 直连：%s", exc)
                LOGGER.debug("导入 ROS2 Python 模块失败详情。", exc_info=True)
            return None

    def _ensure_ros_python_environment(self) -> None:
        """把 setup.bash 导出的环境同步到当前 Python 进程。"""

        if self._ros_setup_loaded:
            return
        if importlib.util.find_spec("rclpy") is not None:
            self._ros_setup_loaded = True
            return

        setup_script = str(self.config.setup_script or "").strip()
        if not setup_script:
            LOGGER.warning("Go2 ROS2 setup 脚本未配置，无法自动导入 ROS2 Python 环境。")
            return

        setup_path = Path(setup_script).expanduser()
        if not setup_path.exists():
            LOGGER.warning("Go2 ROS2 setup 脚本不存在，无法自动导入 ROS2 Python 环境：%s", setup_path)
            return

        try:
            completed = subprocess.run(
                [
                    "bash",
                    "-lc",
                    f"source {shlex.quote(str(setup_path))} >/dev/null 2>&1 && env -0",
                ],
                check=True,
                capture_output=True,
                env=dict(os.environ),
            )
        except Exception:
            LOGGER.exception("执行 Go2 ROS2 setup 脚本失败：%s", setup_path)
            return

        loaded_env = self._parse_null_delimited_environment(completed.stdout)
        if not loaded_env:
            LOGGER.warning("Go2 ROS2 setup 脚本未导出可用环境：%s", setup_path)
            return

        self._merge_ros_environment(loaded_env)
        importlib.invalidate_caches()
        self._ros_setup_loaded = True
        LOGGER.info("已自动加载 Go2 ROS2 Python 环境 setup=%s", setup_path)

    def _parse_null_delimited_environment(self, payload: bytes) -> Dict[str, str]:
        """把 env -0 输出解析成环境变量字典。"""

        result: Dict[str, str] = {}
        for raw_item in payload.split(b"\x00"):
            if not raw_item or b"=" not in raw_item:
                continue
            key_bytes, value_bytes = raw_item.split(b"=", 1)
            key = key_bytes.decode("utf-8", errors="ignore").strip()
            if not key:
                continue
            result[key] = value_bytes.decode("utf-8", errors="ignore")
        return result

    def _merge_ros_environment(self, loaded_env: Dict[str, str]) -> None:
        """把 ROS2 相关环境注入当前进程，并同步 PYTHONPATH 到 sys.path。"""

        for key, value in loaded_env.items():
            os.environ[key] = value

        python_path = str(loaded_env.get("PYTHONPATH", "") or "").strip()
        if not python_path:
            return

        inserted_paths: List[str] = []
        for raw_item in python_path.split(":"):
            item = raw_item.strip()
            if not item or item in inserted_paths:
                continue
            inserted_paths.append(item)

        for item in reversed(inserted_paths):
            if item not in sys.path:
                sys.path.insert(0, item)

    def _start_direct_dds_backend(self) -> None:
        """启动 Go2 官方 DDS 原始数据订阅。"""

        if not getattr(self.config, "direct_dds", None) or not self.config.direct_dds.enabled:
            return
        modules = _load_go2_official_sdk_modules(self.config, include_dds_topics=True)
        if modules is None:
            return

        channel_factory_initialize = modules["ChannelFactoryInitialize"]
        iface = str(self.config.dds_iface or "").strip()
        try:
            if iface:
                channel_factory_initialize(0, iface)
            else:
                channel_factory_initialize(0)
        except TypeError:
            channel_factory_initialize(0)
        except Exception as exc:
            LOGGER.warning("初始化 Go2 官方 DDS 直连失败：%s", exc)
            return

        try:
            subscriber_class = modules["ChannelSubscriber"]
            self._direct_pose_subscriber = subscriber_class(
                self.config.direct_dds.pose_topic,
                modules["SportModeState"],
            )
            self._direct_pose_subscriber.Init()
            self._direct_point_cloud_subscriber = subscriber_class(
                self.config.direct_dds.point_cloud_topic,
                modules["DirectPointCloud2"],
            )
            self._direct_point_cloud_subscriber.Init()
        except Exception as exc:
            LOGGER.warning("初始化 Go2 官方 DDS 订阅器失败：%s", exc)
            self._direct_pose_subscriber = None
            self._direct_point_cloud_subscriber = None
            return

        self._direct_dds_stop_event.clear()
        self._direct_dds_threads = [
            threading.Thread(
                target=self._direct_pose_reader_loop,
                name="go2_direct_pose_reader",
                daemon=True,
            ),
            threading.Thread(
                target=self._direct_point_cloud_reader_loop,
                name="go2_direct_point_cloud_reader",
                daemon=True,
            ),
        ]
        for thread in self._direct_dds_threads:
            thread.start()
        self._direct_dds_started = True

    def _stop_direct_dds_backend(self) -> None:
        """停止 Go2 官方 DDS 原始数据订阅。"""

        self._direct_dds_stop_event.set()
        for thread in self._direct_dds_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        self._direct_dds_threads = []
        for subscriber in (self._direct_pose_subscriber, self._direct_point_cloud_subscriber):
            if subscriber is None:
                continue
            try:
                subscriber.Close()
            except Exception:
                LOGGER.debug("关闭 Go2 官方 DDS 订阅器失败。", exc_info=True)
        self._direct_pose_subscriber = None
        self._direct_point_cloud_subscriber = None
        self._direct_dds_started = False

    def _direct_pose_reader_loop(self) -> None:
        timeout_sec = float(getattr(self.config.direct_dds, "read_timeout_sec", 0.5))
        source_topic = f"dds:{self.config.direct_dds.pose_topic}"
        while not self._direct_dds_stop_event.is_set():
            subscriber = self._direct_pose_subscriber
            if subscriber is None:
                return
            try:
                sample = subscriber.Read(timeout_sec)
            except Exception:
                LOGGER.debug("读取 Go2 官方 DDS 位姿失败。", exc_info=True)
                time.sleep(0.1)
                continue
            if sample is None:
                continue
            self._on_sport_mode_state(sample, source_topic=source_topic)

    def _direct_point_cloud_reader_loop(self) -> None:
        timeout_sec = float(getattr(self.config.direct_dds, "read_timeout_sec", 0.5))
        source_topic = f"dds:{self.config.direct_dds.point_cloud_topic}"
        while not self._direct_dds_stop_event.is_set():
            subscriber = self._direct_point_cloud_subscriber
            if subscriber is None:
                return
            try:
                sample = subscriber.Read(timeout_sec)
            except Exception:
                LOGGER.debug("读取 Go2 官方 DDS 点云失败。", exc_info=True)
                time.sleep(0.1)
                continue
            if sample is None:
                continue
            self._on_direct_point_cloud(sample, source_topic=source_topic)

    def _update_preferred_pose_unlocked(self) -> None:
        """按照优先级更新当前位姿。"""

        if self._latest_pose_from_dds is not None:
            self._latest_pose = self._latest_pose_from_dds
            self._latest_pose_source = f"dds:{self.config.direct_dds.pose_topic}"
            return
        self._latest_pose = self._latest_pose_from_ros
        self._latest_pose_source = self._latest_odom_source_topic

    def _update_preferred_imu_unlocked(self) -> None:
        """按照优先级更新当前 IMU 来源。"""

        if self._latest_imu_state_from_dds is not None:
            self._latest_imu_state = self._latest_imu_state_from_dds
            self._latest_imu_source_topic = f"dds:{self.config.direct_dds.pose_topic}#imu_state"
            return
        if self._latest_imu_state_from_ros is not None:
            self._latest_imu_state = self._latest_imu_state_from_ros
            self._latest_imu_source_topic = str(self.config.topics.imu_topic or "")
            return
        self._latest_imu_state = None
        self._latest_imu_source_topic = ""

    def _update_preferred_point_cloud_status_unlocked(self) -> None:
        """按照优先级更新当前点云来源状态。"""

        if self._direct_point_cloud_ready:
            self._latest_point_cloud_source_topic = f"dds:{self.config.direct_dds.point_cloud_topic}"
            self._point_cloud_available = True
            return
        if self._ros_point_cloud_ready:
            self._latest_point_cloud_source_topic = str(self.config.topics.point_cloud_topic or "")
            self._point_cloud_available = True
            return
        self._latest_point_cloud_source_topic = ""
        self._point_cloud_available = False

    def _create_subscriptions(self, modules: Dict[str, object]) -> None:
        assert self._node is not None
        topics = self.config.topics
        subscribed_topics = set()
        odom_topics = [topics.odom_topic, *list(getattr(topics, "secondary_odom_topics", ()) or ())]
        for topic_name in odom_topics:
            normalized = str(topic_name or "").strip()
            if not normalized or normalized in subscribed_topics:
                continue
            subscribed_topics.add(normalized)
            self._node.create_subscription(
                modules["Odometry"],
                normalized,
                lambda message, source_topic=normalized: self._on_odom(message, source_topic=source_topic),
                20,
            )
        self._node.create_subscription(modules["TFMessage"], topics.tf_topic, self._on_tf_message, 50)
        self._node.create_subscription(modules["TFMessage"], topics.tf_static_topic, self._on_tf_message, 10)
        if str(getattr(topics, "imu_topic", "") or "").strip():
            self._node.create_subscription(
                modules["Imu"],
                topics.imu_topic,
                lambda message, source_topic=topics.imu_topic: self._on_imu(message, source_topic=source_topic),
                20,
            )
        if str(getattr(topics, "point_cloud_topic", "") or "").strip():
            self._node.create_subscription(
                modules["PointCloud2"],
                topics.point_cloud_topic,
                lambda message, source_topic=topics.point_cloud_topic: self._on_point_cloud(
                    message, source_topic=source_topic
                ),
                5,
            )
        self._node.create_subscription(modules["RosOccupancyGrid"], topics.occupancy_topic, self._on_occupancy_grid, 10)
        if modules.get("GridMap") is not None and topics.grid_map_topic.strip():
            self._node.create_subscription(modules["GridMap"], topics.grid_map_topic, self._on_grid_map, 10)
        if modules.get("Nav2Costmap") is not None and topics.cost_map_topic.strip():
            self._node.create_subscription(modules["Nav2Costmap"], topics.cost_map_topic, self._on_cost_map, 10)

    def _on_odom(self, message: object, *, source_topic: str = "") -> None:
        pose = self._build_pose_from_odom(message)
        with self._lock:
            self._latest_pose_from_ros = pose
            self._ros_pose_ready = True
            self._latest_odom_source_topic = str(source_topic or "")
            self._update_preferred_pose_unlocked()

    def _on_imu(self, message: object, *, source_topic: str = "") -> None:
        imu_state = self._build_imu_from_ros(message)
        with self._lock:
            self._latest_imu_state_from_ros = imu_state
            self._ros_imu_ready = imu_state is not None
            if not str(self.config.topics.imu_topic or "").strip():
                self._latest_imu_source_topic = str(source_topic or "")
            self._update_preferred_imu_unlocked()

    def _is_point_cloud_mapping_enabled(self) -> bool:
        """判断当前是否启用了任意点云地图合成链。"""

        return bool(
            self.config.map_synthesis.local_map_enabled
            or self.config.map_synthesis.global_map_enabled
        )

    def _on_point_cloud(self, message: object, *, source_topic: str = "") -> None:
        with self._lock:
            current_pose = self._latest_pose.model_copy(deep=True) if self._latest_pose is not None else None
            self._ros_point_cloud_ready = True
            self._update_preferred_point_cloud_status_unlocked()
            should_ingest = not self._direct_dds_started or not self._direct_point_cloud_ready
        if current_pose is None or not should_ingest or not self._is_point_cloud_mapping_enabled():
            return
        self._ingest_point_cloud_sample(
            message,
            current_pose=current_pose,
            source_label="ros_point_cloud",
        )

    def _on_sport_mode_state(self, message: object, *, source_topic: str = "") -> None:
        pose = self._build_pose_from_sport_mode_state(message)
        imu_state = self._build_imu_from_sport_mode_state(message)
        with self._lock:
            self._latest_pose_from_dds = pose
            self._direct_pose_ready = True
            self._latest_odom_source_topic = str(source_topic or "")
            self._latest_imu_state_from_dds = imu_state
            self._direct_imu_ready = imu_state is not None
            self._update_preferred_pose_unlocked()
            self._update_preferred_imu_unlocked()

    def _on_direct_point_cloud(self, message: object, *, source_topic: str = "") -> None:
        with self._lock:
            current_pose = self._latest_pose.model_copy(deep=True) if self._latest_pose is not None else None
            self._direct_point_cloud_ready = True
            self._update_preferred_point_cloud_status_unlocked()
        if current_pose is None or not self._is_point_cloud_mapping_enabled():
            return
        self._ingest_point_cloud_sample(
            message,
            current_pose=current_pose,
            source_label="direct_dds_point_cloud",
        )

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
            self._ros_occupancy_ready = True

    def _on_cost_map(self, message: object) -> None:
        with self._lock:
            self._latest_cost_direct = self._build_contract_cost_map(message)
            self._ros_cost_map_ready = True

    def _on_grid_map(self, message: object) -> None:
        occupancy = self._build_occupancy_from_grid_map(message)
        cost_map = self._build_cost_map_from_grid_map(message)
        semantic_map = self._build_semantic_map_from_grid_map(message)
        with self._lock:
            self._latest_occupancy_from_grid = occupancy
            self._latest_cost_from_grid = cost_map
            self._latest_semantic_map = semantic_map
            self._ros_grid_map_ready = bool(occupancy is not None or cost_map is not None or semantic_map is not None)

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

    def _build_imu_from_ros(self, message: object) -> IMUState:
        header = getattr(message, "header", None)
        frame_id = str(getattr(header, "frame_id", "") or "imu")
        orientation = getattr(message, "orientation", None)
        angular_velocity = getattr(message, "angular_velocity", None)
        linear_acceleration = getattr(message, "linear_acceleration", None)
        return IMUState(
            frame_id=frame_id,
            orientation=Quaternion(
                x=float(getattr(orientation, "x", 0.0)),
                y=float(getattr(orientation, "y", 0.0)),
                z=float(getattr(orientation, "z", 0.0)),
                w=float(getattr(orientation, "w", 1.0)),
            ),
            angular_velocity_rad_s=Vector3(
                x=float(getattr(angular_velocity, "x", 0.0)),
                y=float(getattr(angular_velocity, "y", 0.0)),
                z=float(getattr(angular_velocity, "z", 0.0)),
            ),
            linear_acceleration_m_s2=Vector3(
                x=float(getattr(linear_acceleration, "x", 0.0)),
                y=float(getattr(linear_acceleration, "y", 0.0)),
                z=float(getattr(linear_acceleration, "z", 0.0)),
            ),
        )

    def _build_pose_from_sport_mode_state(self, message: object) -> Pose:
        imu_state = getattr(message, "imu_state", None)
        quaternion = list(getattr(imu_state, "quaternion", [0.0, 0.0, 0.0, 1.0]) or [0.0, 0.0, 0.0, 1.0])
        while len(quaternion) < 4:
            quaternion.append(0.0)
        if abs(float(quaternion[3])) < 1e-6 and all(abs(float(item)) < 1e-6 for item in quaternion[:3]):
            quaternion = [0.0, 0.0, 0.0, 1.0]

        position = list(getattr(message, "position", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
        while len(position) < 3:
            position.append(0.0)

        return Pose(
            frame_id=self.config.map_synthesis.map_frame,
            position=Vector3(
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]),
            ),
            orientation=Quaternion(
                x=float(quaternion[0]),
                y=float(quaternion[1]),
                z=float(quaternion[2]),
                w=float(quaternion[3]),
            ),
        )

    def _build_imu_from_sport_mode_state(self, message: object) -> IMUState:
        imu_state = getattr(message, "imu_state", None)
        quaternion = list(getattr(imu_state, "quaternion", [0.0, 0.0, 0.0, 1.0]) or [0.0, 0.0, 0.0, 1.0])
        gyroscope = list(getattr(imu_state, "gyroscope", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
        accelerometer = list(getattr(imu_state, "accelerometer", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
        while len(quaternion) < 4:
            quaternion.append(0.0)
        while len(gyroscope) < 3:
            gyroscope.append(0.0)
        while len(accelerometer) < 3:
            accelerometer.append(0.0)
        if abs(float(quaternion[3])) < 1e-6 and all(abs(float(item)) < 1e-6 for item in quaternion[:3]):
            quaternion = [0.0, 0.0, 0.0, 1.0]

        return IMUState(
            frame_id=str(self.config.map_synthesis.base_frame or "body"),
            orientation=Quaternion(
                x=float(quaternion[0]),
                y=float(quaternion[1]),
                z=float(quaternion[2]),
                w=float(quaternion[3]),
            ),
            angular_velocity_rad_s=Vector3(
                x=float(gyroscope[0]),
                y=float(gyroscope[1]),
                z=float(gyroscope[2]),
            ),
            linear_acceleration_m_s2=Vector3(
                x=float(accelerometer[0]),
                y=float(accelerometer[1]),
                z=float(accelerometer[2]),
            ),
        )

    def _ingest_point_cloud_sample(
        self,
        message: object,
        *,
        current_pose: Pose,
        source_label: str,
    ) -> None:
        points_xyz = self._extract_xyz_from_point_cloud(message)
        if points_xyz.size == 0:
            return

        world_points, sensor_x, sensor_y = self._transform_point_cloud_to_map(points_xyz, current_pose=current_pose)
        if world_points.size == 0:
            return

        now_sec = time.time()
        now_mono = time.monotonic()
        with self._lock:
            if self.config.map_synthesis.global_map_enabled:
                ingested = self._global_map_builder.ingest_scan(
                    world_points=world_points,
                    sensor_x=sensor_x,
                    sensor_y=sensor_y,
                    source_label=source_label,
                )
                if ingested and (
                    now_mono - self._last_global_map_update_monotonic
                    >= self.config.map_synthesis.global_map_update_interval_sec
                ):
                    build_result = self._global_map_builder.build()
                    occupancy, cost_map, semantic_map = self._build_contract_maps_from_global_result(
                        build_result,
                        current_pose=current_pose,
                    )
                    self._latest_occupancy_from_global_cloud = occupancy
                    self._latest_cost_from_global_cloud = cost_map
                    self._latest_semantic_map_from_global_cloud = semantic_map
                    self._global_map_ready = bool(occupancy is not None or cost_map is not None or semantic_map is not None)
                    if build_result is not None:
                        self._latest_global_map_source = build_result.source_label
                    self._last_global_map_update_monotonic = now_mono

            if self.config.map_synthesis.local_map_enabled:
                self._local_map_scans.append(
                    LocalMapScan(
                        stamp_sec=now_sec,
                        sensor_x=sensor_x,
                        sensor_y=sensor_y,
                        source_label=source_label,
                        points_xyz_world=world_points,
                    )
                )
                self._prune_local_map_scans_unlocked(now_sec)
                if now_mono - self._last_local_map_update_monotonic < self.config.map_synthesis.local_map_update_interval_sec:
                    return
                occupancy, cost_map, semantic_map = self._build_local_maps_from_scans_unlocked(current_pose)
                self._latest_occupancy_from_local_cloud = occupancy
                self._latest_cost_from_local_cloud = cost_map
                self._latest_semantic_map_from_local_cloud = semantic_map
                self._local_map_ready = bool(occupancy is not None or cost_map is not None or semantic_map is not None)
                self._last_local_map_update_monotonic = now_mono

    def _extract_xyz_from_point_cloud(self, message: object) -> np.ndarray:
        fields = list(getattr(message, "fields", []) or [])
        endian_prefix = ">" if bool(getattr(message, "is_bigendian", False)) else "<"
        field_specs: Dict[str, Tuple[int, np.dtype]] = {}
        for field in fields:
            name = str(getattr(field, "name", "") or "").strip()
            if name in {"x", "y", "z"}:
                datatype = int(getattr(field, "datatype", 0) or 0)
                count = int(getattr(field, "count", 1) or 1)
                base_dtype = _POINT_FIELD_DTYPE_MAP.get(datatype)
                if base_dtype is None or count != 1:
                    continue
                field_specs[name] = (
                    int(getattr(field, "offset", 0)),
                    np.dtype(base_dtype).newbyteorder(endian_prefix),
                )
        if {"x", "y", "z"} - set(field_specs):
            return np.empty((0, 3), dtype=np.float32)

        point_step = int(getattr(message, "point_step", 0) or 0)
        payload = bytes(getattr(message, "data", b"") or b"")
        if point_step <= 0 or not payload or len(payload) < point_step:
            return np.empty((0, 3), dtype=np.float32)

        point_dtype = np.dtype(
            {
                "names": ["x", "y", "z"],
                "formats": [field_specs["x"][1], field_specs["y"][1], field_specs["z"][1]],
                "offsets": [field_specs["x"][0], field_specs["y"][0], field_specs["z"][0]],
                "itemsize": point_step,
            }
        )
        width = int(getattr(message, "width", 0) or 0)
        height = max(1, int(getattr(message, "height", 1) or 1))
        row_step = int(getattr(message, "row_step", 0) or 0)
        expected_row_bytes = point_step * width if width > 0 else 0
        if width > 0 and expected_row_bytes > 0 and row_step >= expected_row_bytes:
            row_chunks: List[np.ndarray] = []
            for row_index in range(height):
                row_start = row_index * row_step
                row_end = row_start + expected_row_bytes
                if row_end > len(payload):
                    break
                row_chunks.append(np.frombuffer(payload[row_start:row_end], dtype=point_dtype, count=width))
            points = np.concatenate(row_chunks) if row_chunks else np.empty((0,), dtype=point_dtype)
        else:
            count = len(payload) // point_step
            points = np.frombuffer(payload, dtype=point_dtype, count=count)
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        xyz = np.stack((points["x"], points["y"], points["z"]), axis=1).astype(np.float32, copy=False)
        finite_mask = np.all(np.isfinite(xyz), axis=1)
        xyz = xyz[finite_mask]
        if xyz.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        max_points = int(getattr(self.config.direct_dds, "max_points_per_scan", 1200))
        if xyz.shape[0] > max_points:
            stride = max(1, int(math.ceil(float(xyz.shape[0]) / float(max_points))))
            xyz = xyz[::stride]
        return xyz

    def _transform_point_cloud_to_map(self, points_xyz: np.ndarray, *, current_pose: Pose) -> Tuple[np.ndarray, float, float]:
        if points_xyz.size == 0:
            return np.empty((0, 3), dtype=np.float32), float(current_pose.position.x), float(current_pose.position.y)

        yaw = self._yaw_from_quaternion(current_pose.orientation)
        map_cfg = self.config.map_synthesis

        range_mask = np.hypot(points_xyz[:, 0], points_xyz[:, 1]) <= float(map_cfg.local_map_max_range_m)
        filtered = points_xyz[range_mask]
        if filtered.size == 0:
            return np.empty((0, 3), dtype=np.float32), float(current_pose.position.x), float(current_pose.position.y)

        body_cos = math.cos(yaw)
        body_sin = math.sin(yaw)
        sensor_x = float(current_pose.position.x) + body_cos * float(map_cfg.lidar_offset_x_m) - body_sin * float(
            map_cfg.lidar_offset_y_m
        )
        sensor_y = float(current_pose.position.y) + body_sin * float(map_cfg.lidar_offset_x_m) + body_cos * float(
            map_cfg.lidar_offset_y_m
        )
        sensor_z = float(current_pose.position.z) + float(map_cfg.lidar_offset_z_m)

        sensor_yaw = yaw + float(map_cfg.lidar_yaw_offset_rad)
        sensor_cos = math.cos(sensor_yaw)
        sensor_sin = math.sin(sensor_yaw)

        world_x = sensor_x + filtered[:, 0] * sensor_cos - filtered[:, 1] * sensor_sin
        world_y = sensor_y + filtered[:, 0] * sensor_sin + filtered[:, 1] * sensor_cos
        world_z = sensor_z + filtered[:, 2]

        relative_z = world_z - float(current_pose.position.z)
        z_mask = (relative_z >= float(map_cfg.local_map_min_obstacle_height_m)) & (
            relative_z <= float(map_cfg.local_map_max_obstacle_height_m)
        )
        world_points = np.stack((world_x[z_mask], world_y[z_mask], world_z[z_mask]), axis=1).astype(np.float32, copy=False)
        return world_points, sensor_x, sensor_y

    def _prune_local_map_scans_unlocked(self, now_sec: float) -> None:
        max_age_sec = float(self.config.map_synthesis.local_map_max_scan_age_sec)
        max_scans = int(self.config.map_synthesis.local_map_max_scans)
        while self._local_map_scans and (now_sec - self._local_map_scans[0].stamp_sec) > max_age_sec:
            self._local_map_scans.popleft()
        while len(self._local_map_scans) > max_scans:
            self._local_map_scans.popleft()

    def _build_local_maps_from_scans_unlocked(
        self,
        current_pose: Pose,
    ) -> Tuple[Optional[OccupancyGrid], Optional[CostMap], Optional[SemanticMap]]:
        if not self._local_map_scans:
            return None, None, None

        map_cfg = self.config.map_synthesis
        width = int(map_cfg.local_map_width)
        height = int(map_cfg.local_map_height)
        resolution = float(map_cfg.local_map_resolution_m)
        origin_x = float(current_pose.position.x) - (width * resolution) / 2.0
        origin_y = float(current_pose.position.y) - (height * resolution) / 2.0
        frame_id = current_pose.frame_id or map_cfg.map_frame
        observed = np.zeros((height, width), dtype=np.uint8)
        occupied = np.zeros((height, width), dtype=np.uint8)
        source_labels = {scan.source_label for scan in self._local_map_scans if str(scan.source_label or "").strip()}

        for scan in list(self._local_map_scans):
            sensor_cell = self._world_to_grid_cell(
                x=scan.sensor_x,
                y=scan.sensor_y,
                origin_x=origin_x,
                origin_y=origin_y,
                resolution=resolution,
                width=width,
                height=height,
            )
            if sensor_cell is None:
                continue
            sensor_row, sensor_col = sensor_cell
            for point_x, point_y, _point_z in scan.points_xyz_world:
                point_cell = self._world_to_grid_cell(
                    x=float(point_x),
                    y=float(point_y),
                    origin_x=origin_x,
                    origin_y=origin_y,
                    resolution=resolution,
                    width=width,
                    height=height,
                )
                if point_cell is None:
                    continue
                point_row, point_col = point_cell
                for free_row, free_col in self._bresenham_cells(sensor_row, sensor_col, point_row, point_col)[:-1]:
                    observed[free_row, free_col] = 1
                observed[point_row, point_col] = 1
                occupied[point_row, point_col] = 1

        if not np.any(observed):
            return None, None, None

        if not source_labels:
            local_map_source = "local_point_cloud"
        elif len(source_labels) == 1:
            local_map_source = next(iter(source_labels))
        else:
            local_map_source = "mixed_point_cloud"
        self._latest_local_map_source = local_map_source

        occupancy_data = np.full((height, width), -1, dtype=np.int32)
        occupancy_data[observed == 1] = 0
        occupancy_data[occupied == 1] = 100

        cost_data = np.full((height, width), 100.0, dtype=np.float32)
        cost_data[observed == 1] = 0.0
        cost_data[occupied == 1] = 100.0
        self._inflate_cost_map(cost_data, occupied_mask=occupied.astype(bool), resolution=resolution)

        origin_pose = Pose(
            frame_id=frame_id,
            position=Vector3(x=origin_x, y=origin_y, z=float(current_pose.position.z)),
            orientation=Quaternion(w=1.0),
        )
        occupancy_grid = OccupancyGrid(
            map_id="go2_local_cloud_occupancy",
            frame_id=frame_id,
            width=width,
            height=height,
            resolution_m=resolution,
            origin=origin_pose,
            data=[int(value) for value in occupancy_data.flatten().tolist()],
        )
        cost_map = CostMap(
            map_id="go2_local_cloud_cost",
            frame_id=frame_id,
            width=width,
            height=height,
            resolution_m=resolution,
            origin=origin_pose,
            data=[float(value) for value in cost_data.flatten().tolist()],
        )
        semantic_map = self._build_semantic_map_from_local_cloud(
            occupancy=occupancy_data,
            occupied_mask=occupied.astype(bool),
            frame_id=frame_id,
            origin=origin_pose,
            resolution=resolution,
            width=width,
            height=height,
            region_source=local_map_source,
        )
        return occupancy_grid, cost_map, semantic_map

    def _build_contract_maps_from_global_result(
        self,
        build_result: Optional[Go2GlobalMapBuildResult],
        *,
        current_pose: Pose,
    ) -> Tuple[Optional[OccupancyGrid], Optional[CostMap], Optional[SemanticMap]]:
        if build_result is None:
            return None, None, None
        frame_id = current_pose.frame_id or self.config.map_synthesis.map_frame
        origin_pose = Pose(
            frame_id=frame_id,
            position=Vector3(
                x=float(build_result.origin_x),
                y=float(build_result.origin_y),
                z=float(current_pose.position.z),
            ),
            orientation=Quaternion(w=1.0),
        )
        occupancy_grid = OccupancyGrid(
            map_id="go2_global_cloud_occupancy",
            frame_id=frame_id,
            width=int(build_result.width),
            height=int(build_result.height),
            resolution_m=float(build_result.resolution_m),
            origin=origin_pose,
            data=[int(value) for value in build_result.occupancy_data.flatten().tolist()],
        )
        cost_map = CostMap(
            map_id="go2_global_cloud_cost",
            frame_id=frame_id,
            width=int(build_result.width),
            height=int(build_result.height),
            resolution_m=float(build_result.resolution_m),
            origin=origin_pose,
            data=[float(value) for value in build_result.cost_data.flatten().tolist()],
        )
        semantic_map = self._build_semantic_map_from_global_cloud(
            occupancy=build_result.occupancy_data,
            occupied_mask=build_result.occupied_mask,
            frame_id=frame_id,
            origin=origin_pose,
            resolution=float(build_result.resolution_m),
            width=int(build_result.width),
            height=int(build_result.height),
            region_source=build_result.source_label,
        )
        return occupancy_grid, cost_map, semantic_map

    def _world_to_grid_cell(
        self,
        *,
        x: float,
        y: float,
        origin_x: float,
        origin_y: float,
        resolution: float,
        width: int,
        height: int,
    ) -> Optional[Tuple[int, int]]:
        col = int((x - origin_x) / resolution)
        row = int((y - origin_y) / resolution)
        if row < 0 or col < 0 or row >= height or col >= width:
            return None
        return row, col

    def _bresenham_cells(self, start_row: int, start_col: int, end_row: int, end_col: int) -> List[Tuple[int, int]]:
        cells: List[Tuple[int, int]] = []
        x0 = start_col
        y0 = start_row
        x1 = end_col
        y1 = end_row
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            cells.append((y0, x0))
            if x0 == x1 and y0 == y1:
                return cells
            doubled = 2 * error
            if doubled >= dy:
                error += dy
                x0 += sx
            if doubled <= dx:
                error += dx
                y0 += sy

    def _inflate_cost_map(self, cost_data: np.ndarray, *, occupied_mask: np.ndarray, resolution: float) -> None:
        radius_m = float(self.config.map_synthesis.local_map_inflation_radius_m)
        if radius_m <= 0.0 or not np.any(occupied_mask):
            return
        radius_cells = max(1, int(math.ceil(radius_m / resolution)))
        kernel: List[Tuple[int, int, float]] = []
        for row_offset in range(-radius_cells, radius_cells + 1):
            for col_offset in range(-radius_cells, radius_cells + 1):
                distance = math.hypot(float(row_offset), float(col_offset))
                if distance <= 0.0 or distance > radius_cells:
                    continue
                cost = max(1.0, 100.0 * (1.0 - (distance / float(radius_cells + 1))))
                kernel.append((row_offset, col_offset, cost))

        height, width = cost_data.shape
        occupied_rows, occupied_cols = np.where(occupied_mask)
        for center_row, center_col in zip(occupied_rows.tolist(), occupied_cols.tolist()):
            for row_offset, col_offset, inflated_cost in kernel:
                row = center_row + row_offset
                col = center_col + col_offset
                if row < 0 or col < 0 or row >= height or col >= width:
                    continue
                cost_data[row, col] = max(float(cost_data[row, col]), float(inflated_cost))

    def _build_semantic_map_from_local_cloud(
        self,
        *,
        occupancy: np.ndarray,
        occupied_mask: np.ndarray,
        frame_id: str,
        origin: Pose,
        resolution: float,
        width: int,
        height: int,
        region_source: str,
    ) -> Optional[SemanticMap]:
        return self._build_semantic_map_from_grid_data(
            occupancy=occupancy,
            occupied_mask=occupied_mask,
            frame_id=frame_id,
            origin=origin,
            resolution=resolution,
            width=width,
            height=height,
            region_source=region_source,
            map_id="go2_local_cloud_semantic_map",
        )

    def _build_semantic_map_from_global_cloud(
        self,
        *,
        occupancy: np.ndarray,
        occupied_mask: np.ndarray,
        frame_id: str,
        origin: Pose,
        resolution: float,
        width: int,
        height: int,
        region_source: str,
    ) -> Optional[SemanticMap]:
        return self._build_semantic_map_from_grid_data(
            occupancy=occupancy,
            occupied_mask=occupied_mask,
            frame_id=frame_id,
            origin=origin,
            resolution=resolution,
            width=width,
            height=height,
            region_source=region_source,
            map_id="go2_global_cloud_semantic_map",
        )

    def _build_semantic_map_from_grid_data(
        self,
        *,
        occupancy: np.ndarray,
        occupied_mask: np.ndarray,
        frame_id: str,
        origin: Pose,
        resolution: float,
        width: int,
        height: int,
        region_source: str,
        map_id: str,
    ) -> Optional[SemanticMap]:
        free_mask = np.asarray(occupancy == 0, dtype=np.uint8)
        hazard_mask = np.asarray(occupied_mask, dtype=np.uint8)
        regions: List[SemanticRegion] = []
        if np.any(free_mask):
            regions.extend(
                self._regions_from_mask(
                    mask=free_mask,
                    score_source=np.asarray(free_mask, dtype=np.float32),
                    label="traversable",
                    frame_id=frame_id,
                    width=width,
                    height=height,
                    origin=origin,
                    resolution=resolution,
                    region_source=region_source,
                )
            )
        if np.any(hazard_mask):
            regions.extend(
                self._regions_from_mask(
                    mask=hazard_mask,
                    score_source=np.asarray(hazard_mask, dtype=np.float32),
                    label="hazard",
                    frame_id=frame_id,
                    width=width,
                    height=height,
                    origin=origin,
                    resolution=resolution,
                    region_source=region_source,
                )
            )
        if not regions:
            return None
        return SemanticMap(
            map_id=map_id,
            frame_id=frame_id,
            regions=regions,
            metadata={"source": region_source},
        )

    def _yaw_from_quaternion(self, rotation: Quaternion) -> float:
        siny_cosp = 2.0 * (rotation.w * rotation.z + rotation.x * rotation.y)
        cosy_cosp = 1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z)
        return math.atan2(siny_cosp, cosy_cosp)

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
            map_id="go2_official_cost_map",
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
        region_source: str = "grid_map",
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
                            "source": region_source,
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
        self._processes: List[ManagedRos2Process] = []
        self._startup_diagnostics: List[str] = []
        self._navigation_lock = threading.RLock()
        self._exploration_lock = threading.RLock()
        self._exploration_stop_event = threading.Event()
        self._exploration_thread: Optional[threading.Thread] = None
        self._exploration_state = ExplorationState(status=ExplorationStatus.IDLE)
        self._official_sport_client = None
        self._official_robot_state_client = None
        self._official_obstacles_avoid_client = None
        self._official_sport_ready = False
        self._official_robot_state_ready = False
        self._official_obstacles_avoid_ready = False
        self._official_service_states: Dict[str, Dict[str, object]] = {}
        self._official_goal_thread: Optional[threading.Thread] = None
        self._official_goal_stop_event = threading.Event()
        self._official_navigation_goal: Optional[NavigationGoal] = None
        self._official_navigation_state = NavigationState(status=NavigationStatus.IDLE, message="官方导航后端未启用。")
        self._official_goal_reached = False
        self._official_last_error: Optional[str] = None
        self._grid_navigation_planner = Go2GridNavigationPlanner(self.config.navigation)
        self._frontier_explorer = Go2FrontierExplorer(self.config.exploration)
        self._started = False

    def start(self) -> None:
        """启动数据面运行时。"""

        if not self.config.enabled:
            LOGGER.info("Go2 数据面未启用，跳过启动。")
            return
        if self._started:
            return
        if _uses_sdk2_in_process(self.config) and _is_ros_environment_active():
            message = _sdk2_ros_conflict_message()
            self._startup_diagnostics = [message]
            raise RuntimeError(message)
        self._startup_diagnostics = []
        self._start_official_backend()
        self._processes = []
        self.bridge.start()
        self._wait_for_initial_readiness()
        self._refresh_official_navigation_idle_state()
        self._started = True

    def stop(self) -> None:
        """停止数据面运行时。"""

        self.stop_exploration()
        self._stop_official_navigation(update_state=False)
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

        return self.config.enabled and self._official_sport_ready and self.is_localization_available()

    def is_exploration_available(self) -> bool:
        """返回探索能力是否可用。"""

        if not self.config.exploration.enabled or not self.is_navigation_available():
            return False
        return self.is_localization_available() or self.is_map_available()

    def get_current_pose(self) -> Optional[Pose]:
        """读取当前位姿。"""

        return self.bridge.get_current_pose()

    def get_frame_tree(self) -> Optional[FrameTree]:
        """读取当前 TF 树。"""

        return self.bridge.get_frame_tree()

    def get_imu_state(self) -> Optional[IMUState]:
        """读取当前 IMU 状态。"""

        return self.bridge.get_imu_state()

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

        if self._uses_official_backend():
            return self._set_official_goal(goal)
        return self.bridge.set_goal(goal)

    def cancel_goal(self) -> bool:
        """取消导航目标。"""

        if self._uses_official_backend():
            return self._stop_official_navigation(update_state=True)
        return self.bridge.cancel_goal()

    def get_navigation_state(self) -> NavigationState:
        """读取导航状态。"""

        if self._uses_official_backend():
            with self._navigation_lock:
                return self._official_navigation_state.model_copy(deep=True)
        return self.bridge.get_navigation_state()

    def is_goal_reached(self) -> bool:
        """判断当前导航目标是否已完成。"""

        if self._uses_official_backend():
            with self._navigation_lock:
                return self._official_goal_reached
        return self.bridge.is_goal_reached()

    def start_exploration(self, request: ExploreAreaRequest) -> bool:
        """启动探索任务。"""

        if not self.is_exploration_available():
            with self._exploration_lock:
                self._exploration_state = ExplorationState(
                    current_request_id=request.request_id,
                    status=ExplorationStatus.FAILED,
                    strategy=request.strategy,
                    message=self._build_exploration_unavailable_message(),
                )
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

        bridge_status = self.bridge.get_status() if hasattr(self.bridge, "get_status") else {}
        if bridge_status:
            bridge_status = dict(bridge_status)
            raw_bridge_navigation_available = bool(bridge_status.get("navigation_available", False))
            bridge_status["bridge_navigation_available"] = raw_bridge_navigation_available
            bridge_status["navigation_available"] = self.is_navigation_available()
            bridge_status["navigation_availability_source"] = (
                "official_backend" if self._uses_official_backend() else "bridge"
            )

        return {
            "enabled": self.config.enabled,
            "started": self._started,
            "bridge": bridge_status,
            "localization_available": self.is_localization_available(),
            "map_available": self.is_map_available(),
            "navigation_available": self.is_navigation_available(),
            "exploration_available": self.is_exploration_available(),
            "startup_diagnostics": list(self._startup_diagnostics),
            "official_backend": {
                "enabled": self._uses_official_backend(),
                "sport_ready": self._official_sport_ready,
                "robot_state_ready": self._official_robot_state_ready,
                "obstacles_avoid_ready": self._official_obstacles_avoid_ready,
                "last_error": self._official_last_error,
                "service_states": dict(self._official_service_states),
            },
            "managed_processes": [
                {
                    "name": process.name,
                    "command": process.command,
                    "running": process.is_running(),
                }
                for process in self._processes
            ],
        }

    def _uses_official_backend(self) -> bool:
        return bool(getattr(self.config, "official", None) and self.config.official.enabled)

    def _wait_for_initial_readiness(self) -> None:
        """在启动阶段等待关键数据就绪，避免平台首轮刷新拿到空结果。"""

        if not self.config.startup_wait_for_ready:
            return
        if not self._wait_for_condition(
            condition=self.bridge.is_localization_available,
            timeout_sec=self.config.startup_localization_timeout_sec,
        ):
            message = (
                "Go2 数据面启动后在 "
                f"{self.config.startup_localization_timeout_sec:.1f} 秒内未收到定位数据。"
            )
            self._startup_diagnostics.append(message)
            LOGGER.warning(message)

        if not self._should_wait_for_map_ready():
            return
        if self._wait_for_condition(
            condition=self.bridge.is_map_available,
            timeout_sec=self.config.startup_map_timeout_sec,
        ):
            return
        message = f"Go2 数据面启动后在 {self.config.startup_map_timeout_sec:.1f} 秒内未生成地图数据。"
        self._startup_diagnostics.append(message)
        LOGGER.warning(message)

    def _should_wait_for_map_ready(self) -> bool:
        """判断启动阶段是否应该等待地图就绪。"""

        if self.config.map_synthesis.local_map_enabled:
            return True
        if str(self.config.topics.occupancy_topic or "").strip():
            return True
        if str(self.config.topics.grid_map_topic or "").strip():
            return True
        if str(self.config.topics.cost_map_topic or "").strip():
            return True
        return False

    def _wait_for_condition(self, *, condition: Any, timeout_sec: float) -> bool:
        """轮询等待指定条件成立。"""

        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        poll_interval_sec = max(0.02, float(self.config.startup_poll_interval_sec))
        while time.monotonic() <= deadline:
            try:
                if bool(condition()):
                    return True
            except Exception:
                LOGGER.debug("等待 Go2 数据面条件时发生异常。", exc_info=True)
            time.sleep(poll_interval_sec)
        try:
            return bool(condition())
        except Exception:
            LOGGER.debug("读取 Go2 数据面最终条件状态失败。", exc_info=True)
            return False

    def _refresh_official_navigation_idle_state(self) -> None:
        """把官方导航空闲态更新为当前真实后端状态。"""

        if not self._uses_official_backend():
            return
        with self._navigation_lock:
            if self._official_navigation_goal is not None:
                return
            if self._official_sport_ready:
                self._official_navigation_state = NavigationState(
                    status=NavigationStatus.IDLE,
                    current_pose=self.get_current_pose(),
                    message="Go2 工程导航后端已就绪，等待目标。",
                    metadata={"backend": "official_grid_navigation"},
                )
            else:
                self._official_navigation_state = NavigationState(
                    status=NavigationStatus.IDLE,
                    current_pose=self.get_current_pose(),
                    message="Go2 工程导航后端未就绪。",
                    metadata={"backend": "official_grid_navigation"},
                )

    def _start_official_backend(self) -> None:
        modules = self._load_official_sdk_modules()
        if modules is None:
            return

        channel_factory_initialize = modules["ChannelFactoryInitialize"]
        iface = str(self.config.dds_iface or "").strip()
        try:
            if iface:
                channel_factory_initialize(0, iface)
            else:
                channel_factory_initialize(0)
        except TypeError:
            channel_factory_initialize(0)
        except Exception:
            LOGGER.debug("初始化 Go2 官方 DDS 通道失败。", exc_info=True)

        timeout = float(self.config.official.client_timeout_sec)

        try:
            sport_client = modules["SportClient"]()
            sport_client.SetTimeout(timeout)
            sport_client.Init()
            self._official_sport_client = sport_client
            self._official_sport_ready = True
        except Exception as exc:
            self._official_last_error = f"初始化 Go2 官方运动客户端失败：{exc}"
            self._startup_diagnostics.append(self._official_last_error)
            LOGGER.warning(self._official_last_error)

        try:
            robot_state_client = modules["RobotStateClient"]()
            robot_state_client.SetTimeout(timeout)
            robot_state_client.Init()
            self._official_robot_state_client = robot_state_client
            self._official_robot_state_ready = True
        except Exception as exc:
            message = f"初始化 Go2 官方 robot_state 客户端失败：{exc}"
            self._startup_diagnostics.append(message)
            LOGGER.warning(message)

        try:
            obstacles_avoid_client = modules["ObstaclesAvoidClient"]()
            obstacles_avoid_client.SetTimeout(timeout)
            obstacles_avoid_client.Init()
            self._official_obstacles_avoid_client = obstacles_avoid_client
            version_code, version = obstacles_avoid_client.GetServerApiVersion()
            if version_code == 0:
                self._official_obstacles_avoid_ready = True
                LOGGER.info("Go2 官方 obstacles_avoid 客户端已连接 server_api_version=%s", version)
            else:
                message = f"Go2 官方 obstacles_avoid 当前未就绪 code={version_code}"
                if version_code in {3102, 3104}:
                    LOGGER.info("%s，先继续启动其余 Go2 端侧能力。", message)
                else:
                    self._startup_diagnostics.append(message)
                    LOGGER.warning(message)
        except Exception as exc:
            message = f"初始化 Go2 官方 obstacles_avoid 客户端失败：{exc}"
            self._startup_diagnostics.append(message)
            LOGGER.warning(message)

        if self._official_robot_state_ready:
            self._refresh_official_service_states()
            if self.config.official.auto_start_services:
                for service_name in self.config.official.service_names:
                    self._ensure_official_service_enabled(service_name)
                self._refresh_official_service_states()

    def _load_official_sdk_modules(self) -> Optional[Dict[str, object]]:
        modules = _load_go2_official_sdk_modules(self.config, include_dds_topics=False)
        if modules is not None:
            return modules
        message = "导入 Go2 官方 SDK 模块失败。"
        self._startup_diagnostics.append(message)
        return None

    def _refresh_official_service_states(self) -> None:
        client = self._official_robot_state_client
        if client is None:
            return
        try:
            code, items = client.ServiceList()
        except Exception:
            LOGGER.debug("读取 Go2 官方服务列表失败。", exc_info=True)
            return
        if code != 0 or not items:
            return
        service_states: Dict[str, Dict[str, object]] = {}
        for item in items:
            service_states[str(getattr(item, "name", ""))] = {
                "status": int(getattr(item, "status", -1)),
                "protect": bool(getattr(item, "protect", False)),
            }
        self._official_service_states = service_states

    def _ensure_official_service_enabled(self, service_name: str) -> None:
        normalized = str(service_name or "").strip()
        if not normalized or self._official_robot_state_client is None:
            return
        try:
            code = self._official_robot_state_client.ServiceSwitch(normalized, True)
        except Exception:
            LOGGER.debug("切换 Go2 官方服务失败 service=%s", normalized, exc_info=True)
            return
        if code != 0:
            message = f"启用 Go2 官方服务失败 service={normalized} code={code}"
            if code in {3102, 3104}:
                LOGGER.info("%s，当前按 Go2 端侧最佳努力模式继续。", message)
                return
            self._startup_diagnostics.append(message)
            LOGGER.warning(message)

    def _set_official_goal(self, goal: NavigationGoal) -> bool:
        if goal.target_pose is None or not self.is_navigation_available():
            return False

        self._stop_official_navigation(update_state=False)
        stop_event = threading.Event()
        with self._navigation_lock:
            self._official_goal_stop_event = stop_event
            self._official_navigation_goal = goal.model_copy(deep=True)
            self._official_goal_reached = False
            self._official_navigation_state = NavigationState(
                current_goal_id=goal.goal_id,
                status=NavigationStatus.ACCEPTED,
                current_pose=self.get_current_pose(),
                message="已提交 Go2 工程导航目标。",
                metadata={"backend": "official_grid_navigation"},
            )
            self._official_goal_thread = threading.Thread(
                target=self._run_official_navigation_goal,
                args=(goal.model_copy(deep=True), stop_event),
                name=f"go2_official_nav_{goal.goal_id}",
                daemon=True,
            )
            thread = self._official_goal_thread
        thread.start()
        return True

    def _stop_official_navigation(self, *, update_state: bool) -> bool:
        with self._navigation_lock:
            stop_event = self._official_goal_stop_event
            thread = self._official_goal_thread
            current_goal = self._official_navigation_goal
            stop_event.set()
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._stop_sport_motion()
        with self._navigation_lock:
            self._official_goal_thread = None
            self._official_navigation_goal = None
            if update_state:
                self._official_goal_reached = False
                self._official_navigation_state = NavigationState(
                    current_goal_id=current_goal.goal_id if current_goal is not None else None,
                    status=NavigationStatus.CANCELLED if current_goal is not None else NavigationStatus.IDLE,
                    current_pose=self.get_current_pose(),
                    message="Go2 工程导航目标已取消。" if current_goal is not None else None,
                    metadata={"backend": "official_grid_navigation"},
                )
        return current_goal is not None

    def _stop_sport_motion(self) -> None:
        if self._official_sport_client is None:
            return
        try:
            self._official_sport_client.StopMove()
        except Exception:
            LOGGER.debug("停止 Go2 官方运动失败。", exc_info=True)

    def _run_official_navigation_goal(self, goal: NavigationGoal, stop_event: threading.Event) -> None:
        try:
            current_pose = self.get_current_pose()
            if current_pose is None:
                self._set_official_navigation_failed(goal, "当前官方里程计尚未就绪。")
                return
            if goal.target_pose is None:
                self._set_official_navigation_failed(goal, "导航目标缺少目标位姿。")
                return
            if goal.target_pose.frame_id and current_pose.frame_id and goal.target_pose.frame_id != current_pose.frame_id:
                self._set_official_navigation_failed(
                    goal,
                    f"目标坐标系 {goal.target_pose.frame_id} 与当前里程计坐标系 {current_pose.frame_id} 不一致。",
                )
                return

            session = Go2NavigationSession(
                goal=goal,
                planner=self._grid_navigation_planner,
                planner_config=self.config.navigation,
                control_config=self.config.official,
            )
            with self._navigation_lock:
                self._official_navigation_state = NavigationState(
                    current_goal_id=goal.goal_id,
                    status=NavigationStatus.PLANNING,
                    current_pose=current_pose,
                    remaining_distance_m=self._compute_direct_goal_distance(
                        current_pose=current_pose,
                        target_pose=goal.target_pose,
                    ),
                    remaining_yaw_rad=self._compute_direct_goal_yaw_error(
                        current_pose=current_pose,
                        target_pose=goal.target_pose,
                    ),
                    goal_reached=False,
                    message="Go2 工程导航正在生成初始路径。",
                    metadata={"backend": "official_grid_navigation"},
                )

            while not stop_event.is_set():
                current_pose = self.get_current_pose()
                if current_pose is None:
                    time.sleep(self.config.official.control_loop_interval_sec)
                    continue
                tick_result = session.tick(
                    current_pose=current_pose,
                    occupancy_grid=self.get_occupancy_grid(),
                    cost_map=self.get_cost_map(),
                    now_monotonic=time.monotonic(),
                )
                if tick_result.status == NavigationStatus.SUCCEEDED:
                    self._stop_sport_motion()
                    with self._navigation_lock:
                        self._official_goal_reached = True
                        self._official_navigation_state = NavigationState(
                            current_goal_id=goal.goal_id,
                            status=NavigationStatus.SUCCEEDED,
                            current_pose=current_pose,
                            remaining_distance_m=0.0,
                            remaining_yaw_rad=0.0,
                            goal_reached=True,
                            message=tick_result.message,
                            metadata=tick_result.metadata,
                        )
                        self._official_goal_thread = None
                        self._official_navigation_goal = None
                    return
                if tick_result.status == NavigationStatus.FAILED:
                    self._set_official_navigation_failed(goal, tick_result.message, metadata=tick_result.metadata)
                    return
                if tick_result.status == NavigationStatus.PLANNING:
                    self._stop_sport_motion()
                    with self._navigation_lock:
                        self._official_navigation_state = NavigationState(
                            current_goal_id=goal.goal_id,
                            status=NavigationStatus.PLANNING,
                            current_pose=current_pose,
                            remaining_distance_m=tick_result.remaining_distance_m,
                            remaining_yaw_rad=tick_result.remaining_yaw_rad,
                            goal_reached=False,
                            message=tick_result.message,
                            metadata=tick_result.metadata,
                        )
                    time.sleep(self.config.official.control_loop_interval_sec)
                    continue

                code = self._official_sport_client.Move(
                    float(tick_result.linear_x_mps),
                    0.0,
                    float(tick_result.angular_z_rps),
                )
                if code != 0:
                    self._set_official_navigation_failed(goal, f"Go2 官方运动命令执行失败 code={code}")
                    return
                with self._navigation_lock:
                    self._official_navigation_state = NavigationState(
                        current_goal_id=goal.goal_id,
                        status=NavigationStatus.RUNNING,
                        current_pose=current_pose,
                        remaining_distance_m=tick_result.remaining_distance_m,
                        remaining_yaw_rad=tick_result.remaining_yaw_rad,
                        goal_reached=False,
                        message=tick_result.message,
                        metadata=tick_result.metadata,
                    )
                time.sleep(self.config.official.control_loop_interval_sec)

            self._stop_sport_motion()
        except Exception:
            LOGGER.exception("执行 Go2 官方运动导航失败。")
            self._set_official_navigation_failed(goal, "执行 Go2 官方运动导航时发生异常。")
        finally:
            with self._navigation_lock:
                if self._official_goal_thread is threading.current_thread():
                    self._official_goal_thread = None

    def _set_official_navigation_failed(
        self,
        goal: NavigationGoal,
        message: str,
        *,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        self._stop_sport_motion()
        failure_metadata = {"backend": "official_grid_navigation"}
        if metadata:
            failure_metadata.update(dict(metadata))
        with self._navigation_lock:
            self._official_goal_reached = False
            self._official_navigation_goal = None
            self._official_navigation_state = NavigationState(
                current_goal_id=goal.goal_id,
                status=NavigationStatus.FAILED,
                current_pose=self.get_current_pose(),
                message=message,
                metadata=failure_metadata,
            )
            self._official_last_error = message

    def _compute_direct_goal_distance(self, *, current_pose: Pose, target_pose: Pose) -> float:
        dx = float(target_pose.position.x - current_pose.position.x)
        dy = float(target_pose.position.y - current_pose.position.y)
        return math.hypot(dx, dy)

    def _compute_direct_goal_yaw_error(self, *, current_pose: Pose, target_pose: Pose) -> float:
        current_yaw = self._yaw_from_quaternion(current_pose.orientation)
        target_yaw = self._yaw_from_quaternion(target_pose.orientation)
        return self._normalize_angle(target_yaw - current_yaw)

    def _compute_official_goal_errors(self, *, current_pose: Pose, target_pose: Pose) -> Tuple[float, float, float]:
        dx = float(target_pose.position.x - current_pose.position.x)
        dy = float(target_pose.position.y - current_pose.position.y)
        distance = math.hypot(dx, dy)
        current_yaw = self._yaw_from_quaternion(current_pose.orientation)
        desired_heading = math.atan2(dy, dx) if distance > 1e-6 else current_yaw
        target_yaw = self._yaw_from_quaternion(target_pose.orientation)
        heading_error = self._normalize_angle(desired_heading - current_yaw)
        goal_yaw_error = self._normalize_angle(target_yaw - current_yaw)
        return distance, heading_error, goal_yaw_error

    def _yaw_from_quaternion(self, rotation: Quaternion) -> float:
        siny_cosp = 2.0 * (rotation.w * rotation.z + rotation.x * rotation.y)
        cosy_cosp = 1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _normalize_angle(self, angle_rad: float) -> float:
        return math.atan2(math.sin(angle_rad), math.cos(angle_rad))

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def _build_exploration_unavailable_message(self) -> str:
        if self._startup_diagnostics:
            return self._startup_diagnostics[-1]
        if not self.is_localization_available():
            return "当前定位尚未就绪，无法生成探索候选点。"
        if not self.is_map_available():
            return "当前地图尚未就绪，无法生成探索候选点。"
        return "当前地图中未找到可用探索候选点。"

    def _run_exploration(self, request: ExploreAreaRequest) -> None:
        try:
            with self._exploration_lock:
                self._exploration_state = ExplorationState(
                    current_request_id=request.request_id,
                    status=ExplorationStatus.RUNNING,
                    strategy=request.strategy,
                    message="探索任务运行中。",
                )
            completed = 0
            attempted_poses: List[Pose] = []
            frontier_count = 0
            iteration = 0
            known_cells = self._count_known_map_cells()
            no_gain_rounds = 0
            exploration_started_at = time.monotonic()
            final_status = ExplorationStatus.SUCCEEDED
            final_message = "探索任务完成。"

            while not self._exploration_stop_event.is_set():
                if request.max_duration_sec is not None and (
                    time.monotonic() - exploration_started_at
                ) >= float(request.max_duration_sec):
                    final_status = ExplorationStatus.FAILED
                    final_message = f"探索超时，已运行 {float(request.max_duration_sec):.1f} 秒。"
                    break

                candidate_poses = self._build_exploration_candidates(
                    request,
                    attempted_poses=attempted_poses,
                )
                frontier_count = len(candidate_poses)
                if not candidate_poses:
                    if completed > 0:
                        final_status = ExplorationStatus.SUCCEEDED
                        final_message = "探索完成，当前无更多可达前沿。"
                    else:
                        final_status = ExplorationStatus.FAILED
                        final_message = self._build_exploration_unavailable_message()
                    break

                iteration += 1
                pose = candidate_poses[0]
                if self._exploration_stop_event.is_set():
                    break
                goal = NavigationGoal(
                    goal_id=f"{request.request_id}_candidate_{iteration}",
                    target_pose=pose,
                    target_name=request.target_name,
                    metadata={"exploration_request_id": request.request_id, "candidate_index": iteration},
                )
                if not self.set_goal(goal):
                    attempted_poses.append(pose)
                    continue
                success = self._wait_navigation_goal(goal_timeout_sec=self.config.exploration.goal_timeout_sec)
                attempted_poses.append(pose)
                if success:
                    completed += 1

                next_known_cells = self._count_known_map_cells()
                info_gain_cells = max(0, next_known_cells - known_cells)
                known_cells = max(known_cells, next_known_cells)
                if self.get_occupancy_grid() is not None:
                    if info_gain_cells <= int(self.config.exploration.frontier_min_information_gain_cells):
                        no_gain_rounds += 1
                    else:
                        no_gain_rounds = 0
                    if no_gain_rounds >= int(self.config.exploration.frontier_max_no_gain_rounds):
                        final_status = ExplorationStatus.SUCCEEDED
                        final_message = "探索停止：连续多轮未获得足够的新地图信息。"
                        with self._exploration_lock:
                            self._exploration_state = ExplorationState(
                                current_request_id=request.request_id,
                                status=ExplorationStatus.RUNNING,
                                strategy=request.strategy,
                                covered_ratio=float(completed) / float(max(1, iteration)),
                                frontier_count=frontier_count,
                                message=(
                                    f"探索进行中，已完成 {completed} 次到达，"
                                    f"本轮新增已知栅格 {info_gain_cells}。"
                                ),
                                metadata={
                                    "known_cells": known_cells,
                                    "info_gain_cells": info_gain_cells,
                                    "attempted_frontiers": len(attempted_poses),
                                },
                            )
                        break

                with self._exploration_lock:
                    self._exploration_state = ExplorationState(
                        current_request_id=request.request_id,
                        status=ExplorationStatus.RUNNING,
                        strategy=request.strategy,
                        covered_ratio=float(completed) / float(max(1, iteration)),
                        frontier_count=frontier_count,
                        message=(
                            f"探索进行中，已完成 {completed} 次到达，"
                            f"剩余候选 {frontier_count}，本轮新增已知栅格 {info_gain_cells}。"
                        ),
                        metadata={
                            "known_cells": known_cells,
                            "info_gain_cells": info_gain_cells,
                            "attempted_frontiers": len(attempted_poses),
                        },
                    )
            with self._exploration_lock:
                if self._exploration_stop_event.is_set():
                    self._exploration_state = ExplorationState(
                        current_request_id=request.request_id,
                        status=ExplorationStatus.CANCELLED,
                        strategy=request.strategy,
                        covered_ratio=float(completed) / float(max(1, iteration)),
                        frontier_count=frontier_count,
                        message="探索任务已取消。",
                    )
                else:
                    self._exploration_state = ExplorationState(
                        current_request_id=request.request_id,
                        status=final_status,
                        strategy=request.strategy,
                        covered_ratio=float(completed) / float(max(1, iteration)),
                        frontier_count=0,
                        message=final_message,
                        metadata={
                            "known_cells": known_cells,
                            "attempted_frontiers": len(attempted_poses),
                            "completed_frontiers": completed,
                        },
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

    def _build_exploration_candidates(
        self,
        request: ExploreAreaRequest,
        *,
        attempted_poses: Optional[List[Pose]] = None,
    ) -> List[Pose]:
        current_pose = self.get_current_pose()
        center_pose = self._resolve_exploration_center_pose(request, current_pose=current_pose)
        if center_pose is None:
            return []

        radius = request.radius_m or self.config.exploration.sample_radius_m
        cost_map = self.get_cost_map()
        occupancy_grid = self.get_occupancy_grid()
        attempted_items = tuple(attempted_poses or [])

        if (
            request.strategy.strip().lower() == "frontier"
            and self.config.exploration.frontier_enabled
            and current_pose is not None
            and occupancy_grid is not None
        ):
            frontier_candidates = self._frontier_explorer.select_candidates(
                current_pose=current_pose,
                occupancy_grid=occupancy_grid,
                cost_map=cost_map,
                planner=self._grid_navigation_planner,
                center_pose=center_pose,
                radius_m=float(radius),
                attempted_poses=attempted_items,
            )
            if frontier_candidates:
                return [candidate.pose for candidate in frontier_candidates]

        return self._build_ring_exploration_candidates(
            center_pose=center_pose,
            current_pose=current_pose,
            radius=float(radius),
            cost_map=cost_map,
            occupancy_grid=occupancy_grid,
            attempted_poses=attempted_items,
        )

    def _resolve_exploration_center_pose(
        self,
        request: ExploreAreaRequest,
        *,
        current_pose: Optional[Pose],
    ) -> Optional[Pose]:
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
            center_pose = current_pose
        return center_pose

    def _build_ring_exploration_candidates(
        self,
        *,
        center_pose: Pose,
        current_pose: Optional[Pose],
        radius: float,
        cost_map: Optional[CostMap],
        occupancy_grid: Optional[OccupancyGrid],
        attempted_poses: Sequence[Pose],
    ) -> List[Pose]:
        sample_count = self.config.exploration.sample_count
        for candidate_radius in self._build_exploration_sample_radii(radius):
            reachable_candidates: List[Tuple[float, Pose]] = []
            candidates: List[Pose] = []
            for index in range(sample_count):
                angle = (2.0 * math.pi * float(index)) / float(sample_count)
                x = center_pose.position.x + candidate_radius * math.cos(angle)
                y = center_pose.position.y + candidate_radius * math.sin(angle)
                if any(self._poses_are_near(x=x, y=y, pose=item) for item in attempted_poses):
                    continue
                if cost_map is not None:
                    cost = self._sample_cost(cost_map, x=x, y=y)
                    if cost is None or cost > self.config.exploration.max_goal_cost:
                        continue
                candidate_pose = self._make_exploration_pose(center_pose=center_pose, x=x, y=y)
                candidates.append(candidate_pose)
                if current_pose is None or (cost_map is None and occupancy_grid is None):
                    continue
                planned_path = self._grid_navigation_planner.plan_preview(
                    current_pose=current_pose,
                    target_pose=candidate_pose,
                    occupancy_grid=occupancy_grid,
                    cost_map=cost_map,
                )
                if planned_path is None or planned_path.planning_mode != "goal":
                    candidates.pop()
                    continue
                reachable_candidates.append((planned_path.total_distance_m, candidate_pose))
            if reachable_candidates:
                reachable_candidates.sort(key=lambda item: item[0])
                return [pose for _distance, pose in reachable_candidates]
            if candidates:
                return candidates
        return []

    def _make_exploration_pose(self, *, center_pose: Pose, x: float, y: float) -> Pose:
        return Pose(
            frame_id=center_pose.frame_id,
            position=Vector3(x=x, y=y, z=center_pose.position.z),
            orientation=Quaternion(
                x=center_pose.orientation.x,
                y=center_pose.orientation.y,
                z=center_pose.orientation.z,
                w=center_pose.orientation.w,
            ),
        )

    def _poses_are_near(self, *, x: float, y: float, pose: Pose) -> bool:
        return (
            math.hypot(float(pose.position.x) - x, float(pose.position.y) - y)
            < float(self.config.exploration.frontier_revisit_separation_m)
        )

    def _count_known_map_cells(self) -> int:
        return self._frontier_explorer.count_known_cells(self.get_occupancy_grid())

    def _build_exploration_sample_radii(self, requested_radius: float) -> List[float]:
        normalized_radius = max(0.1, float(requested_radius))
        default_radius = max(0.1, float(self.config.exploration.sample_radius_m))
        minimum_radius = min(
            normalized_radius,
            max(0.5, min(normalized_radius, default_radius) * 0.5),
        )
        raw_radii = [normalized_radius]
        if default_radius < normalized_radius:
            raw_radii.append(default_radius)

        current_radius = normalized_radius
        for _ in range(6):
            current_radius *= 0.75
            if current_radius <= minimum_radius + 1e-6:
                break
            raw_radii.append(current_radius)
        raw_radii.extend([1.0, 0.75, 0.5, minimum_radius])

        radii: List[float] = []
        for radius in sorted(raw_radii, reverse=True):
            if radius > normalized_radius + 1e-6:
                continue
            rounded_radius = round(max(0.1, float(radius)), 3)
            if any(abs(existing - rounded_radius) < 1e-6 for existing in radii):
                continue
            radii.append(rounded_radius)
        return radii

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
