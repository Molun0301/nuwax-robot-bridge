from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from contracts.geometry import FrameTree, Pose, Quaternion, Twist, Vector3
from contracts.image import CameraInfo, ImageEncoding, ImageFrame
from contracts.maps import CostMap, OccupancyGrid, SemanticMap
from contracts.navigation import ExplorationState, ExploreAreaRequest, NavigationGoal, NavigationState, NavigationStatus
from contracts.robot_state import IMUState, JointState, RobotControlMode, RobotState, SafetyState
from drivers.robots.common import MotionCommandDataPlane
from drivers.robots.go2.defaults import GO2_DEFAULT_CAMERA_INFO
from providers.exploration import ExplorationProvider
from providers.image import ImageProvider
from providers.localization import LocalizationProvider
from providers.maps import MapProvider
from providers.motion import MotionControl
from providers.navigation import NavigationProvider
from providers.safety import SafetyProvider
from providers.state import StateProvider

if TYPE_CHECKING:
    from drivers.robots.go2.assembly import Go2RobotAssembly


def _to_control_mode(mode: str) -> RobotControlMode:
    if mode == "low":
        return RobotControlMode.LOW_LEVEL
    if mode == "high":
        return RobotControlMode.HIGH_LEVEL
    return RobotControlMode.UNKNOWN


def _resolve_motion_command_data_plane(data_plane) -> Optional[MotionCommandDataPlane]:
    """返回支持底层运动命令链路的数据面。"""

    if data_plane is None or not isinstance(data_plane, MotionCommandDataPlane):
        return None
    if not data_plane.can_accept_motion_command():
        return None
    return data_plane


@dataclass
class Go2ProviderBundle(
    StateProvider,
    ImageProvider,
    LocalizationProvider,
    MapProvider,
    NavigationProvider,
    ExplorationProvider,
    MotionControl,
    SafetyProvider,
):
    """Go2 默认提供器集合。"""

    assembly: "Go2RobotAssembly"
    provider_name: str = "go2_default_bundle"
    provider_version: str = "0.1.0"

    def is_available(self) -> bool:
        return self.assembly.high_level_initialized

    def is_state_available(self) -> bool:
        return self.assembly.high_level_initialized

    def is_image_available(self) -> bool:
        return self.assembly.high_level_initialized

    def is_localization_available(self) -> bool:
        return self.assembly.data_plane is not None and self.assembly.data_plane.is_localization_available()

    def is_map_available(self) -> bool:
        return self.assembly.data_plane is not None and self.assembly.data_plane.is_map_available()

    def is_navigation_available(self) -> bool:
        return self.assembly.data_plane is not None and self.assembly.data_plane.is_navigation_available()

    def is_exploration_available(self) -> bool:
        return self.assembly.data_plane is not None and self.assembly.data_plane.is_exploration_available()

    def get_robot_state(self) -> RobotState:
        metadata = {
            "control_mode": self.assembly.current_mode,
            "mode_state": {
                "control_mode": self.assembly.current_mode,
                "low_level_ready": self.assembly.low_level_controller is not None,
                "high_level_initialized": self.assembly.high_level_initialized,
            },
            "low_level_ready": self.assembly.low_level_controller is not None,
            "default_sensors": list(self.assembly.manifest.default_sensors),
            "default_audio_backends": list(self.assembly.manifest.default_audio_backends),
        }
        try:
            metadata["robot_volume"] = self.assembly.get_vui_volume_info()
        except Exception:
            metadata["robot_volume"] = None
        metadata["volume_state"] = metadata["robot_volume"]
        if self.assembly.data_plane is not None:
            metadata["data_plane"] = self.assembly.data_plane.get_status()

        return RobotState(
            robot_id=self.assembly.defaults.parameters["robot_id"],
            frame_id=self.assembly.defaults.frame_ids["base"],
            mode=_to_control_mode(self.assembly.current_mode),
            pose=self.get_current_pose(),
            joints=self.get_joint_states(),
            imu=self.get_imu_state(),
            safety=self.get_safety_state(),
            metadata=metadata,
        )

    def get_joint_states(self) -> List[JointState]:
        if self.assembly.low_level_controller is None:
            return []

        raw_state = self.assembly.low_level_controller.get_joint_state()
        joints = raw_state.get("joints", []) if isinstance(raw_state, dict) else []
        return [
            JointState(
                name=joint["name"],
                position_rad=float(joint["position"]),
                velocity_rad_s=float(joint["velocity"]),
                effort_nm=float(joint["torque"]),
            )
            for joint in joints
        ]

    def get_imu_state(self) -> Optional[IMUState]:
        if self.assembly.data_plane is not None:
            imu_state = self.assembly.data_plane.get_imu_state()
            if imu_state is not None:
                return imu_state

        if self.assembly.low_level_controller is None:
            return None

        raw_state = self.assembly.low_level_controller.get_imu_state()
        if not raw_state:
            return None

        quaternion = raw_state.get("quaternion", [0.0, 0.0, 0.0, 1.0])
        gyroscope = raw_state.get("gyroscope", [0.0, 0.0, 0.0])
        accelerometer = raw_state.get("accelerometer", [0.0, 0.0, 0.0])
        return IMUState(
            frame_id=self.assembly.defaults.frame_ids["imu"],
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

    def capture_image(self, camera_id: Optional[str] = None) -> ImageFrame:
        image_bytes = self.assembly.capture_image_bytes()
        camera_info = self.get_camera_info(camera_id)
        assert camera_info is not None
        return ImageFrame(
            camera_id=camera_id or camera_info.camera_id,
            frame_id=camera_info.frame_id,
            width_px=camera_info.width_px,
            height_px=camera_info.height_px,
            encoding=ImageEncoding.JPEG,
            data=image_bytes,
            metadata={"size_bytes": len(image_bytes)},
        )

    def get_camera_info(self, camera_id: Optional[str] = None) -> Optional[CameraInfo]:
        camera_info = GO2_DEFAULT_CAMERA_INFO.model_copy(deep=True)
        if camera_id:
            camera_info.camera_id = camera_id
        return camera_info

    def get_current_pose(self) -> Optional[Pose]:
        if self.assembly.data_plane is None:
            return None
        return self.assembly.data_plane.get_current_pose()

    def get_frame_tree(self) -> Optional[FrameTree]:
        if self.assembly.data_plane is None:
            return None
        return self.assembly.data_plane.get_frame_tree()

    def get_occupancy_grid(self) -> Optional[OccupancyGrid]:
        if self.assembly.data_plane is None:
            return None
        return self.assembly.data_plane.get_occupancy_grid()

    def get_cost_map(self) -> Optional[CostMap]:
        if self.assembly.data_plane is None:
            return None
        return self.assembly.data_plane.get_cost_map()

    def get_semantic_map(self) -> Optional[SemanticMap]:
        if self.assembly.data_plane is None:
            return None
        return self.assembly.data_plane.get_semantic_map()

    def set_goal(self, goal: NavigationGoal) -> bool:
        if self.assembly.data_plane is None:
            return False
        return self.assembly.data_plane.set_goal(goal)

    def cancel_goal(self) -> bool:
        if self.assembly.data_plane is None:
            return False
        return self.assembly.data_plane.cancel_goal()

    def get_navigation_state(self) -> NavigationState:
        if self.assembly.data_plane is None:
            return NavigationState(status=NavigationStatus.IDLE, message="Go2 数据面未启用。")
        return self.assembly.data_plane.get_navigation_state()

    def is_goal_reached(self) -> bool:
        if self.assembly.data_plane is None:
            return False
        return self.assembly.data_plane.is_goal_reached()

    def start_exploration(self, request: ExploreAreaRequest) -> bool:
        if self.assembly.data_plane is None:
            return False
        return self.assembly.data_plane.start_exploration(request)

    def stop_exploration(self) -> bool:
        if self.assembly.data_plane is None:
            return False
        return self.assembly.data_plane.stop_exploration()

    def get_exploration_state(self) -> ExplorationState:
        if self.assembly.data_plane is None:
            return ExplorationState(message="Go2 数据面未启用。")
        return self.assembly.data_plane.get_exploration_state()

    def send_twist(self, twist: Twist) -> None:
        data_plane = _resolve_motion_command_data_plane(self.assembly.data_plane)
        if data_plane is not None:
            code = data_plane.send_motion_command(
                twist.linear.x,
                twist.linear.y,
                twist.angular.z,
            )
            if code != 0:
                raise RuntimeError(f"Go2 数据面运动命令执行失败: code={code}")
            return
        self.assembly.move(
            twist.linear.x,
            twist.linear.y,
            twist.angular.z,
        )

    def stop_motion(self) -> None:
        data_plane = _resolve_motion_command_data_plane(self.assembly.data_plane)
        if data_plane is not None:
            data_plane.stop_motion_command()
            return
        self.assembly.stop_move()

    def get_safety_state(self) -> SafetyState:
        return SafetyState(
            is_estopped=False,
            motors_enabled=self.assembly.high_level_initialized,
            can_move=self.assembly.high_level_initialized,
            warning_message=None,
        )

    def request_safe_stop(self, reason: Optional[str] = None) -> None:
        self.assembly.stop_all_motion(reason)
