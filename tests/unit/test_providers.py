from __future__ import annotations

from contracts.geometry import FrameTree, Pose, Quaternion, Transform, Twist, Vector3
from contracts.image import CameraInfo, ImageEncoding, ImageFrame
from contracts.maps import CostMap, OccupancyGrid, SemanticMap
from contracts.navigation import ExplorationState, ExploreAreaRequest, NavigationGoal, NavigationState
from contracts.robot_state import IMUState, JointState, RobotState, SafetyState
from providers import ExplorationProvider, ImageProvider, LocalizationProvider, MapProvider, MotionControl, NavigationProvider, SafetyProvider, StateProvider
from typing import List, Optional


class DummyRobotProvider:
    """用于测试的最小提供器实现。"""

    provider_name = "dummy_robot"
    provider_version = "0.1.0"

    def is_available(self) -> bool:
        return True

    def get_robot_state(self) -> RobotState:
        return RobotState(robot_id="dummy", frame_id="world/dummy/base")

    def get_joint_states(self) -> List[JointState]:
        return []

    def get_imu_state(self) -> Optional[IMUState]:
        return IMUState(frame_id="world/dummy/imu")

    def capture_image(self, camera_id: Optional[str] = None) -> ImageFrame:
        return ImageFrame(
            camera_id=camera_id or "front",
            frame_id="world/dummy/camera_front",
            width_px=16,
            height_px=16,
            encoding=ImageEncoding.JPEG,
            data=b"demo",
        )

    def get_camera_info(self, camera_id: Optional[str] = None) -> Optional[CameraInfo]:
        return CameraInfo(
            camera_id=camera_id or "front",
            frame_id="world/dummy/camera_front",
            width_px=16,
            height_px=16,
            fx=10.0,
            fy=10.0,
            cx=8.0,
            cy=8.0,
        )

    def get_occupancy_grid(self) -> Optional[OccupancyGrid]:
        return None

    def get_cost_map(self) -> Optional[CostMap]:
        return None

    def get_semantic_map(self) -> Optional[SemanticMap]:
        return None

    def get_current_pose(self) -> Optional[Pose]:
        return Pose(frame_id="world/dummy/base")

    def get_frame_tree(self) -> Optional[FrameTree]:
        return FrameTree(
            root_frame_id="world",
            transforms=[
                Transform(
                    parent_frame_id="world",
                    child_frame_id="world/dummy/base",
                    translation=Vector3(),
                    rotation=Quaternion(w=1.0),
                )
            ],
        )

    def set_goal(self, goal: NavigationGoal) -> bool:
        self.last_goal = goal
        return True

    def cancel_goal(self) -> bool:
        self.goal_cancelled = True
        return True

    def get_navigation_state(self) -> NavigationState:
        return NavigationState(current_goal_id=getattr(self, "last_goal", None) and self.last_goal.goal_id)

    def is_goal_reached(self) -> bool:
        return False

    def start_exploration(self, request: ExploreAreaRequest) -> bool:
        self.last_explore_request = request
        return True

    def stop_exploration(self) -> bool:
        self.exploration_stopped = True
        return True

    def get_exploration_state(self) -> ExplorationState:
        return ExplorationState(current_request_id=getattr(self, "last_explore_request", None) and self.last_explore_request.request_id)

    def send_twist(self, twist: Twist) -> None:
        self.last_twist = twist

    def stop_motion(self) -> None:
        self.last_twist = None

    def get_safety_state(self) -> SafetyState:
        return SafetyState()

    def request_safe_stop(self, reason: Optional[str] = None) -> None:
        self.safe_stop_reason = reason


def test_dummy_provider_matches_protocols() -> None:
    """统一提供器协议应能被后续驱动直接实现。"""

    provider = DummyRobotProvider()

    assert isinstance(provider, StateProvider)
    assert isinstance(provider, ImageProvider)
    assert isinstance(provider, LocalizationProvider)
    assert isinstance(provider, MapProvider)
    assert isinstance(provider, NavigationProvider)
    assert isinstance(provider, ExplorationProvider)
    assert isinstance(provider, MotionControl)
    assert isinstance(provider, SafetyProvider)

    provider.stop_motion()
    provider.request_safe_stop("单元测试")

    assert provider.is_available() is True
    assert provider.safe_stop_reason == "单元测试"
