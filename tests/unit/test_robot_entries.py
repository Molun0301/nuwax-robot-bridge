from __future__ import annotations

from dataclasses import dataclass

from adapters.base import AdapterCategory, AdapterConfig
from adapters.streams import RosImageAdapter
from contracts.geometry import FrameTree, Pose, Quaternion, Transform, Twist, Vector3
from contracts.image import ImageFrame
from contracts.maps import CostMap, OccupancyGrid, SemanticMap, SemanticRegion
from contracts.navigation import ExplorationState, ExplorationStatus, ExploreAreaRequest, NavigationGoal, NavigationState, NavigationStatus
from contracts.pointcloud import PointCloudFrame
from contracts.robot_state import IMUState, RobotControlMode
from drivers.robots.common import ManagedRobotDataPlane, MotionCommandDataPlane
from drivers.robots.g1 import G1_MANIFEST, create_g1_assembly
from drivers.robots.go2 import GO2_CAPABILITY_DESCRIPTORS, GO2_CAPABILITY_MATRIX, create_go2_assembly
from drivers.robots.go2.assembly import Go2ClientFactories
from settings import APP_CONFIG
from typing import Dict, List, Optional, Tuple


class FakeSportClient:
    """Go2 高层运动客户端桩。"""

    def __init__(self) -> None:
        self.timeout = None
        self.initialized = False
        self.calls: List[Tuple[str, Tuple[object, ...]]] = []

    def SetTimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def Init(self) -> None:
        self.initialized = True

    def Move(self, vx: float, vy: float, vyaw: float) -> int:
        self.calls.append(("Move", (vx, vy, vyaw)))
        return 0

    def StopMove(self) -> int:
        self.calls.append(("StopMove", ()))
        return 0

    def Euler(self, roll: float, pitch: float, yaw: float) -> int:
        self.calls.append(("Euler", (roll, pitch, yaw)))
        return 0

    def SpeedLevel(self, level: int) -> int:
        self.calls.append(("SpeedLevel", (level,)))
        return 0

    def __getattr__(self, name: str):
        def _method(*args):
            self.calls.append((name, args))
            return 0

        return _method


class FakeVideoClient:
    """Go2 图像客户端桩。"""

    def __init__(self) -> None:
        self.timeout = None
        self.initialized = False

    def SetTimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def Init(self) -> None:
        self.initialized = True

    def GetImageSample(self) -> Tuple[int, bytes]:
        return 0, b"fake-image"


class FakeVuiClient:
    """Go2 音量客户端桩。"""

    def __init__(self) -> None:
        self.timeout = None
        self.initialized = False
        self.level = 5
        self.switch_enabled = 1

    def SetTimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def Init(self) -> None:
        self.initialized = True

    def GetVolume(self) -> Tuple[int, int]:
        return 0, self.level

    def GetSwitch(self) -> Tuple[int, int]:
        return 0, self.switch_enabled

    def SetSwitch(self, enabled: int) -> int:
        self.switch_enabled = enabled
        return 0

    def SetVolume(self, level: int) -> int:
        self.level = level
        return 0


class FakeLowLevelController:
    """Go2 低层控制器桩。"""

    def __init__(self, iface: Optional[str] = None) -> None:
        self.iface = iface
        self.is_running = False
        self.initialized = False

    def init(self) -> bool:
        self.initialized = True
        return True

    def start(self) -> None:
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False

    def set_joint_position(self, joint_name: str, position: float, kp: float, kd: float) -> bool:
        self.last_joint_position = (joint_name, position, kp, kd)
        return True

    def set_joints_position(self, joint_positions: Dict[str, float], kp: float, kd: float) -> bool:
        self.last_joint_positions = (joint_positions, kp, kd)
        return True

    def set_joint_velocity(self, joint_name: str, velocity: float, kd: float) -> bool:
        self.last_joint_velocity = (joint_name, velocity, kd)
        return True

    def set_joint_torque(self, joint_name: str, torque: float) -> bool:
        self.last_joint_torque = (joint_name, torque)
        return True

    def get_joint_state(self, joint_name: Optional[str] = None) -> Dict[str, object]:
        if joint_name:
            return {"name": joint_name, "position": 0.1, "velocity": 0.2, "torque": 0.3}
        return {
            "joints": [
                {"name": "FR_0", "position": 0.1, "velocity": 0.2, "torque": 0.3},
                {"name": "FR_1", "position": 0.4, "velocity": 0.5, "torque": 0.6},
            ]
        }

    def get_imu_state(self) -> Dict[str, object]:
        return {
            "quaternion": [0.0, 0.0, 0.0, 1.0],
            "gyroscope": [0.1, 0.2, 0.3],
            "accelerometer": [1.0, 2.0, 3.0],
            "temperature": 30.0,
        }

    def apply_pose_preset(self, preset_name: str) -> bool:
        self.last_pose_preset = preset_name
        return True


@dataclass
class FakeChannelInitializer:
    """DDS 初始化桩。"""

    last_call: Optional[Tuple[int, Optional[str]]] = None

    def __call__(self, domain_id: int, iface: Optional[str]) -> None:
        self.last_call = (domain_id, iface)


class FakeGo2DataPlane:
    """Go2 端侧数据面桩。"""

    def __init__(self) -> None:
        self.started = False
        self.pose = Pose(frame_id="odom", position=Vector3(x=1.0, y=2.0, z=0.0))
        self.frame_tree = FrameTree(
            root_frame_id="odom",
            transforms=[
                Transform(
                    parent_frame_id="odom",
                    child_frame_id="body",
                    translation=Vector3(x=1.0, y=2.0, z=0.0),
                    rotation=Quaternion(w=1.0),
                )
            ],
        )
        self.occupancy_grid = OccupancyGrid(
            map_id="go2_occ",
            frame_id="odom",
            width=2,
            height=2,
            resolution_m=0.1,
            origin=Pose(frame_id="odom", position=Vector3()),
            data=[0, 0, 50, 100],
        )
        self.cost_map = CostMap(
            map_id="go2_cost",
            frame_id="odom",
            width=2,
            height=2,
            resolution_m=0.1,
            origin=Pose(frame_id="odom", position=Vector3()),
            data=[0.0, 10.0, 40.0, 100.0],
        )
        self.semantic_map = SemanticMap(
            map_id="go2_semantic",
            frame_id="odom",
            regions=[
                SemanticRegion(
                    region_id="dock_1",
                    label="dock",
                    centroid=Pose(frame_id="odom", position=Vector3(x=1.2, y=2.3, z=0.0)),
                    attributes={"alias": "充电桩"},
                )
            ],
        )
        self.navigation_state = NavigationState(status=NavigationStatus.IDLE, current_pose=self.pose)
        self.exploration_state = ExplorationState(status=ExplorationStatus.IDLE)
        self.motion_commands: List[Tuple[float, float, float]] = []
        self.motion_stop_count = 0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def is_localization_available(self) -> bool:
        return True

    def is_map_available(self) -> bool:
        return True

    def is_navigation_available(self) -> bool:
        return True

    def is_exploration_available(self) -> bool:
        return True

    def get_current_pose(self) -> Optional[Pose]:
        return self.pose

    def get_frame_tree(self) -> Optional[FrameTree]:
        return self.frame_tree

    def get_occupancy_grid(self) -> Optional[OccupancyGrid]:
        return self.occupancy_grid

    def get_cost_map(self) -> Optional[CostMap]:
        return self.cost_map

    def get_semantic_map(self) -> Optional[SemanticMap]:
        return self.semantic_map

    def get_latest_local_point_cloud(self) -> Optional[PointCloudFrame]:
        return PointCloudFrame(
            frame_id="odom",
            point_count=1,
            points=[Vector3(x=1.0, y=0.0, z=0.2)],
            metadata={"source_topic": "/fake/cloud"},
        )

    def set_goal(self, goal: NavigationGoal) -> bool:
        self.navigation_state = NavigationState(
            current_goal_id=goal.goal_id,
            status=NavigationStatus.ACCEPTED,
            current_pose=self.pose,
        )
        self.last_goal = goal
        return True

    def cancel_goal(self) -> bool:
        self.navigation_state = self.navigation_state.model_copy(
            update={"status": NavigationStatus.CANCELLED},
            deep=True,
        )
        return True

    def get_navigation_state(self) -> NavigationState:
        return self.navigation_state

    def is_goal_reached(self) -> bool:
        return self.navigation_state.status == NavigationStatus.SUCCEEDED

    def start_exploration(self, request: ExploreAreaRequest) -> bool:
        self.exploration_state = ExplorationState(
            current_request_id=request.request_id,
            status=ExplorationStatus.RUNNING,
            strategy=request.strategy,
        )
        self.last_explore_request = request
        return True

    def stop_exploration(self) -> bool:
        self.exploration_state = self.exploration_state.model_copy(
            update={"status": ExplorationStatus.CANCELLED},
            deep=True,
        )
        return True

    def get_exploration_state(self) -> ExplorationState:
        return self.exploration_state

    def get_status(self) -> Dict[str, object]:
        return {
            "enabled": True,
            "started": self.started,
            "localization_available": True,
            "map_available": True,
            "navigation_available": True,
            "exploration_available": True,
        }

    def get_imu_state(self) -> Optional[IMUState]:
        return IMUState(frame_id="odom/imu")

    def can_accept_motion_command(self) -> bool:
        return self.started

    def send_motion_command(self, vx: float, vy: float, vyaw: float) -> int:
        self.motion_commands.append((vx, vy, vyaw))
        return 0

    def stop_motion_command(self) -> None:
        self.motion_stop_count += 1


def test_fake_go2_data_plane_matches_common_protocols() -> None:
    """Go2 数据面桩应满足通用数据面协议，避免平台继续耦合具体实现。"""

    data_plane = FakeGo2DataPlane()

    assert isinstance(data_plane, ManagedRobotDataPlane)
    assert isinstance(data_plane, MotionCommandDataPlane)


def _build_go2_factories(channel_initializer: FakeChannelInitializer) -> Go2ClientFactories:
    return Go2ClientFactories(
        channel_factory_initialize=channel_initializer,
        sport_client_factory=FakeSportClient,
        video_client_factory=FakeVideoClient,
        vui_client_factory=FakeVuiClient,
        low_level_controller_factory=FakeLowLevelController,
    )


def test_go2_assembly_can_be_used_as_single_entrypoint(tmp_path) -> None:
    """Go2 应通过唯一装配入口完成现有能力接入。"""

    channel_initializer = FakeChannelInitializer()
    assembly = create_go2_assembly(APP_CONFIG, iface="eth0", factories=_build_go2_factories(channel_initializer))

    assembly.start()
    status = assembly.get_status()

    assert status.initialized is True
    assert status.control_mode == "high"
    assert channel_initializer.last_call == (0, "eth0")

    code, payload = assembly.execute_action("save_image", {"filename": str(tmp_path / "go2.jpg")})
    assert code == 0
    assert payload["size"] == len(b"fake-image")


def test_go2_provider_outputs_match_platform_contracts() -> None:
    """Go2 提供器输出必须是统一契约。"""

    assembly = create_go2_assembly(APP_CONFIG, factories=_build_go2_factories(FakeChannelInitializer()))
    assembly.start()

    image = assembly.providers.capture_image()
    state = assembly.providers.get_robot_state()

    assert isinstance(image, ImageFrame)
    assert image.width_px == 1280
    assert state.mode == RobotControlMode.HIGH_LEVEL
    assert state.metadata["default_sensors"]


def test_go2_provider_exposes_local_point_cloud_when_data_plane_supports_it() -> None:
    """Go2 提供器应在数据面可用时转发最近局部点云。"""

    assembly = create_go2_assembly(
        APP_CONFIG,
        factories=_build_go2_factories(FakeChannelInitializer()),
        data_plane=FakeGo2DataPlane(),
    )
    assembly.start()

    point_cloud = assembly.providers.get_latest_point_cloud()

    assert point_cloud is not None
    assert point_cloud.frame_id == "odom"
    assert point_cloud.point_count == 1


def test_go2_low_level_mode_and_capability_matrix_are_consistent() -> None:
    """Go2 低层接入和能力矩阵需要自洽。"""

    assembly = create_go2_assembly(APP_CONFIG, factories=_build_go2_factories(FakeChannelInitializer()))
    assembly.start()

    code, data = assembly.execute_action("switch_mode", {"mode": "low"})
    assert code == 0
    assert data["mode"] == "low"

    robot_state = assembly.providers.get_robot_state()
    assert robot_state.mode == RobotControlMode.LOW_LEVEL
    assert len(robot_state.joints) == 2

    descriptor_names = {descriptor.name for descriptor in GO2_CAPABILITY_DESCRIPTORS}
    matrix_names = {item.name for item in GO2_CAPABILITY_MATRIX.capabilities}
    manifest_names = {item.name for item in assembly.manifest.capability_matrix.capabilities}

    assert descriptor_names == matrix_names
    assert descriptor_names == manifest_names


def test_go2_assembly_can_initialize_prebound_adapters() -> None:
    """Go2 装配入口应能显式绑定适配器并汇总健康状态。"""

    override_config = AdapterConfig(
        name="go2_front_camera_stream",
        category=AdapterCategory.STREAMS,
        source_kind="ros2",
        contract_type="ImageFrame",
        source_ref="/robot/front/image",
        settings={"default_camera_id": "go2_front_camera", "default_frame_id": "go2/front_camera"},
    )
    adapter = RosImageAdapter(RosImageAdapter.build_default_config(name="go2_front_camera_stream"))
    assembly = create_go2_assembly(
        APP_CONFIG,
        factories=_build_go2_factories(FakeChannelInitializer()),
        adapter_configs={"go2_front_camera_stream": override_config},
        prebound_adapters=(adapter,),
    )

    assembly.start()
    status = assembly.get_status()
    health_statuses = assembly.get_adapter_health_statuses()
    frame = adapter.adapt(
        {
            "width": 320,
            "height": 240,
            "encoding": "jpeg",
            "data": b"frame",
        }
    )

    assert status.adapter_count == 1
    assert status.healthy_adapter_count == 1
    assert len(health_statuses) == 1
    assert health_statuses[0].is_healthy is True
    assert health_statuses[0].metadata["source_ref"] == "/robot/front/image"
    assert frame.camera_id == "go2_front_camera"
    assert frame.frame_id == "go2/front_camera"


def test_g1_skeleton_keeps_entry_format_stable() -> None:
    """G1 即使未实现，也要先保持入口格式稳定。"""

    assembly = create_g1_assembly(APP_CONFIG)
    status = assembly.get_status()

    assert G1_MANIFEST.robot_name == "g1"
    assert status.initialized is False


def test_go2_data_plane_is_managed_under_robot_entry() -> None:
    """Go2 端侧定位、地图、导航与探索能力应由机器人入口统一管理。"""

    data_plane = FakeGo2DataPlane()
    assembly = create_go2_assembly(
        APP_CONFIG,
        factories=_build_go2_factories(FakeChannelInitializer()),
        data_plane=data_plane,
    )

    assembly.start()

    assert data_plane.started is True
    assert assembly.providers.is_localization_available() is True
    assert assembly.providers.get_current_pose() is not None
    assert assembly.providers.get_occupancy_grid() is not None
    assert assembly.providers.get_semantic_map() is not None

    accepted = assembly.providers.set_goal(
        NavigationGoal(
            goal_id="nav_to_dock",
            target_pose=Pose(frame_id="odom", position=Vector3(x=1.2, y=2.3, z=0.0)),
        )
    )
    exploring = assembly.providers.start_exploration(
        ExploreAreaRequest(request_id="explore_dock", target_name="充电桩", strategy="frontier")
    )

    assert accepted is True
    assert exploring is True
    assert assembly.providers.get_navigation_state().current_goal_id == "nav_to_dock"
    assert assembly.providers.get_exploration_state().current_request_id == "explore_dock"

    assembly.stop()
    assert data_plane.started is False


def test_go2_provider_motion_prefers_data_plane_command_path() -> None:
    """Go2 MotionControl 应优先复用数据面运动链，避免绕开避障后端。"""

    data_plane = FakeGo2DataPlane()
    assembly = create_go2_assembly(
        APP_CONFIG,
        factories=_build_go2_factories(FakeChannelInitializer()),
        data_plane=data_plane,
    )

    assembly.start()
    assembly.providers.send_twist(
        Twist(
            frame_id="odom",
            linear=Vector3(x=0.4, y=0.0, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=0.1),
        )
    )
    assembly.providers.stop_motion()

    assert data_plane.motion_commands[-1] == (0.4, 0.0, 0.1)
    assert data_plane.motion_stop_count == 1
    assert ("Move", (0.4, 0.0, 0.1)) not in assembly.sport_client.calls
