from __future__ import annotations

from dataclasses import dataclass

from adapters.base import AdapterCategory, AdapterConfig
from adapters.streams import RosImageAdapter
from contracts.image import ImageFrame
from contracts.robot_state import RobotControlMode
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
