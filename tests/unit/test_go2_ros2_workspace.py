from __future__ import annotations

import os
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

from contracts.geometry import Pose, Quaternion, Vector3
from contracts.maps import CostMap
from contracts.navigation import ExplorationStatus, ExploreAreaRequest
from drivers.robots.go2.data_plane import Go2DataPlaneRuntime, RclpyGo2RosBridge
from drivers.robots.go2 import settings as go2_settings
from drivers.robots.go2.settings import (
    Go2DataPlaneConfig,
    Go2DirectDdsConfig,
    Go2ExplorationConfig,
    Go2MapSynthesisConfig,
)


class _FakeBridge:
    """测试用 Go2 ROS2 桥。"""

    def __init__(self, *, localization_available: bool, map_available: bool, navigation_available: bool) -> None:
        self.localization_available = localization_available
        self.map_available = map_available
        self.navigation_available = navigation_available
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def is_localization_available(self) -> bool:
        return self.localization_available

    def is_map_available(self) -> bool:
        return self.map_available

    def is_navigation_available(self) -> bool:
        return self.navigation_available

    def get_status(self):
        return {"started": self.started}

    def get_current_pose(self):
        return None

    def get_frame_tree(self):
        return None

    def get_occupancy_grid(self):
        return None

    def get_cost_map(self):
        return None

    def get_semantic_map(self):
        return None

    def set_goal(self, goal):
        del goal
        return False

    def cancel_goal(self):
        return False

    def get_navigation_state(self):
        return None

    def is_goal_reached(self):
        return False


def _build_pose_odom_message(*, x: float, y: float, z: float = 0.0, frame_id: str = "odom") -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=frame_id),
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=y, z=z),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        ),
    )


def _build_sport_mode_state_message(*, x: float, y: float, z: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        position=[x, y, z],
        imu_state=SimpleNamespace(quaternion=[0.0, 0.0, 0.0, 1.0]),
    )


def _build_point_cloud_message(points_xyz: list[tuple[float, float, float]]) -> SimpleNamespace:
    data = bytearray()
    for x, y, z in points_xyz:
        data.extend(struct.pack("<fff", x, y, z))
        data.extend(b"\x00\x00\x00\x00")
    return SimpleNamespace(
        header=SimpleNamespace(frame_id="utlidar_lidar"),
        height=1,
        width=len(points_xyz),
        is_bigendian=False,
        point_step=16,
        row_step=16 * len(points_xyz),
        fields=[
            SimpleNamespace(name="x", offset=0, datatype=7, count=1),
            SimpleNamespace(name="y", offset=4, datatype=7, count=1),
            SimpleNamespace(name="z", offset=8, datatype=7, count=1),
        ],
        data=bytes(data),
    )


def _grid_cell_index(*, width: int, resolution: float, origin_x: float, origin_y: float, x: float, y: float) -> int:
    col = int((x - origin_x) / resolution)
    row = int((y - origin_y) / resolution)
    return row * width + col


def test_go2_ros2_workspace_defaults_point_to_repo() -> None:
    assert go2_settings._DEFAULT_GO2_SETUP_SCRIPT.endswith("drivers/robots/go2/ros2_ws/install/setup.bash")


def test_go2_ros2_workspace_assets_exist() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    workspace_root = repo_root / "drivers" / "robots" / "go2" / "ros2_ws"
    assert (workspace_root / "src" / "nuwax_go2_bringup" / "package.xml").exists()
    assert (repo_root / "drivers" / "robots" / "go2" / "build_ros2_workspace.sh").exists()
    assert (repo_root / "drivers" / "robots" / "go2" / "source_runtime_env.sh").exists()


def test_go2_setup_script_path_falls_back_to_repo_default_when_legacy_path_missing(tmp_path: Path) -> None:
    fallback_setup = tmp_path / "ros2_ws" / "install" / "setup.bash"
    fallback_setup.parent.mkdir(parents=True)
    fallback_setup.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    legacy_missing = "/home/unitree/nuwax_robot_bridge/drivers/robots/go2/ros2_ws/install/setup.bash"

    resolved = go2_settings._resolve_setup_script_path(legacy_missing, str(fallback_setup))

    assert resolved == str(fallback_setup)


def test_go2_bridge_can_apply_ros_setup_environment(monkeypatch, tmp_path: Path) -> None:
    setup_script = tmp_path / "setup.bash"
    setup_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    bridge = RclpyGo2RosBridge(
        Go2DataPlaneConfig(
            enabled=True,
            setup_script=str(setup_script),
        )
    )

    python_path_a = str(tmp_path / "ros_py_a")
    python_path_b = str(tmp_path / "ros_py_b")
    completed = subprocess.CompletedProcess(
        args=["bash", "-lc", "source setup && env -0"],
        returncode=0,
        stdout=(
            f"PYTHONPATH={python_path_a}:{python_path_b}\x00"
            "ROS_DISTRO=humble\x00"
        ).encode("utf-8"),
    )

    monkeypatch.setattr("importlib.util.find_spec", lambda name: None if name == "rclpy" else object())
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)
    monkeypatch.delenv("ROS_DISTRO", raising=False)

    original_sys_path = list(sys.path)
    try:
        bridge._ensure_ros_python_environment()
        assert os.environ["ROS_DISTRO"] == "humble"
        assert sys.path[0] == python_path_a
        assert sys.path[1] == python_path_b
    finally:
        sys.path[:] = original_sys_path


def test_go2_bridge_prefers_direct_dds_pose_when_available() -> None:
    bridge = RclpyGo2RosBridge(Go2DataPlaneConfig(enabled=True))

    bridge._on_odom(_build_pose_odom_message(x=1.0, y=0.0), source_topic="/utlidar/robot_odom")
    ros_pose = bridge.get_current_pose()
    ros_status = bridge.get_status()

    assert ros_pose is not None
    assert ros_pose.position.x == 1.0
    assert ros_status["pose_source"] == "/utlidar/robot_odom"
    assert ros_status["ros_pose_ready"] is True
    assert ros_status["dds_pose_ready"] is False

    bridge._on_sport_mode_state(
        _build_sport_mode_state_message(x=2.0, y=3.0),
        source_topic="dds:rt/sportmodestate",
    )
    dds_pose = bridge.get_current_pose()
    dds_status = bridge.get_status()

    assert dds_pose is not None
    assert dds_pose.position.x == 2.0
    assert dds_pose.position.y == 3.0
    assert dds_status["pose_source"] == "dds:rt/sportmodestate"
    assert dds_status["dds_pose_ready"] is True


def test_go2_bridge_can_synthesize_local_maps_from_direct_dds_point_cloud() -> None:
    bridge = RclpyGo2RosBridge(
        Go2DataPlaneConfig(
            enabled=True,
            map_synthesis=Go2MapSynthesisConfig(
                local_map_enabled=True,
                local_map_resolution_m=0.5,
                local_map_width=20,
                local_map_height=20,
                local_map_update_interval_sec=0.0,
                local_map_inflation_radius_m=0.5,
                semantic_min_cells=1,
            ),
        )
    )
    bridge._on_sport_mode_state(
        _build_sport_mode_state_message(x=0.0, y=0.0),
        source_topic="dds:rt/sportmodestate",
    )
    bridge._on_direct_point_cloud(
        _build_point_cloud_message([(1.0, 0.0, 0.10), (1.5, 0.0, 0.10)]),
        source_topic="dds:rt/utlidar/cloud",
    )

    occupancy = bridge.get_occupancy_grid()
    cost_map = bridge.get_cost_map()
    semantic_map = bridge.get_semantic_map()
    status = bridge.get_status()

    assert occupancy is not None
    assert cost_map is not None
    assert semantic_map is not None
    assert status["dds_cloud_ready"] is True
    assert status["local_map_ready"] is True
    assert status["local_map_source"] == "direct_dds_point_cloud"
    assert status["point_cloud_source_topic"] == "dds:rt/utlidar/cloud"

    obstacle_index = _grid_cell_index(
        width=occupancy.width,
        resolution=occupancy.resolution_m,
        origin_x=occupancy.origin.position.x,
        origin_y=occupancy.origin.position.y,
        x=1.0,
        y=0.0,
    )
    assert occupancy.data[obstacle_index] == 100
    assert cost_map.data[obstacle_index] == 100.0
    assert any(region.label == "hazard" for region in semantic_map.regions)


def test_go2_bridge_can_use_ros_point_cloud_as_local_map_fallback() -> None:
    bridge = RclpyGo2RosBridge(
        Go2DataPlaneConfig(
            enabled=True,
            direct_dds=Go2DirectDdsConfig(enabled=False),
            map_synthesis=Go2MapSynthesisConfig(
                local_map_enabled=True,
                local_map_resolution_m=0.5,
                local_map_width=20,
                local_map_height=20,
                local_map_update_interval_sec=0.0,
                semantic_min_cells=1,
            ),
        )
    )
    bridge._on_odom(_build_pose_odom_message(x=0.0, y=0.0), source_topic="/utlidar/robot_odom")
    bridge._on_point_cloud(
        _build_point_cloud_message([(0.5, 0.0, 0.10), (1.0, 0.0, 0.10)]),
        source_topic="/utlidar/cloud",
    )

    occupancy = bridge.get_occupancy_grid()
    status = bridge.get_status()

    assert occupancy is not None
    assert status["ros_point_cloud_ready"] is True
    assert status["dds_cloud_ready"] is False
    assert status["local_map_ready"] is True
    assert status["local_map_source"] == "ros_point_cloud"
    assert status["point_cloud_source_topic"] == "/utlidar/cloud"


def test_go2_cyclonedds_lib_dir_uses_configured_path(monkeypatch, tmp_path: Path) -> None:
    lib_dir = tmp_path / "cyclonedds" / "lib"
    lib_dir.mkdir(parents=True)
    (lib_dir / "libddsc.so.0.10.2").write_text("", encoding="utf-8")

    monkeypatch.setenv("GO2_CYCLONEDDS_LIB_DIR", str(lib_dir))

    config = go2_settings.load_go2_data_plane_config()

    assert config.cyclonedds_lib_dir == str(lib_dir)
    assert config.topics.odom_topic == "/utlidar/robot_odom"
    assert config.topics.imu_topic == "/utlidar/imu"
    assert config.topics.point_cloud_topic == "/utlidar/cloud"


def test_go2_shell_scripts_pass_bash_syntax_check() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    scripts = (
        repo_root / "drivers" / "robots" / "go2" / "source_runtime_env.sh",
    )
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_go2_data_plane_marks_exploration_unavailable_without_localization_or_map() -> None:
    config = Go2DataPlaneConfig(
        enabled=True,
        exploration=Go2ExplorationConfig(enabled=True),
    )
    runtime = Go2DataPlaneRuntime(
        config,
        bridge=_FakeBridge(localization_available=False, map_available=False, navigation_available=False),
    )
    runtime._official_sport_ready = True

    accepted = runtime.start_exploration(
        ExploreAreaRequest(request_id="explore_test", target_name="dock", strategy="frontier")
    )
    state = runtime.get_exploration_state()

    assert accepted is False
    assert runtime.is_exploration_available() is False
    assert state.status == ExplorationStatus.FAILED
    assert "定位" in (state.message or "")


def test_go2_data_plane_uses_official_navigation_when_localization_ready() -> None:
    config = Go2DataPlaneConfig(
        enabled=True,
        exploration=Go2ExplorationConfig(enabled=True),
    )
    runtime = Go2DataPlaneRuntime(
        config,
        bridge=_FakeBridge(localization_available=True, map_available=False, navigation_available=False),
    )
    runtime._official_sport_ready = True

    assert runtime.is_navigation_available() is True
    assert runtime.is_exploration_available() is True


def test_go2_data_plane_status_exposes_effective_navigation_availability() -> None:
    config = Go2DataPlaneConfig(
        enabled=True,
        exploration=Go2ExplorationConfig(enabled=True),
    )
    runtime = Go2DataPlaneRuntime(
        config,
        bridge=_FakeBridge(localization_available=True, map_available=True, navigation_available=False),
    )
    runtime._official_sport_ready = True
    status = runtime.get_status()

    assert status["navigation_available"] is True
    assert status["bridge"]["bridge_navigation_available"] is False
    assert status["bridge"]["navigation_available"] is True
    assert status["bridge"]["navigation_availability_source"] == "official_backend"


def test_go2_data_plane_exploration_falls_back_to_inner_radius_when_outer_ring_blocked() -> None:
    config = Go2DataPlaneConfig(
        enabled=True,
        exploration=Go2ExplorationConfig(
            enabled=True,
            sample_radius_m=1.5,
            sample_count=8,
            max_goal_cost=75.0,
        ),
    )
    runtime = Go2DataPlaneRuntime(
        config,
        bridge=_FakeBridge(localization_available=True, map_available=True, navigation_available=False),
    )
    runtime._official_sport_ready = True

    center_pose = Pose(
        frame_id="odom",
        position=Vector3(x=0.0, y=0.0, z=0.0),
        orientation=Quaternion(w=1.0),
    )
    width = 240
    resolution = 0.1
    origin = Pose(
        frame_id="odom",
        position=Vector3(x=-12.0, y=-12.0, z=0.0),
        orientation=Quaternion(w=1.0),
    )
    data = [100.0] * (width * width)
    for x, y, cost in (
        (1.5, 0.0, 75.0),
        (-1.5, 0.0, 50.0),
    ):
        index = _grid_cell_index(
            width=width,
            resolution=resolution,
            origin_x=origin.position.x,
            origin_y=origin.position.y,
            x=x,
            y=y,
        )
        data[index] = cost

    cost_map = CostMap(
        map_id="test_cost_map",
        frame_id="odom",
        width=width,
        height=width,
        resolution_m=resolution,
        origin=origin,
        data=data,
    )

    runtime.get_current_pose = lambda: center_pose
    runtime.get_cost_map = lambda: cost_map

    candidates = runtime._build_exploration_candidates(
        ExploreAreaRequest(
            request_id="explore_test",
            center_pose=center_pose,
            radius_m=5.0,
            strategy="frontier",
        )
    )

    assert candidates
    candidate_radii = [
        round((candidate.position.x**2 + candidate.position.y**2) ** 0.5, 3)
        for candidate in candidates
    ]
    assert max(candidate_radii) < 5.0
    assert min(candidate_radii) <= 1.582
