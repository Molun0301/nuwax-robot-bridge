from __future__ import annotations

from pathlib import Path

from drivers.robots.go2 import settings as go2_settings


def test_go2_ros2_workspace_defaults_point_to_repo() -> None:
    assert go2_settings._DEFAULT_GO2_SETUP_SCRIPT.endswith("drivers/robots/go2/ros2_ws/install/setup.bash")
    assert go2_settings._DEFAULT_GO2_LIDAR_LAUNCH_COMMAND == "ros2 launch nuwax_go2_bringup go2_lidar_mapping.launch.py"
    assert go2_settings._DEFAULT_GO2_NAV2_LAUNCH_COMMAND == "ros2 launch nuwax_go2_bringup go2_nav2.launch.py"


def test_go2_ros2_workspace_assets_exist() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    workspace_root = repo_root / "drivers" / "robots" / "go2" / "ros2_ws"
    assert (workspace_root / "src" / "nuwax_go2_bringup" / "package.xml").exists()
    assert (workspace_root / "src" / "nuwax_go2_bringup" / "launch" / "go2_full_stack.launch.py").exists()
    assert (repo_root / "drivers" / "robots" / "go2" / "build_ros2_workspace.sh").exists()
    assert (repo_root / "drivers" / "robots" / "go2" / "launch_ros2_stack.sh").exists()
