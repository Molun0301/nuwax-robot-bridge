from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, Tuple


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ENV_FILE = _PROJECT_ROOT / ".env"
_LEGACY_CONFIG_FILE = _PROJECT_ROOT / ".config"
_GO2_ENV_FILE = Path(__file__).resolve().with_name("data_plane.env")
_GO2_ROS2_WS_DIR = Path(__file__).resolve().parent / "ros2_ws"
_DEFAULT_GO2_SETUP_SCRIPT = str(_GO2_ROS2_WS_DIR / "install" / "setup.bash")
_DEFAULT_GO2_LIDAR_LAUNCH_COMMAND = "ros2 launch nuwax_go2_bringup go2_lidar_mapping.launch.py"
_DEFAULT_GO2_NAV2_LAUNCH_COMMAND = "ros2 launch nuwax_go2_bringup go2_nav2.launch.py"


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[:1] == value[-1:] and value[:1] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_file_values() -> Dict[str, str]:
    values: Dict[str, str] = {}
    for file_path in (_LEGACY_CONFIG_FILE, _ENV_FILE, _GO2_ENV_FILE):
        if not file_path.exists():
            continue
        for raw_line in file_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key:
                values[key] = _strip_quotes(value)
    return values


_FILE_VALUES = _load_file_values()


def _raw_value(name: str, default: str = "") -> str:
    if name in os.environ:
        return os.environ[name]
    return _FILE_VALUES.get(name, default)


def _cfg_str(name: str, default: str = "") -> str:
    value = _raw_value(name, default).strip()
    return value if value else default


def _cfg_bool(name: str, default: bool) -> bool:
    value = _raw_value(name, "")
    if not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _cfg_int(name: str, default: int) -> int:
    value = _raw_value(name, "")
    if not value.strip():
        return default
    return int(value.strip())


def _cfg_float(name: str, default: float) -> float:
    value = _raw_value(name, "")
    if not value.strip():
        return default
    return float(value.strip())


def _cfg_csv(name: str, default: Tuple[str, ...] = ()) -> Tuple[str, ...]:
    value = _raw_value(name, "")
    if not value.strip():
        return default
    result = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if item:
            result.append(item)
    return tuple(result)


@dataclass
class Go2LidarPipelineConfig:
    """Go2 激光雷达与建图数据面启动配置。"""

    enabled: bool = False
    auto_launch: bool = False
    setup_script: str = _DEFAULT_GO2_SETUP_SCRIPT
    launch_commands: Tuple[str, ...] = (
        _DEFAULT_GO2_LIDAR_LAUNCH_COMMAND,
    )
    launch_stagger_sec: float = 2.0


@dataclass
class Go2Ros2TopicsConfig:
    """Go2 ROS2 数据面话题配置。"""

    odom_topic: str = "/odom_lio"
    tf_topic: str = "/tf"
    tf_static_topic: str = "/tf_static"
    occupancy_topic: str = "/map"
    grid_map_topic: str = "/elevation_mapping_cupy/elevation_map_raw"
    cost_map_topic: str = "/global_costmap/costmap"
    navigate_to_pose_action: str = "/navigate_to_pose"


@dataclass
class Go2MapSynthesisConfig:
    """Go2 地图合成与语义层提取配置。"""

    map_frame: str = "odom"
    base_frame: str = "body"
    traversability_layer: str = "traversability"
    validity_layer: str = ""
    free_threshold: float = 0.65
    lethal_threshold: float = 0.20
    semantic_layers: Tuple[str, ...] = ()
    semantic_threshold: float = 0.55
    semantic_min_cells: int = 25


@dataclass
class Go2Nav2Config:
    """Go2 Nav2 接入配置。"""

    enabled: bool = False
    auto_launch: bool = False
    setup_script: str = _DEFAULT_GO2_SETUP_SCRIPT
    launch_command: str = _DEFAULT_GO2_NAV2_LAUNCH_COMMAND
    action_wait_timeout_sec: float = 8.0
    launch_stabilize_sec: float = 5.0


@dataclass
class Go2ExplorationConfig:
    """Go2 探索后端配置。"""

    enabled: bool = False
    sample_radius_m: float = 1.5
    sample_count: int = 8
    max_goal_cost: float = 75.0
    goal_timeout_sec: float = 20.0
    reach_distance_m: float = 0.45
    status_poll_interval_sec: float = 0.2


@dataclass
class Go2DataPlaneConfig:
    """Go2 定位、地图、导航、探索数据面配置。"""

    enabled: bool = False
    setup_script: str = ""
    require_ros2: bool = False
    dds_iface: str = ""
    topics: Go2Ros2TopicsConfig = field(default_factory=Go2Ros2TopicsConfig)
    map_synthesis: Go2MapSynthesisConfig = field(default_factory=Go2MapSynthesisConfig)
    lidar_pipeline: Go2LidarPipelineConfig = field(default_factory=Go2LidarPipelineConfig)
    nav2: Go2Nav2Config = field(default_factory=Go2Nav2Config)
    exploration: Go2ExplorationConfig = field(default_factory=Go2ExplorationConfig)


def load_go2_data_plane_config() -> Go2DataPlaneConfig:
    """加载 Go2 端侧定位/地图/Nav2/探索配置。"""

    return Go2DataPlaneConfig(
        enabled=_cfg_bool("GO2_DATA_PLANE_ENABLED", False),
        setup_script=_cfg_str("GO2_DATA_PLANE_SETUP_SCRIPT", _DEFAULT_GO2_SETUP_SCRIPT),
        require_ros2=_cfg_bool("GO2_DATA_PLANE_REQUIRE_ROS2", False),
        dds_iface=_cfg_str("GO2_DATA_PLANE_DDS_IFACE", _cfg_str("GO2_DDS_IFACE", "")),
        topics=Go2Ros2TopicsConfig(
            odom_topic=_cfg_str("GO2_DATA_PLANE_ODOM_TOPIC", "/odom_lio"),
            tf_topic=_cfg_str("GO2_DATA_PLANE_TF_TOPIC", "/tf"),
            tf_static_topic=_cfg_str("GO2_DATA_PLANE_TF_STATIC_TOPIC", "/tf_static"),
            occupancy_topic=_cfg_str("GO2_DATA_PLANE_OCCUPANCY_TOPIC", "/map"),
            grid_map_topic=_cfg_str("GO2_DATA_PLANE_GRID_MAP_TOPIC", "/elevation_mapping_cupy/elevation_map_raw"),
            cost_map_topic=_cfg_str("GO2_DATA_PLANE_COST_MAP_TOPIC", "/global_costmap/costmap"),
            navigate_to_pose_action=_cfg_str("GO2_NAV2_NAVIGATE_TO_POSE_ACTION", "/navigate_to_pose"),
        ),
        map_synthesis=Go2MapSynthesisConfig(
            map_frame=_cfg_str("GO2_DATA_PLANE_MAP_FRAME", "odom"),
            base_frame=_cfg_str("GO2_DATA_PLANE_BASE_FRAME", "body"),
            traversability_layer=_cfg_str("GO2_DATA_PLANE_TRAVERSABILITY_LAYER", "traversability"),
            validity_layer=_cfg_str("GO2_DATA_PLANE_VALIDITY_LAYER", ""),
            free_threshold=max(0.0, min(1.0, _cfg_float("GO2_DATA_PLANE_FREE_THRESHOLD", 0.65))),
            lethal_threshold=max(0.0, min(1.0, _cfg_float("GO2_DATA_PLANE_LETHAL_THRESHOLD", 0.20))),
            semantic_layers=_cfg_csv("GO2_DATA_PLANE_SEMANTIC_LAYERS", ()),
            semantic_threshold=max(0.0, min(1.0, _cfg_float("GO2_DATA_PLANE_SEMANTIC_THRESHOLD", 0.55))),
            semantic_min_cells=max(1, _cfg_int("GO2_DATA_PLANE_SEMANTIC_MIN_CELLS", 25)),
        ),
        lidar_pipeline=Go2LidarPipelineConfig(
            enabled=_cfg_bool("GO2_LIDAR_PIPELINE_ENABLED", False),
            auto_launch=_cfg_bool("GO2_LIDAR_PIPELINE_AUTO_LAUNCH", False),
            setup_script=_cfg_str("GO2_LIDAR_PIPELINE_SETUP_SCRIPT", _DEFAULT_GO2_SETUP_SCRIPT),
            launch_commands=tuple(
                command
                for command in (
                    _cfg_str("GO2_LIDAR_PIPELINE_LAUNCH_1", _DEFAULT_GO2_LIDAR_LAUNCH_COMMAND),
                    _cfg_str("GO2_LIDAR_PIPELINE_LAUNCH_2", ""),
                    _cfg_str("GO2_LIDAR_PIPELINE_LAUNCH_3", ""),
                )
                if command.strip()
            ),
            launch_stagger_sec=max(0.0, _cfg_float("GO2_LIDAR_PIPELINE_LAUNCH_STAGGER_SEC", 2.0)),
        ),
        nav2=Go2Nav2Config(
            enabled=_cfg_bool("GO2_NAV2_ENABLED", False),
            auto_launch=_cfg_bool("GO2_NAV2_AUTO_LAUNCH", False),
            setup_script=_cfg_str("GO2_NAV2_SETUP_SCRIPT", _DEFAULT_GO2_SETUP_SCRIPT),
            launch_command=_cfg_str(
                "GO2_NAV2_LAUNCH_COMMAND",
                _DEFAULT_GO2_NAV2_LAUNCH_COMMAND,
            ),
            action_wait_timeout_sec=max(0.1, _cfg_float("GO2_NAV2_ACTION_WAIT_TIMEOUT_SEC", 8.0)),
            launch_stabilize_sec=max(0.0, _cfg_float("GO2_NAV2_LAUNCH_STABILIZE_SEC", 5.0)),
        ),
        exploration=Go2ExplorationConfig(
            enabled=_cfg_bool("GO2_EXPLORATION_ENABLED", False),
            sample_radius_m=max(0.1, _cfg_float("GO2_EXPLORATION_SAMPLE_RADIUS_M", 1.5)),
            sample_count=max(1, _cfg_int("GO2_EXPLORATION_SAMPLE_COUNT", 8)),
            max_goal_cost=max(0.0, _cfg_float("GO2_EXPLORATION_MAX_GOAL_COST", 75.0)),
            goal_timeout_sec=max(0.1, _cfg_float("GO2_EXPLORATION_GOAL_TIMEOUT_SEC", 20.0)),
            reach_distance_m=max(0.0, _cfg_float("GO2_EXPLORATION_REACH_DISTANCE_M", 0.45)),
            status_poll_interval_sec=max(0.05, _cfg_float("GO2_EXPLORATION_STATUS_POLL_INTERVAL_SEC", 0.2)),
        ),
    )


GO2_DATA_PLANE_CONFIG = load_go2_data_plane_config()
