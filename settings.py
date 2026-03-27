#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
nuwax_robot_bridge 统一配置解析层。

说明：
1. 用户直接编辑同目录下的 `.env` 文件。
2. 为兼容旧部署，也会回退读取 `.config`。
3. 同名环境变量优先级高于文件配置。
4. 这里仅负责把配置解析成结构化对象，业务逻辑不要直接散落读取环境变量。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import threading


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
LEGACY_CONFIG_FILE = BASE_DIR / ".config"


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[:1] == value[-1:] and value[:1] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_file_values() -> dict[str, str]:
    values = {}
    for file_path in (LEGACY_CONFIG_FILE, ENV_FILE):
        if not file_path.exists():
            continue
        for raw_line in file_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            values[key] = _strip_quotes(value)
    return values


_FILE_VALUES = _load_file_values()


def _raw_value(name: str, default: str = "") -> str:
    if name in os.environ:
        return os.environ[name]
    return _FILE_VALUES.get(name, default)


def _cfg_str(name: str, default: str = "") -> str:
    value = _raw_value(name, default)
    value = value.strip()
    return value if value else default


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


def _cfg_bool(name: str, default: bool) -> bool:
    value = _raw_value(name, "")
    if not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _clamp_volume(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class LoggingConfig:
    """日志配置。"""

    log_dir: str
    log_file: str
    level: str = "INFO"
    max_lines: int = 100000
    trim_check_interval: int = 200


@dataclass
class TcpServerConfig:
    """TCP 控制服务配置。"""

    host: str = "0.0.0.0"
    port: int = 8765


@dataclass
class DDSConfig:
    """DDS 与 SDK 路径配置。"""

    cyclone_dds_uri: str
    iface: str = ""
    sdk_path: str = "/home/unitree/unitree_sdk2_python"


@dataclass
class LowLevelControlConfig:
    """低级控制默认参数。"""

    default_kp: float = 60.0
    default_kd: float = 5.0
    max_velocity: float = 2.0
    max_torque: float = 10.0
    control_interval_sec: float = 0.002


@dataclass
class LocalPlaybackConfig:
    """Jetson 本地声卡兼容播放链路配置。"""

    preferred_backend: str = "aplay"
    audio_device: str = ""
    enable_spk_ctl: bool = True
    enable_ape_route: bool = True
    ape_card: str = "APE"
    ape_admaif: int = 1
    ape_speaker: str = "DSPK1"


@dataclass
class RobotWebRTCPlaybackConfig:
    """Go2 机身扬声器 WebRTC 播放链路配置。"""

    robot_ip: str = ""
    connection_mode: str = "ai"
    python_path: str = ""
    use_megaphone: bool = True
    streaming_enabled: bool = False
    streaming_chunk_ms: int = 200
    connect_timeout: float = 15.0
    request_timeout: float = 20.0
    upload_chunk_size: int = 61440
    megaphone_chunk_size: int = 16384
    sentence_segment_min_duration_sec: float = 3.0
    auto_fallback_to_local: bool = True


@dataclass
class TTSPlayerConfig:
    """本地播放器配置。"""

    auto_start: bool = True
    print_subtitle: bool = False
    initial_volume: float = 1.0
    backend_mode: str = "robot_webrtc"
    fallback_backend_mode: str = "alsa_local"
    local: LocalPlaybackConfig = field(default_factory=LocalPlaybackConfig)
    robot: RobotWebRTCPlaybackConfig = field(default_factory=RobotWebRTCPlaybackConfig)


@dataclass
class DoubaoConfig:
    """豆包鉴权与默认参数配置。"""

    app_id: str = ""
    access_key: str = ""
    resource_id: str = "seed-tts-1.0"
    default_speaker: str = ""
    default_uid: str = "nuwax_robot_bridge-tts"
    default_model: str = ""
    default_audio_format: str = "pcm"
    default_sample_rate: int = 24000
    request_usage_tokens: bool = True
    send_app_id_header: bool = False
    send_app_key_header: bool = True


@dataclass
class TTSLogBridgeConfig:
    """日志监听转 TTS 的配置。"""

    enabled: bool = True
    log_path: str = "/var/lib/docker/volumes/nuwax_home/_data/.nuwaxbot/logs/main.log"
    require_root: bool = True
    speaker: str = ""
    resource_id: str = "seed-tts-1.0"
    uid: str = "nuwax_robot_bridge-tts"
    model: str = ""
    audio_format: str = "pcm"
    sample_rate: int = 24000
    speech_rate: int = 0
    loudness_rate: int = 0
    emotion: str = ""
    emotion_scale: int = 4
    enable_subtitle: bool = False
    enable_timestamp: bool = False
    disable_markdown_filter: bool = True
    silence_duration_ms: int = 0
    interrupt: bool = True
    explicit_language: str = "zh"
    start_from_end: bool = True
    poll_interval_sec: float = 0.2
    quiet_period_sec: float = 1.5
    flush_interval_sec: float = 0.35
    stream_min_chars: int = 8
    min_text_length: int = 2
    request_timeout_sec: float = 15.0
    start_retry_backoff_sec: float = 2.0
    max_start_failures_per_turn: int = 3


@dataclass
class VolumeControlConfig:
    """音量控制配置。

    默认优先走 Go2 的 VUI 音量，Jetson 本地混音器仅作为兼容链路保留。
    """

    prefer_vui: bool = True
    auto_enable_vui_switch: bool = True
    vui_min_level: int = 0
    vui_max_level: int = 10
    init_volume_on_start: bool = False
    mixer_device: str = "default"
    mixer_control: str = ""
    audio_command_timeout: float = 1.0


@dataclass
class NuwaxRobotBridgeConfig:
    """项目总配置。"""

    base_dir: str
    logging: LoggingConfig
    tcp: TcpServerConfig
    dds: DDSConfig
    low_level: LowLevelControlConfig
    tts: TTSPlayerConfig
    doubao: DoubaoConfig
    log_tts: TTSLogBridgeConfig
    volume: VolumeControlConfig


def load_config() -> NuwaxRobotBridgeConfig:
    """统一加载配置。"""

    log_dir = _cfg_str("GO2_PROXY_LOG_DIR", str(BASE_DIR))
    log_file = _cfg_str("GO2_PROXY_LOG_FILE", str(Path(log_dir) / "go2_proxy.log"))

    local_playback = LocalPlaybackConfig(
        preferred_backend=_cfg_str("TTS_PLAYER_PREFERRED_BACKEND", "aplay"),
        audio_device=_cfg_str("TTS_PLAYER_AUDIO_DEVICE", ""),
        enable_spk_ctl=_cfg_bool("TTS_PLAYER_ENABLE_SPK_CTL", True),
        enable_ape_route=_cfg_bool("TTS_PLAYER_ENABLE_APE_ROUTE", True),
        ape_card=_cfg_str("TTS_PLAYER_APE_CARD", "APE"),
        ape_admaif=_cfg_int("TTS_PLAYER_APE_ADMAIF", 1),
        ape_speaker=_cfg_str("TTS_PLAYER_APE_SPEAKER", "DSPK1"),
    )

    robot_playback = RobotWebRTCPlaybackConfig(
        robot_ip=_cfg_str("TTS_PLAYER_ROBOT_IP", _cfg_str("GO2_ROBOT_IP", _cfg_str("ROBOT_IP", ""))),
        connection_mode=_cfg_str("TTS_PLAYER_ROBOT_MODE", "ai"),
        python_path=_cfg_str("TTS_PLAYER_ROBOT_PYTHONPATH", ""),
        use_megaphone=_cfg_bool("TTS_PLAYER_USE_MEGAPHONE", True),
        streaming_enabled=_cfg_bool("TTS_PLAYER_STREAMING_ENABLED", False),
        streaming_chunk_ms=_cfg_int("TTS_PLAYER_STREAMING_CHUNK_MS", 200),
        connect_timeout=_cfg_float("TTS_PLAYER_ROBOT_CONNECT_TIMEOUT", 15.0),
        request_timeout=_cfg_float("TTS_PLAYER_ROBOT_REQUEST_TIMEOUT", 20.0),
        upload_chunk_size=_cfg_int("TTS_PLAYER_UPLOAD_CHUNK_SIZE", 61440),
        megaphone_chunk_size=_cfg_int("TTS_PLAYER_MEGAPHONE_CHUNK_SIZE", 16384),
        sentence_segment_min_duration_sec=_cfg_float(
            "TTS_PLAYER_SEGMENT_MIN_DURATION_SEC",
            3.0,
        ),
        auto_fallback_to_local=_cfg_bool("TTS_PLAYER_AUTO_FALLBACK_TO_LOCAL", True),
    )

    tts_player = TTSPlayerConfig(
        auto_start=_cfg_bool("TTS_PLAYER_AUTO_START", True),
        print_subtitle=_cfg_bool("TTS_PLAYER_PRINT_SUBTITLE", False),
        initial_volume=_clamp_volume(_cfg_float("TTS_PLAYER_VOLUME", 1.0)),
        backend_mode=_cfg_str("TTS_PLAYER_BACKEND_MODE", "robot_webrtc"),
        fallback_backend_mode=_cfg_str("TTS_PLAYER_FALLBACK_BACKEND_MODE", "alsa_local"),
        local=local_playback,
        robot=robot_playback,
    )

    doubao = DoubaoConfig(
        app_id=_cfg_str("DOUBAO_APP_ID", ""),
        access_key=_cfg_str("DOUBAO_ACCESS_KEY", ""),
        resource_id=_cfg_str("DOUBAO_RESOURCE_ID", "seed-tts-1.0"),
        default_speaker=_cfg_str("DOUBAO_DEFAULT_SPEAKER", ""),
        default_uid=_cfg_str("DOUBAO_DEFAULT_UID", "nuwax_robot_bridge-tts"),
        default_model=_cfg_str("DOUBAO_DEFAULT_MODEL", ""),
        default_audio_format=_cfg_str("DOUBAO_DEFAULT_AUDIO_FORMAT", "pcm"),
        default_sample_rate=_cfg_int("DOUBAO_DEFAULT_SAMPLE_RATE", 24000),
        request_usage_tokens=_cfg_bool("DOUBAO_REQUEST_USAGE_TOKENS", True),
        send_app_id_header=_cfg_bool("DOUBAO_SEND_APP_ID_HEADER", False),
        send_app_key_header=_cfg_bool("DOUBAO_SEND_APP_KEY_HEADER", True),
    )

    log_tts = TTSLogBridgeConfig(
        enabled=_cfg_bool("TTS_LOG_BRIDGE_ENABLED", True),
        log_path=_cfg_str(
            "TTS_LOG_BRIDGE_LOG_PATH",
            "/var/lib/docker/volumes/nuwax_home/_data/.nuwaxbot/logs/main.log",
        ),
        require_root=_cfg_bool("TTS_LOG_BRIDGE_REQUIRE_ROOT", True),
        speaker=_cfg_str("TTS_LOG_BRIDGE_SPEAKER", doubao.default_speaker),
        resource_id=_cfg_str("TTS_LOG_BRIDGE_RESOURCE_ID", doubao.resource_id),
        uid=_cfg_str("TTS_LOG_BRIDGE_UID", doubao.default_uid),
        model=_cfg_str("TTS_LOG_BRIDGE_MODEL", doubao.default_model),
        audio_format=_cfg_str("TTS_LOG_BRIDGE_AUDIO_FORMAT", doubao.default_audio_format),
        sample_rate=_cfg_int("TTS_LOG_BRIDGE_SAMPLE_RATE", doubao.default_sample_rate),
        speech_rate=_cfg_int("TTS_LOG_BRIDGE_SPEECH_RATE", 0),
        loudness_rate=_cfg_int("TTS_LOG_BRIDGE_LOUDNESS_RATE", 0),
        emotion=_cfg_str("TTS_LOG_BRIDGE_EMOTION", ""),
        emotion_scale=_cfg_int("TTS_LOG_BRIDGE_EMOTION_SCALE", 4),
        enable_subtitle=_cfg_bool("TTS_LOG_BRIDGE_ENABLE_SUBTITLE", False),
        enable_timestamp=_cfg_bool("TTS_LOG_BRIDGE_ENABLE_TIMESTAMP", False),
        disable_markdown_filter=_cfg_bool("TTS_LOG_BRIDGE_DISABLE_MARKDOWN_FILTER", True),
        silence_duration_ms=_cfg_int("TTS_LOG_BRIDGE_SILENCE_DURATION_MS", 0),
        interrupt=_cfg_bool("TTS_LOG_BRIDGE_INTERRUPT", True),
        explicit_language=_cfg_str("TTS_LOG_BRIDGE_EXPLICIT_LANGUAGE", "zh"),
        start_from_end=_cfg_bool("TTS_LOG_BRIDGE_START_FROM_END", True),
        poll_interval_sec=_cfg_float("TTS_LOG_BRIDGE_POLL_INTERVAL_SEC", 0.2),
        quiet_period_sec=_cfg_float("TTS_LOG_BRIDGE_QUIET_PERIOD_SEC", 1.5),
        flush_interval_sec=_cfg_float("TTS_LOG_BRIDGE_FLUSH_INTERVAL_SEC", 0.35),
        stream_min_chars=_cfg_int("TTS_LOG_BRIDGE_STREAM_MIN_CHARS", 8),
        min_text_length=_cfg_int("TTS_LOG_BRIDGE_MIN_TEXT_LENGTH", 2),
        request_timeout_sec=_cfg_float("TTS_LOG_BRIDGE_REQUEST_TIMEOUT_SEC", 15.0),
        start_retry_backoff_sec=_cfg_float("TTS_LOG_BRIDGE_START_RETRY_BACKOFF_SEC", 2.0),
        max_start_failures_per_turn=_cfg_int("TTS_LOG_BRIDGE_MAX_START_FAILURES_PER_TURN", 3),
    )

    volume = VolumeControlConfig(
        prefer_vui=_cfg_bool("GO2_VOLUME_PREFER_VUI", True),
        auto_enable_vui_switch=_cfg_bool("GO2_VOLUME_AUTO_ENABLE_VUI_SWITCH", True),
        vui_min_level=_cfg_int("GO2_VUI_MIN_LEVEL", 0),
        vui_max_level=_cfg_int("GO2_VUI_MAX_LEVEL", 10),
        init_volume_on_start=_cfg_bool("GO2_INIT_VOLUME_ON_START", False),
        mixer_device=_cfg_str("GO2_VOLUME_MIXER_DEVICE", "default"),
        mixer_control=_cfg_str("GO2_VOLUME_MIXER_CONTROL", ""),
        audio_command_timeout=_cfg_float("GO2_AUDIO_COMMAND_TIMEOUT", 1.0),
    )

    dds = DDSConfig(
        cyclone_dds_uri=_cfg_str(
            "CYCLONEDDS_URI",
            "<CycloneDDS><Domain><Compatibility><StandardsConformance>lax</StandardsConformance></Compatibility></Domain></CycloneDDS>",
        ),
        iface=_cfg_str("GO2_DDS_IFACE", ""),
        sdk_path=_cfg_str("GO2_SDK_PATH", "/home/unitree/unitree_sdk2_python"),
    )

    low_level = LowLevelControlConfig(
        default_kp=_cfg_float("GO2_LOW_LEVEL_KP", 60.0),
        default_kd=_cfg_float("GO2_LOW_LEVEL_KD", 5.0),
        max_velocity=_cfg_float("GO2_LOW_LEVEL_MAX_VELOCITY", 2.0),
        max_torque=_cfg_float("GO2_LOW_LEVEL_MAX_TORQUE", 10.0),
        control_interval_sec=_cfg_float("GO2_LOW_LEVEL_INTERVAL_SEC", 0.002),
    )

    return NuwaxRobotBridgeConfig(
        base_dir=str(BASE_DIR),
        logging=LoggingConfig(
            log_dir=log_dir,
            log_file=log_file,
            level=_cfg_str("GO2_PROXY_LOG_LEVEL", "INFO"),
            max_lines=max(0, _cfg_int("GO2_PROXY_LOG_MAX_LINES", 100000)),
            trim_check_interval=max(1, _cfg_int("GO2_PROXY_LOG_TRIM_CHECK_INTERVAL", 200)),
        ),
        tcp=TcpServerConfig(
            host=_cfg_str("GO2_PROXY_TCP_HOST", "0.0.0.0"),
            port=_cfg_int("GO2_PROXY_TCP_PORT", 8765),
        ),
        dds=dds,
        low_level=low_level,
        tts=tts_player,
        doubao=doubao,
        log_tts=log_tts,
        volume=volume,
    )


class _NoisyLogFilter(logging.Filter):
    """过滤第三方库的高频低价值日志，避免 proxy.log 爆量。"""

    _DROP_PATTERNS = (
        "Received message on data channel:",
        "> message sent:",
        "Heartbeat response received.",
        "Creating offer...",
        "Trying to send SDP using the old method...",
        "An error occurred with the old method:",
        "Data channel closed",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True

        for pattern in self._DROP_PATTERNS:
            if pattern in message:
                return False

        if "HTTPConnectionPool(" in message and "/offer" in message and "Connection refused" in message:
            return False

        if "MediaStreamError" in message and "AsyncIOEventEmitter" in message:
            return False

        return True


class _LineLimitedFileHandler(logging.FileHandler):
    """按最大行数保留日志文件，超出后自动裁剪前面的旧内容。"""

    def __init__(
        self,
        filename: str,
        *,
        max_lines: int,
        trim_check_interval: int,
        **kwargs,
    ) -> None:
        super().__init__(filename, **kwargs)
        self.max_lines = max(0, int(max_lines))
        self.trim_check_interval = max(1, int(trim_check_interval))
        self._emit_count = 0
        self._trim_lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        if self.max_lines <= 0:
            return

        self._emit_count += 1
        if self._emit_count % self.trim_check_interval != 0:
            return

        self._trim_file_if_needed()

    def _trim_file_if_needed(self) -> None:
        if not self._trim_lock.acquire(blocking=False):
            return

        try:
            self.flush()
            path = Path(self.baseFilename)
            if not path.exists():
                return

            tail_lines: deque[str] = deque(maxlen=self.max_lines)
            total_lines = 0
            with path.open("r", encoding=self.encoding or "utf-8", errors="ignore") as stream:
                for raw_line in stream:
                    total_lines += 1
                    tail_lines.append(raw_line)

            if total_lines <= self.max_lines:
                return

            if self.stream is not None:
                try:
                    self.stream.close()
                finally:
                    self.stream = None

            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with tmp_path.open("w", encoding=self.encoding or "utf-8") as stream:
                stream.writelines(tail_lines)
            tmp_path.replace(path)
            self.stream = self._open()
        finally:
            self._trim_lock.release()


def _configure_third_party_logger_levels() -> None:
    """压低第三方库默认日志级别，保留我们自己的业务日志。"""

    for logger_name in (
        "aiortc",
        "aioice",
        "asyncio",
        "av",
        "pyee",
        "unitree_webrtc_connect",
        "websockets",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def configure_logging(config: LoggingConfig) -> None:
    """统一初始化日志。"""

    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    level_name = (config.level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    noisy_filter = _NoisyLogFilter()

    file_handler = _LineLimitedFileHandler(
        config.log_file,
        max_lines=config.max_lines,
        trim_check_interval=config.trim_check_interval,
        encoding="utf-8",
    )
    file_handler.addFilter(noisy_filter)

    console_handler = logging.StreamHandler()
    console_handler.addFilter(noisy_filter)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[file_handler, console_handler],
        force=True,
    )
    _configure_third_party_logger_levels()


APP_CONFIG = load_config()
