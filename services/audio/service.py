from __future__ import annotations

from collections import deque
import threading
import time

from contracts.events import RuntimeEventCategory
from core import EventBus
from settings import NuwaxRobotBridgeConfig
from typing import Deque, Dict, Optional, Tuple

try:
    from tts.doubao_realtime_client import SpeechSettings, build_doubao_realtime_client
    from tts.realtime_tts_player import build_player_from_config
except Exception:  # pragma: no cover - 运行环境缺依赖时走降级
    SpeechSettings = None
    build_doubao_realtime_client = None
    build_player_from_config = None


class AudioService:
    """统一音量与文本播报服务。"""

    def __init__(
        self,
        *,
        config: NuwaxRobotBridgeConfig,
        robot,
        event_bus: Optional[EventBus] = None,
        history_limit: int = 50,
    ) -> None:
        self._config = config
        self._robot = robot
        self._event_bus = event_bus
        self._lock = threading.RLock()
        self._player = None
        self._tts_client = None
        self._software_volume = max(0.0, min(1.0, float(config.tts.initial_volume)))
        self._speech_history: Deque[Dict[str, object]] = deque(maxlen=max(1, history_limit))
        self._last_error = ""

    def stop(self) -> None:
        """停止音频运行时。"""

        with self._lock:
            if self._tts_client is not None:
                try:
                    self._tts_client.stop()
                except Exception:
                    pass
                self._tts_client = None
            if self._player is not None:
                try:
                    self._player.stop()
                except Exception:
                    pass
                self._player = None

    def get_volume_state(self) -> Dict[str, object]:
        """读取当前音量状态。"""

        result: Dict[str, object] = {
            "volume": self._software_volume,
            "backend": "software",
            "realtime_speech_available": self._can_use_realtime_speech(),
            "last_error": self._last_error,
        }
        if hasattr(self._robot, "get_vui_volume_info"):
            try:
                robot_volume = self._robot.get_vui_volume_info()
                result["robot_volume"] = robot_volume
                result["volume"] = float(robot_volume.get("volume", result["volume"]))
                result["backend"] = str(robot_volume.get("backend") or "vui")
            except Exception as exc:
                result["robot_volume_error"] = str(exc)
        return result

    def set_volume(self, volume: float) -> Dict[str, object]:
        """设置当前音量。"""

        normalized = max(0.0, min(1.0, float(volume)))
        self._software_volume = normalized
        result: Dict[str, object] = {"volume": normalized, "backend": "software"}

        with self._lock:
            if self._player is not None:
                try:
                    result["player_volume"] = self._player.set_volume(normalized)
                except Exception as exc:
                    self._last_error = str(exc)
                    result["player_volume_error"] = str(exc)

        if hasattr(self._robot, "set_vui_volume_ratio"):
            try:
                result["robot_volume"] = self._robot.set_vui_volume_ratio(
                    normalized,
                    self._config.volume.auto_enable_vui_switch,
                )
                result["backend"] = "vui"
            except Exception as exc:
                self._last_error = str(exc)
                result["robot_volume_error"] = str(exc)
        self._publish_audio_event(
            "audio.volume_changed",
            "音量已更新。",
            payload={"volume": normalized, "backend": result.get("backend")},
        )
        return result

    def speak_text(
        self,
        text: str,
        *,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """提交一次文本播报。"""

        content = str(text or "").strip()
        if not content:
            raise ValueError("播报文本不能为空。")

        requested_session_id = session_id or f"speech_{time.time_ns()}"
        result: Dict[str, object] = {
            "accepted": True,
            "session_id": requested_session_id,
            "text": content,
            "mode": "record_only",
            "realtime_started": False,
        }

        try:
            if self._ensure_realtime_runtime():
                settings = self._build_speech_settings()
                if settings is not None and self._tts_client is not None:
                    actual_session_id = self._tts_client.start_stream(settings, session_id=requested_session_id)
                    self._tts_client.append_stream(actual_session_id, content)
                    self._tts_client.finish_stream(actual_session_id)
                    result["session_id"] = actual_session_id
                    result["mode"] = "doubao_realtime"
                    result["realtime_started"] = True
        except Exception as exc:
            self._last_error = str(exc)
            result["mode"] = "record_only"
            result["realtime_error"] = str(exc)

        record = {
            **result,
            "metadata": dict(metadata or {}),
            "timestamp_ms": int(time.time() * 1000),
        }
        self._speech_history.append(record)
        self._publish_audio_event(
            "audio.speech_requested",
            "已接收文本播报请求。",
            payload={"session_id": result["session_id"], "mode": result["mode"]},
        )
        return record

    def list_speech_history(self, *, limit: Optional[int] = None) -> Tuple[Dict[str, object], ...]:
        """列出近期播报记录。"""

        items = list(self._speech_history)
        if limit is not None:
            items = items[-max(0, limit) :]
        return tuple(items)

    def is_volume_available(self) -> bool:
        """判断音量控制是否可用。"""

        return True

    def is_speech_available(self) -> bool:
        """判断文本播报是否可用。

        首版即使没有实时语音后端，也允许进入记录模式，因此始终返回 True。
        """

        return True

    def _ensure_realtime_runtime(self) -> bool:
        if not self._can_use_realtime_speech():
            return False
        with self._lock:
            if self._player is None:
                self._player = build_player_from_config(self._config.tts)
                self._player.start()
                self._player.set_volume(self._software_volume)
            if self._tts_client is None:
                self._tts_client = build_doubao_realtime_client(self._player, self._config.doubao)
                self._tts_client.start()
        return True

    def _can_use_realtime_speech(self) -> bool:
        return bool(
            SpeechSettings is not None
            and build_player_from_config is not None
            and build_doubao_realtime_client is not None
            and self._config.doubao.app_id
            and self._config.doubao.access_key
            and (self._config.log_tts.speaker or self._config.doubao.default_speaker)
        )

    def _build_speech_settings(self):
        if not self._can_use_realtime_speech() or SpeechSettings is None:
            return None
        speaker = (self._config.log_tts.speaker or self._config.doubao.default_speaker).strip()
        return SpeechSettings(
            speaker=speaker,
            resource_id=(self._config.log_tts.resource_id or self._config.doubao.resource_id).strip()
            or self._config.doubao.resource_id,
            uid=(self._config.log_tts.uid or self._config.doubao.default_uid).strip()
            or self._config.doubao.default_uid,
            audio_format=(self._config.log_tts.audio_format or self._config.doubao.default_audio_format).strip()
            or self._config.doubao.default_audio_format,
            sample_rate=int(self._config.log_tts.sample_rate or self._config.doubao.default_sample_rate),
            speech_rate=int(self._config.log_tts.speech_rate),
            loudness_rate=int(self._config.log_tts.loudness_rate),
            emotion=self._config.log_tts.emotion.strip(),
            emotion_scale=int(self._config.log_tts.emotion_scale),
            enable_subtitle=bool(self._config.log_tts.enable_subtitle),
            enable_timestamp=bool(self._config.log_tts.enable_timestamp),
            disable_markdown_filter=bool(self._config.log_tts.disable_markdown_filter),
            silence_duration_ms=int(self._config.log_tts.silence_duration_ms),
            explicit_language=self._config.log_tts.explicit_language.strip(),
            model=(self._config.log_tts.model or self._config.doubao.default_model).strip(),
            interrupt=True,
            metadata={},
        )

    def _publish_audio_event(self, event_type: str, message: str, *, payload: Dict[str, object]) -> None:
        if self._event_bus is None:
            return
        self._event_bus.publish(
            self._event_bus.build_event(
                event_type,
                category=RuntimeEventCategory.SYSTEM,
                source="audio_service",
                message=message,
                payload=payload,
            )
        )
