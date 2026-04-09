from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, Hashable, Mapping, Optional


_STANDARD_LOG_RECORD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__.keys()) | {"message", "asctime"}
DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"


def _stringify_log_value(value: Any) -> str:
    """把日志扩展字段压成单行可读文本。"""

    if value is None:
        return "-"
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "-"
        if any(char.isspace() for char in text) or "=" in text:
            return json.dumps(text, ensure_ascii=False)
        return text
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return ("%.3f" % value).rstrip("0").rstrip(".")
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            return str(value)
    return str(value)


class ContextAwareFormatter(logging.Formatter):
    """在标准日志后自动追加 extra（扩展）字段。"""

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        extras = []
        for key, value in sorted(record.__dict__.items()):
            if key in _STANDARD_LOG_RECORD_FIELDS or key.startswith("_"):
                continue
            extras.append(f"{key}={_stringify_log_value(value)}")
        if not extras:
            return rendered
        return "%s %s" % (rendered, " ".join(extras))


def build_default_formatter() -> ContextAwareFormatter:
    """构造项目统一日志 formatter（格式化器）。"""

    return ContextAwareFormatter(DEFAULT_LOG_FORMAT)


@dataclass
class _RateLimitState:
    last_emit_at: Optional[float] = None
    suppressed_count: int = 0


class LogRateLimiter:
    """按 key（键）对重复日志做时间窗限流。"""

    def __init__(self, *, clock: Optional[Callable[[], float]] = None) -> None:
        self._clock = clock or time.monotonic
        self._states: Dict[Hashable, _RateLimitState] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        """清空当前限流状态。"""

        with self._lock:
            self._states.clear()

    def log(
        self,
        logger: logging.Logger,
        level: int,
        message: str,
        *args: object,
        key: Hashable,
        interval_sec: float,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> bool:
        """在时间窗内只放行首条日志，并在下次放行时补 suppressed（压制）统计。"""

        allowed, suppressed_count = self._consume(key=key, interval_sec=interval_sec)
        if not allowed:
            return False

        merged_extra: Dict[str, Any] = dict(extra or {})
        if suppressed_count > 0:
            merged_extra.setdefault("suppressed_count", suppressed_count)
            merged_extra.setdefault("rate_limit_window_sec", interval_sec)

        if merged_extra:
            kwargs["extra"] = merged_extra
        logger.log(level, message, *args, **kwargs)
        return True

    def _consume(self, *, key: Hashable, interval_sec: float) -> tuple[bool, int]:
        now = self._clock()
        interval = max(0.0, float(interval_sec))
        with self._lock:
            state = self._states.setdefault(key, _RateLimitState())
            if state.last_emit_at is None or interval <= 0.0 or (now - state.last_emit_at) >= interval:
                suppressed_count = state.suppressed_count
                state.last_emit_at = now
                state.suppressed_count = 0
                return True, suppressed_count
            state.suppressed_count += 1
            return False, 0
