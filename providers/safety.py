from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from contracts.robot_state import SafetyState
from providers.base import BaseProvider


@runtime_checkable
class SafetyProvider(BaseProvider, Protocol):
    """安全状态与安全动作接口。"""

    def get_safety_state(self) -> SafetyState:
        """返回当前安全状态。"""

    def request_safe_stop(self, reason: Optional[str] = None) -> None:
        """请求执行安全停止。"""

