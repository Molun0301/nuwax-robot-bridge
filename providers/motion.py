from __future__ import annotations

from typing import Protocol, runtime_checkable

from contracts.geometry import Twist
from providers.base import BaseProvider


@runtime_checkable
class MotionControl(BaseProvider, Protocol):
    """运动控制接口。"""

    def send_twist(self, twist: Twist) -> None:
        """发送速度控制指令。"""

    def stop_motion(self) -> None:
        """停止当前运动。"""

