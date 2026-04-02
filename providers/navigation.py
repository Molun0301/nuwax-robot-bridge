from __future__ import annotations

from typing import Protocol, runtime_checkable

from contracts.navigation import NavigationGoal, NavigationState
from providers.base import BaseProvider


@runtime_checkable
class NavigationProvider(BaseProvider, Protocol):
    """导航后端统一接口。"""

    def set_goal(self, goal: NavigationGoal) -> bool:
        """提交一个非阻塞导航目标。"""

    def cancel_goal(self) -> bool:
        """取消当前导航目标。"""

    def get_navigation_state(self) -> NavigationState:
        """读取当前导航状态。"""

    def is_goal_reached(self) -> bool:
        """判断当前目标是否已到达。"""
