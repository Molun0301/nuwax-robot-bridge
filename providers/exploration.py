from __future__ import annotations

from typing import Protocol, runtime_checkable

from contracts.navigation import ExplorationState, ExploreAreaRequest
from providers.base import BaseProvider


@runtime_checkable
class ExplorationProvider(BaseProvider, Protocol):
    """探索后端统一接口。"""

    def start_exploration(self, request: ExploreAreaRequest) -> bool:
        """启动一次探索任务。"""

    def stop_exploration(self) -> bool:
        """停止当前探索任务。"""

    def get_exploration_state(self) -> ExplorationState:
        """读取当前探索状态。"""
