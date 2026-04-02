from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from contracts.geometry import FrameTree, Pose
from providers.base import BaseProvider


@runtime_checkable
class LocalizationProvider(BaseProvider, Protocol):
    """定位读取接口。"""

    def get_current_pose(self) -> Optional[Pose]:
        """返回当前机器人位姿。"""

    def get_frame_tree(self) -> Optional[FrameTree]:
        """返回当前 TF 坐标树快照。"""
