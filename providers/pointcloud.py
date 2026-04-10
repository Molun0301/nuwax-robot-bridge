from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from contracts.pointcloud import PointCloudFrame
from providers.base import BaseProvider


@runtime_checkable
class PointCloudProvider(BaseProvider, Protocol):
    """点云读取接口。"""

    def get_latest_point_cloud(self) -> Optional[PointCloudFrame]:
        """返回最近可用的一帧点云。"""
