from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from contracts.maps import CostMap, OccupancyGrid, SemanticMap
from providers.base import BaseProvider


@runtime_checkable
class MapProvider(BaseProvider, Protocol):
    """地图读取接口。"""

    def get_occupancy_grid(self) -> Optional[OccupancyGrid]:
        """返回占据栅格地图。"""

    def get_cost_map(self) -> Optional[CostMap]:
        """返回代价地图。"""

    def get_semantic_map(self) -> Optional[SemanticMap]:
        """返回语义地图。"""

