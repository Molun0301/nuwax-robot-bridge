from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from drivers.robots.go2.settings import Go2MapSynthesisConfig


Cell2D = Tuple[int, int]


@dataclass
class Go2GlobalMapBuildResult:
    """Go2 端侧全局栅格地图导出结果。"""

    width: int
    height: int
    resolution_m: float
    origin_x: float
    origin_y: float
    occupancy_data: np.ndarray
    cost_data: np.ndarray
    occupied_mask: np.ndarray
    source_label: str
    known_cell_count: int


class Go2SparseGlobalMapBuilder:
    """基于点云射线更新的 Go2 稀疏全局地图构建器。"""

    def __init__(self, config: "Go2MapSynthesisConfig") -> None:
        self.config = config
        self._log_odds_by_cell: Dict[Cell2D, float] = {}
        self._source_labels: Set[str] = set()
        self._latest_sensor_cell: Optional[Cell2D] = None

    def ingest_scan(
        self,
        *,
        world_points: np.ndarray,
        sensor_x: float,
        sensor_y: float,
        source_label: str,
    ) -> bool:
        """写入一帧已转到地图坐标系下的点云观测。"""

        if not bool(getattr(self.config, "global_map_enabled", False)):
            return False
        if world_points.size == 0:
            return False

        resolution = float(self.config.global_map_resolution_m)
        sensor_cell = self._world_to_sparse_cell(sensor_x, sensor_y, resolution=resolution)
        free_cells: Set[Cell2D] = set()
        occupied_cells: Set[Cell2D] = set()

        for point_x, point_y, _point_z in world_points:
            point_cell = self._world_to_sparse_cell(float(point_x), float(point_y), resolution=resolution)
            if point_cell == sensor_cell:
                continue
            ray_cells = _bresenham_cells(sensor_cell, point_cell)
            if len(ray_cells) <= 1:
                occupied_cells.add(point_cell)
                continue
            free_cells.update(ray_cells[:-1])
            occupied_cells.add(point_cell)

        if not free_cells and not occupied_cells:
            return False

        self._latest_sensor_cell = sensor_cell
        if str(source_label or "").strip():
            self._source_labels.add(str(source_label).strip())

        for cell in free_cells - occupied_cells:
            self._update_log_odds(cell, delta=-float(self.config.global_map_free_log_odds_delta))
        for cell in occupied_cells:
            self._update_log_odds(cell, delta=float(self.config.global_map_hit_log_odds_delta))
        return True

    def has_data(self) -> bool:
        """返回当前是否已有全局地图观测。"""

        return bool(self._log_odds_by_cell)

    def build(self) -> Optional[Go2GlobalMapBuildResult]:
        """导出当前全局地图快照。"""

        if not self._log_odds_by_cell:
            return None

        min_row = min(row for row, _col in self._log_odds_by_cell.keys())
        max_row = max(row for row, _col in self._log_odds_by_cell.keys())
        min_col = min(col for _row, col in self._log_odds_by_cell.keys())
        max_col = max(col for _row, col in self._log_odds_by_cell.keys())

        padding_cells = max(0, int(getattr(self.config, "global_map_padding_cells", 0)))
        min_row -= padding_cells
        max_row += padding_cells
        min_col -= padding_cells
        max_col += padding_cells

        max_width = max(20, int(self.config.global_map_max_width))
        max_height = max(20, int(self.config.global_map_max_height))
        width = max_col - min_col + 1
        height = max_row - min_row + 1

        if width > max_width or height > max_height:
            center_row, center_col = self._latest_sensor_cell or (
                (min_row + max_row) // 2,
                (min_col + max_col) // 2,
            )
            half_height = max_height // 2
            half_width = max_width // 2
            min_row = center_row - half_height
            max_row = min_row + max_height - 1
            min_col = center_col - half_width
            max_col = min_col + max_width - 1
            width = max_width
            height = max_height

        occupancy_data = np.full((height, width), -1, dtype=np.int32)
        cost_data = np.full((height, width), 100.0, dtype=np.float32)
        occupied_mask = np.zeros((height, width), dtype=bool)

        occupied_threshold = float(self.config.global_map_occupied_log_odds_threshold)
        known_cell_count = 0
        for (row, col), log_odds in self._log_odds_by_cell.items():
            grid_row = row - min_row
            grid_col = col - min_col
            if grid_row < 0 or grid_row >= height or grid_col < 0 or grid_col >= width:
                continue
            known_cell_count += 1
            if log_odds >= occupied_threshold:
                occupancy_data[grid_row, grid_col] = 100
                cost_data[grid_row, grid_col] = 100.0
                occupied_mask[grid_row, grid_col] = True
            else:
                occupancy_data[grid_row, grid_col] = 0
                cost_data[grid_row, grid_col] = 0.0

        if known_cell_count == 0:
            return None

        self._inflate_cost_map(
            cost_data=cost_data,
            occupied_mask=occupied_mask,
            resolution=float(self.config.global_map_resolution_m),
            radius_m=float(self.config.global_map_inflation_radius_m),
        )
        return Go2GlobalMapBuildResult(
            width=width,
            height=height,
            resolution_m=float(self.config.global_map_resolution_m),
            origin_x=float(min_col) * float(self.config.global_map_resolution_m),
            origin_y=float(min_row) * float(self.config.global_map_resolution_m),
            occupancy_data=occupancy_data,
            cost_data=cost_data,
            occupied_mask=occupied_mask,
            source_label=self._resolve_source_label(),
            known_cell_count=known_cell_count,
        )

    def _resolve_source_label(self) -> str:
        if not self._source_labels:
            return "global_point_cloud"
        if len(self._source_labels) == 1:
            return next(iter(self._source_labels))
        return "mixed_global_point_cloud"

    def _update_log_odds(self, cell: Cell2D, *, delta: float) -> None:
        current_value = float(self._log_odds_by_cell.get(cell, 0.0))
        updated_value = current_value + float(delta)
        updated_value = max(
            float(self.config.global_map_log_odds_min),
            min(float(self.config.global_map_log_odds_max), updated_value),
        )
        self._log_odds_by_cell[cell] = updated_value

    def _world_to_sparse_cell(self, x: float, y: float, *, resolution: float) -> Cell2D:
        col = int(math.floor(float(x) / max(resolution, 1e-6)))
        row = int(math.floor(float(y) / max(resolution, 1e-6)))
        return row, col

    def _inflate_cost_map(
        self,
        *,
        cost_data: np.ndarray,
        occupied_mask: np.ndarray,
        resolution: float,
        radius_m: float,
    ) -> None:
        if radius_m <= 1e-6 or not np.any(occupied_mask):
            return
        radius_cells = max(1, int(math.ceil(radius_m / max(resolution, 1e-6))))
        offsets: List[Tuple[int, int]] = []
        for row_offset in range(-radius_cells, radius_cells + 1):
            for col_offset in range(-radius_cells, radius_cells + 1):
                if math.hypot(float(row_offset), float(col_offset)) <= float(radius_cells) + 1e-6:
                    offsets.append((row_offset, col_offset))

        height, width = cost_data.shape
        occupied_rows, occupied_cols = np.where(occupied_mask)
        for row, col in zip(occupied_rows.tolist(), occupied_cols.tolist()):
            for row_offset, col_offset in offsets:
                next_row = row + row_offset
                next_col = col + col_offset
                if 0 <= next_row < height and 0 <= next_col < width:
                    cost_data[next_row, next_col] = max(cost_data[next_row, next_col], 100.0)


def _bresenham_cells(start_cell: Cell2D, end_cell: Cell2D) -> List[Cell2D]:
    start_row, start_col = start_cell
    end_row, end_col = end_cell
    row_delta = abs(end_row - start_row)
    col_delta = abs(end_col - start_col)
    step_row = 1 if start_row < end_row else -1
    step_col = 1 if start_col < end_col else -1
    error = row_delta - col_delta
    row = start_row
    col = start_col
    result: List[Cell2D] = []
    while True:
        result.append((row, col))
        if row == end_row and col == end_col:
            break
        double_error = 2 * error
        if double_error > -col_delta:
            error -= col_delta
            row += step_row
        if double_error < row_delta:
            error += row_delta
            col += step_col
    return result
