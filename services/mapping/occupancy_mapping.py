from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from contracts.geometry import Pose


Cell2D = Tuple[int, int]

_UNKNOWN_OCCUPANCY = -1
_FREE_OCCUPANCY = 0
_LETHAL_OCCUPANCY = 100


@dataclass
class PointCloudScanObservation:
    """平台侧点云扫描样本。"""

    stamp_sec: float
    sensor_x: float
    sensor_y: float
    source_label: str
    points_xyz_world: np.ndarray


@dataclass
class OccupancyMapBuildResult:
    """平台侧占据地图导出结果。"""

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


def smooth_occupied_grid(
    occupancy_grid: np.ndarray,
    *,
    obstacle_threshold: int = _LETHAL_OCCUPANCY,
    min_neighbor_fraction: float = 0.4,
) -> np.ndarray:
    """平滑占用区，避免孤立占用在后续膨胀时被过度放大。"""

    grid = _ensure_int_grid(occupancy_grid)
    occupied_mask = grid >= int(obstacle_threshold)
    if not np.any(occupied_mask):
        return grid.copy()

    height, width = grid.shape
    padded = np.pad(occupied_mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    neighbor_count = np.zeros((height, width), dtype=np.uint8)
    for row_offset in range(3):
        for col_offset in range(3):
            if row_offset == 1 and col_offset == 1:
                continue
            neighbor_count += padded[row_offset : row_offset + height, col_offset : col_offset + width]

    minimum_neighbors = int(math.ceil(8.0 * max(0.0, min(1.0, float(min_neighbor_fraction)))))
    unsupported_mask = occupied_mask & (neighbor_count < minimum_neighbors)
    result = grid.copy()
    result[unsupported_mask] = _FREE_OCCUPANCY
    return result


def overlay_occupied_grid(
    base_grid: np.ndarray,
    overlay_grid: np.ndarray,
    *,
    obstacle_threshold: int = _LETHAL_OCCUPANCY,
) -> np.ndarray:
    """把原始占用重新叠加回处理后的地图。"""

    base = _ensure_int_grid(base_grid)
    overlay = _ensure_int_grid(overlay_grid)
    if base.shape != overlay.shape:
        raise ValueError(f"地图尺寸不一致: base={base.shape} overlay={overlay.shape}")
    result = base.copy()
    result[overlay >= int(obstacle_threshold)] = _LETHAL_OCCUPANCY
    return result


def inflate_occupied_grid(
    occupancy_grid: np.ndarray,
    *,
    resolution_m: float,
    radius_m: float,
    obstacle_threshold: int = _LETHAL_OCCUPANCY,
) -> np.ndarray:
    """按机器人半宽膨胀障碍物。"""

    grid = _ensure_int_grid(occupancy_grid)
    occupied_mask = grid >= int(obstacle_threshold)
    if radius_m <= 1e-6 or not np.any(occupied_mask):
        return grid.copy()

    radius_cells = max(1, int(math.ceil(float(radius_m) / max(float(resolution_m), 1e-6))))
    kernel = _disk_kernel(radius_cells)
    dilated_mask = cv2.dilate(occupied_mask.astype(np.uint8), kernel, iterations=1) > 0

    result = grid.copy()
    result[dilated_mask] = _LETHAL_OCCUPANCY
    return result


def gradient_cost_grid(
    occupancy_grid: np.ndarray,
    *,
    resolution_m: float,
    obstacle_threshold: int = _LETHAL_OCCUPANCY,
    max_distance_m: float = 2.0,
) -> np.ndarray:
    """生成普通距离梯度代价图。"""

    grid = _ensure_int_grid(occupancy_grid)
    unknown_mask = grid < 0
    obstacle_mask = grid >= int(obstacle_threshold)
    if not np.any(obstacle_mask):
        result = np.zeros(grid.shape, dtype=np.float32)
        result[unknown_mask] = 100.0
        return result

    distance_px = cv2.distanceTransform((~obstacle_mask).astype(np.uint8), cv2.DIST_L2, 5)
    distance_m = np.clip(distance_px * float(resolution_m), 0.0, max(0.01, float(max_distance_m)))

    result = (1.0 - distance_m / max(0.01, float(max_distance_m))) * 100.0
    result[obstacle_mask] = 100.0
    result[distance_m >= float(max_distance_m)] = 0.0
    result[unknown_mask] = 100.0
    return result.astype(np.float32)


def voronoi_gradient_cost_grid(
    occupancy_grid: np.ndarray,
    *,
    resolution_m: float,
    obstacle_threshold: int = _LETHAL_OCCUPANCY,
    max_distance_m: float = 2.0,
) -> np.ndarray:
    """生成 Voronoi 梯度代价图，鼓励机器人走廊居中。"""

    grid = _ensure_int_grid(occupancy_grid)
    unknown_mask = grid < 0
    obstacle_mask = grid >= int(obstacle_threshold)
    if not np.any(obstacle_mask):
        result = np.zeros(grid.shape, dtype=np.float32)
        result[unknown_mask] = 100.0
        return result

    obstacle_components = cv2.connectedComponents(obstacle_mask.astype(np.uint8), connectivity=8)[1]
    obstacle_cluster_count = int(np.max(obstacle_components))
    if obstacle_cluster_count <= 1:
        return gradient_cost_grid(
            grid,
            resolution_m=resolution_m,
            obstacle_threshold=obstacle_threshold,
            max_distance_m=max_distance_m,
        )

    distance_px, nearest_labels = cv2.distanceTransformWithLabels(
        (~obstacle_mask).astype(np.uint8),
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_CCOMP,
    )
    nearest_labels = nearest_labels.astype(np.int32)
    nearest_labels[obstacle_mask] = obstacle_components[obstacle_mask]

    local_max = _neighbor_max(nearest_labels)
    local_min = _neighbor_min_positive(nearest_labels)
    voronoi_edges = (local_max != local_min) & (~obstacle_mask)
    if not np.any(voronoi_edges):
        return gradient_cost_grid(
            grid,
            resolution_m=resolution_m,
            obstacle_threshold=obstacle_threshold,
            max_distance_m=max_distance_m,
        )

    voronoi_distance_px = cv2.distanceTransform((~voronoi_edges).astype(np.uint8), cv2.DIST_L2, 5)
    total_distance_px = distance_px + voronoi_distance_px

    result = np.zeros(grid.shape, dtype=np.float32)
    valid_mask = total_distance_px > 1e-6
    result[valid_mask] = (voronoi_distance_px[valid_mask] / total_distance_px[valid_mask]) * 99.0
    result[obstacle_mask] = 100.0
    result[distance_px * float(resolution_m) >= float(max_distance_m)] = 0.0
    result[unknown_mask] = 100.0
    return result.astype(np.float32)


def build_navigation_cost_grid(
    occupancy_grid: np.ndarray,
    *,
    resolution_m: float,
    inflation_radius_m: float,
    strategy: str = "mixed",
    smooth_min_neighbor_fraction: float = 0.4,
    gradient_max_distance_m: float = 1.5,
    obstacle_threshold: int = _LETHAL_OCCUPANCY,
) -> np.ndarray:
    """对照 dimos 组合导航代价图工具链。"""

    grid = _ensure_int_grid(occupancy_grid)
    normalized_strategy = str(strategy or "mixed").strip().lower()
    if normalized_strategy == "simple":
        navigation_grid = inflate_occupied_grid(
            grid,
            resolution_m=resolution_m,
            radius_m=inflation_radius_m,
            obstacle_threshold=obstacle_threshold,
        )
    elif normalized_strategy == "mixed":
        navigation_grid = smooth_occupied_grid(
            grid,
            obstacle_threshold=obstacle_threshold,
            min_neighbor_fraction=smooth_min_neighbor_fraction,
        )
        navigation_grid = inflate_occupied_grid(
            navigation_grid,
            resolution_m=resolution_m,
            radius_m=inflation_radius_m,
            obstacle_threshold=obstacle_threshold,
        )
        navigation_grid = overlay_occupied_grid(
            navigation_grid,
            grid,
            obstacle_threshold=obstacle_threshold,
        )
    else:
        raise ValueError(f"未知导航地图策略: {strategy}")

    return voronoi_gradient_cost_grid(
        navigation_grid,
        resolution_m=resolution_m,
        obstacle_threshold=obstacle_threshold,
        max_distance_m=gradient_max_distance_m,
    )


class SparseOccupancyMapBuilder:
    """平台侧稀疏射线建图器。"""

    def __init__(self, config) -> None:
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

    def export_state(self) -> Dict[str, object]:
        """导出当前稀疏建图状态，用于运行时恢复。"""

        entries = [
            {
                "row": int(row),
                "col": int(col),
                "log_odds": round(float(log_odds), 8),
            }
            for (row, col), log_odds in sorted(self._log_odds_by_cell.items())
        ]
        return {
            "cells": entries,
            "source_labels": sorted(self._source_labels),
            "latest_sensor_cell": (
                [int(self._latest_sensor_cell[0]), int(self._latest_sensor_cell[1])]
                if self._latest_sensor_cell is not None
                else None
            ),
        }

    def restore_state(self, payload: Optional[Dict[str, object]]) -> bool:
        """从持久化载荷恢复稀疏建图状态。"""

        self.reset()
        if not isinstance(payload, dict):
            return False

        raw_cells = payload.get("cells")
        if not isinstance(raw_cells, list):
            return False

        restored_cells: Dict[Cell2D, float] = {}
        for item in raw_cells:
            if not isinstance(item, dict):
                continue
            try:
                row = int(item.get("row"))
                col = int(item.get("col"))
                log_odds = float(item.get("log_odds"))
            except (TypeError, ValueError):
                continue
            restored_cells[(row, col)] = log_odds

        if not restored_cells:
            return False

        self._log_odds_by_cell = restored_cells
        raw_labels = payload.get("source_labels")
        if isinstance(raw_labels, list):
            self._source_labels = {
                str(item).strip()
                for item in raw_labels
                if str(item).strip()
            }

        raw_sensor_cell = payload.get("latest_sensor_cell")
        if isinstance(raw_sensor_cell, (list, tuple)) and len(raw_sensor_cell) >= 2:
            try:
                self._latest_sensor_cell = (int(raw_sensor_cell[0]), int(raw_sensor_cell[1]))
            except (TypeError, ValueError):
                self._latest_sensor_cell = None
        return True

    def reset(self) -> None:
        """清空当前稀疏建图状态。"""

        self._log_odds_by_cell = {}
        self._source_labels = set()
        self._latest_sensor_cell = None

    def build(self) -> Optional[OccupancyMapBuildResult]:
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

        occupancy_data = np.full((height, width), _UNKNOWN_OCCUPANCY, dtype=np.int32)
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
                occupancy_data[grid_row, grid_col] = _LETHAL_OCCUPANCY
                occupied_mask[grid_row, grid_col] = True
            else:
                occupancy_data[grid_row, grid_col] = _FREE_OCCUPANCY

        if known_cell_count == 0:
            return None

        cost_data = build_navigation_cost_grid(
            occupancy_data,
            resolution_m=float(self.config.global_map_resolution_m),
            inflation_radius_m=float(self.config.global_map_inflation_radius_m),
            strategy=str(getattr(self.config, "navigation_map_strategy", "mixed")),
            smooth_min_neighbor_fraction=float(
                getattr(self.config, "navigation_smooth_min_neighbor_fraction", 0.4)
            ),
            gradient_max_distance_m=float(
                getattr(self.config, "navigation_gradient_max_distance_m", 1.5)
            ),
        )
        return OccupancyMapBuildResult(
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


def build_sliding_window_map(
    *,
    scans: Sequence[PointCloudScanObservation],
    current_pose: Pose,
    config,
) -> Optional[OccupancyMapBuildResult]:
    """根据最近点云样本构建局部导航地图。"""

    if not scans:
        return None

    width = int(getattr(config, "local_map_width"))
    height = int(getattr(config, "local_map_height"))
    resolution = float(getattr(config, "local_map_resolution_m"))
    origin_x = float(current_pose.position.x) - (width * resolution) / 2.0
    origin_y = float(current_pose.position.y) - (height * resolution) / 2.0

    observed = np.zeros((height, width), dtype=np.uint8)
    occupied = np.zeros((height, width), dtype=np.uint8)
    source_labels = {scan.source_label for scan in scans if str(scan.source_label or "").strip()}

    for scan in scans:
        sensor_cell = _world_to_grid_cell(
            x=scan.sensor_x,
            y=scan.sensor_y,
            origin_x=origin_x,
            origin_y=origin_y,
            resolution=resolution,
            width=width,
            height=height,
        )
        if sensor_cell is None:
            continue
        sensor_row, sensor_col = sensor_cell
        for point_x, point_y, _point_z in scan.points_xyz_world:
            point_cell = _world_to_grid_cell(
                x=float(point_x),
                y=float(point_y),
                origin_x=origin_x,
                origin_y=origin_y,
                resolution=resolution,
                width=width,
                height=height,
            )
            if point_cell is None:
                continue
            point_row, point_col = point_cell
            for free_row, free_col in _bresenham_cells((sensor_row, sensor_col), (point_row, point_col))[:-1]:
                observed[free_row, free_col] = 1
            observed[point_row, point_col] = 1
            occupied[point_row, point_col] = 1

    if not np.any(observed):
        return None

    occupancy_data = np.full((height, width), _UNKNOWN_OCCUPANCY, dtype=np.int32)
    occupancy_data[observed == 1] = _FREE_OCCUPANCY
    occupancy_data[occupied == 1] = _LETHAL_OCCUPANCY

    cost_data = build_navigation_cost_grid(
        occupancy_data,
        resolution_m=resolution,
        inflation_radius_m=float(getattr(config, "local_map_inflation_radius_m", 0.25)),
        strategy=str(getattr(config, "navigation_map_strategy", "mixed")),
        smooth_min_neighbor_fraction=float(
            getattr(config, "navigation_smooth_min_neighbor_fraction", 0.4)
        ),
        gradient_max_distance_m=float(
            getattr(config, "navigation_gradient_max_distance_m", 1.5)
        ),
    )

    if not source_labels:
        source_label = "local_point_cloud"
    elif len(source_labels) == 1:
        source_label = next(iter(source_labels))
    else:
        source_label = "mixed_point_cloud"

    return OccupancyMapBuildResult(
        width=width,
        height=height,
        resolution_m=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        occupancy_data=occupancy_data,
        cost_data=cost_data,
        occupied_mask=occupied.astype(bool),
        source_label=source_label,
        known_cell_count=int(np.sum(observed)),
    )


def _ensure_int_grid(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.int32)
    if array.ndim != 2:
        raise ValueError(f"占据地图必须是二维数组，当前 shape={array.shape}")
    return array


def _disk_kernel(radius_cells: int) -> np.ndarray:
    radius = max(1, int(radius_cells))
    kernel = np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    center = radius
    for row in range(kernel.shape[0]):
        for col in range(kernel.shape[1]):
            if math.hypot(float(row - center), float(col - center)) <= float(radius) + 1e-6:
                kernel[row, col] = 1
    return kernel


def _neighbor_max(labels: np.ndarray) -> np.ndarray:
    height, width = labels.shape
    padded = np.pad(labels, 1, mode="constant", constant_values=0)
    stacks = [
        padded[row_offset : row_offset + height, col_offset : col_offset + width]
        for row_offset in range(3)
        for col_offset in range(3)
    ]
    return np.maximum.reduce(stacks)


def _neighbor_min_positive(labels: np.ndarray) -> np.ndarray:
    height, width = labels.shape
    padded = np.pad(labels, 1, mode="constant", constant_values=0)
    sentinel = int(np.max(labels)) + 1
    stacks = []
    for row_offset in range(3):
        for col_offset in range(3):
            window = padded[row_offset : row_offset + height, col_offset : col_offset + width]
            stacks.append(np.where(window > 0, window, sentinel))
    result = np.minimum.reduce(stacks)
    result[result == sentinel] = 0
    return result


def _world_to_grid_cell(
    *,
    x: float,
    y: float,
    origin_x: float,
    origin_y: float,
    resolution: float,
    width: int,
    height: int,
) -> Optional[Cell2D]:
    col = int((x - origin_x) / resolution)
    row = int((y - origin_y) / resolution)
    if col < 0 or row < 0 or col >= width or row >= height:
        return None
    return row, col


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


__all__ = [
    "OccupancyMapBuildResult",
    "PointCloudScanObservation",
    "SparseOccupancyMapBuilder",
    "build_navigation_cost_grid",
    "build_sliding_window_map",
    "gradient_cost_grid",
    "inflate_occupied_grid",
    "overlay_occupied_grid",
    "smooth_occupied_grid",
    "voronoi_gradient_cost_grid",
]
