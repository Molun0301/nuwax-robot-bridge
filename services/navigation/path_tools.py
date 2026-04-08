from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from contracts.maps import OccupancyGrid


Point2D = Tuple[float, float]


@dataclass
class PathSafetyResult:
    """路径掩码检测结果。"""

    is_safe: bool
    occupied_ratio: float
    occupied_cell_count: int
    total_cell_count: int
    mask: np.ndarray


def resample_waypoints(
    waypoints: Sequence[Point2D],
    *,
    spacing_m: float,
    smoothing_window: int = 0,
) -> List[Point2D]:
    """把折线路径重采样为更均匀的路径点。"""

    if len(waypoints) <= 2 or spacing_m <= 1e-6:
        return list(waypoints)

    points = _remove_duplicate_waypoints(waypoints)
    if len(points) <= 2:
        return points

    distances = [0.0]
    for index in range(1, len(points)):
        distances.append(distances[-1] + _point_distance(points[index - 1], points[index]))
    total_length = distances[-1]
    if total_length <= spacing_m:
        return points

    sample_count = max(2, int(math.ceil(total_length / spacing_m)) + 1)
    sample_distances = np.linspace(0.0, total_length, sample_count)
    x_values = np.asarray([point[0] for point in points], dtype=np.float32)
    y_values = np.asarray([point[1] for point in points], dtype=np.float32)
    arc_lengths = np.asarray(distances, dtype=np.float32)

    sampled_x = np.interp(sample_distances, arc_lengths, x_values)
    sampled_y = np.interp(sample_distances, arc_lengths, y_values)

    if smoothing_window >= 3 and len(sampled_x) >= smoothing_window:
        kernel = np.ones(int(smoothing_window), dtype=np.float32)
        kernel /= float(np.sum(kernel))
        sampled_x = np.convolve(sampled_x, kernel, mode="same")
        sampled_y = np.convolve(sampled_y, kernel, mode="same")
        sampled_x[0], sampled_y[0] = points[0]
        sampled_x[-1], sampled_y[-1] = points[-1]

    result = [(float(x), float(y)) for x, y in zip(sampled_x.tolist(), sampled_y.tolist())]
    return _remove_duplicate_waypoints(result)


def build_path_mask(
    occupancy_grid: OccupancyGrid,
    *,
    waypoints: Sequence[Point2D],
    robot_width_m: float,
    max_length_m: Optional[float] = None,
) -> np.ndarray:
    """把路径投影成机器人会实际经过的栅格掩码。"""

    mask = np.zeros((int(occupancy_grid.height), int(occupancy_grid.width)), dtype=np.uint8)
    if len(waypoints) < 2:
        return mask.astype(bool)

    line_width_pixels = max(1, int(math.ceil(float(robot_width_m) / max(float(occupancy_grid.resolution_m), 1e-6))))
    cumulative_length_m = 0.0
    length_limit_m = float("inf") if max_length_m is None else max(0.0, float(max_length_m))

    for index in range(len(waypoints) - 1):
        start_point = waypoints[index]
        end_point = waypoints[index + 1]
        segment_length_m = _point_distance(start_point, end_point)
        if segment_length_m <= 1e-6:
            continue
        if cumulative_length_m >= length_limit_m:
            break

        effective_end = end_point
        if cumulative_length_m + segment_length_m > length_limit_m:
            ratio = (length_limit_m - cumulative_length_m) / segment_length_m
            effective_end = (
                start_point[0] + (end_point[0] - start_point[0]) * ratio,
                start_point[1] + (end_point[1] - start_point[1]) * ratio,
            )

        start_cell = _world_to_grid_cell(occupancy_grid, start_point)
        end_cell = _world_to_grid_cell(occupancy_grid, effective_end)
        if start_cell is None or end_cell is None:
            cumulative_length_m += segment_length_m
            continue
        cv2.line(mask, start_cell, end_cell, color=255, thickness=line_width_pixels)
        cumulative_length_m += segment_length_m

    return mask.astype(bool)


def evaluate_path_safety(
    occupancy_grid: OccupancyGrid,
    *,
    waypoints: Sequence[Point2D],
    robot_width_m: float,
    obstacle_threshold: int = 65,
    max_occupied_ratio: float = 0.05,
    max_length_m: Optional[float] = None,
) -> PathSafetyResult:
    """检测路径掩码是否明显穿过障碍物。"""

    mask = build_path_mask(
        occupancy_grid,
        waypoints=waypoints,
        robot_width_m=robot_width_m,
        max_length_m=max_length_m,
    )
    total_cell_count = int(np.sum(mask))
    if total_cell_count <= 0:
        return PathSafetyResult(
            is_safe=True,
            occupied_ratio=0.0,
            occupied_cell_count=0,
            total_cell_count=0,
            mask=mask,
        )

    occupancy = np.asarray(occupancy_grid.data, dtype=np.int32).reshape(int(occupancy_grid.height), int(occupancy_grid.width))
    occupied_mask = occupancy >= int(obstacle_threshold)
    occupied_in_path = mask & occupied_mask
    occupied_cell_count = int(np.sum(occupied_in_path))
    occupied_ratio = float(occupied_cell_count) / float(total_cell_count)
    return PathSafetyResult(
        is_safe=occupied_ratio <= float(max_occupied_ratio),
        occupied_ratio=occupied_ratio,
        occupied_cell_count=occupied_cell_count,
        total_cell_count=total_cell_count,
        mask=mask & (~occupied_mask),
    )


def _remove_duplicate_waypoints(waypoints: Sequence[Point2D]) -> List[Point2D]:
    result: List[Point2D] = []
    for waypoint in waypoints:
        if result and _point_distance(result[-1], waypoint) <= 1e-6:
            continue
        result.append((float(waypoint[0]), float(waypoint[1])))
    return result


def _point_distance(start: Point2D, end: Point2D) -> float:
    return math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))


def _world_to_grid_cell(occupancy_grid: OccupancyGrid, point: Point2D) -> Optional[Tuple[int, int]]:
    resolution = float(occupancy_grid.resolution_m)
    origin_x = float(occupancy_grid.origin.position.x)
    origin_y = float(occupancy_grid.origin.position.y)
    col = int((float(point[0]) - origin_x) / resolution)
    row = int((float(point[1]) - origin_y) / resolution)
    if col < 0 or row < 0 or col >= int(occupancy_grid.width) or row >= int(occupancy_grid.height):
        return None
    return col, row


__all__ = ["PathSafetyResult", "build_path_mask", "evaluate_path_safety", "resample_waypoints"]
