from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from contracts.geometry import Pose, Quaternion, Vector3
from contracts.maps import CostMap, OccupancyGrid

if TYPE_CHECKING:
    from drivers.robots.go2.navigation_backend import Go2GridNavigationPlanner
    from drivers.robots.go2.settings import Go2ExplorationConfig


Cell2D = Tuple[int, int]


@dataclass
class Go2FrontierCandidate:
    """Go2 前沿探索候选。"""

    pose: Pose
    score: float
    cluster_size: int
    info_gain_cells: int
    path_distance_m: float
    average_cost: float


class Go2FrontierExplorer:
    """Go2 显式前沿检测与收益排序器。"""

    _NEIGHBORS: Tuple[Tuple[int, int], ...] = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    def __init__(self, config: "Go2ExplorationConfig") -> None:
        self.config = config

    def select_candidates(
        self,
        *,
        current_pose: Pose,
        occupancy_grid: OccupancyGrid,
        cost_map: Optional[CostMap],
        planner: "Go2GridNavigationPlanner",
        center_pose: Pose,
        radius_m: float,
        attempted_poses: Sequence[Pose],
    ) -> List[Go2FrontierCandidate]:
        """按收益排序返回可达前沿候选。"""

        if not bool(self.config.frontier_enabled):
            return []
        if (
            occupancy_grid.frame_id
            and current_pose.frame_id
            and occupancy_grid.frame_id != current_pose.frame_id
        ):
            return []

        occupancy = np.asarray(occupancy_grid.data, dtype=np.int32).reshape(occupancy_grid.height, occupancy_grid.width)
        free_mask = occupancy == 0
        unknown_mask = occupancy < 0
        if not np.any(free_mask) or not np.any(unknown_mask):
            return []

        frontier_mask = self._build_frontier_mask(
            free_mask=free_mask,
            unknown_mask=unknown_mask,
            cost_map=cost_map,
        )
        if not np.any(frontier_mask):
            return []

        clusters = self._cluster_frontiers(frontier_mask)
        if not clusters:
            return []

        candidates: List[Go2FrontierCandidate] = []
        revisit_separation_m = float(self.config.frontier_revisit_separation_m)
        current_yaw = _yaw_from_pose(current_pose)
        max_radius_m = max(0.1, float(radius_m))
        for cluster in clusters:
            if len(cluster) < int(self.config.frontier_min_cluster_cells):
                continue
            cluster_points_world = [
                _cell_to_world(
                    row=row,
                    col=col,
                    origin_x=float(occupancy_grid.origin.position.x),
                    origin_y=float(occupancy_grid.origin.position.y),
                    resolution=float(occupancy_grid.resolution_m),
                )
                for row, col in cluster
            ]
            centroid_x = sum(point[0] for point in cluster_points_world) / float(len(cluster_points_world))
            centroid_y = sum(point[1] for point in cluster_points_world) / float(len(cluster_points_world))
            if math.hypot(centroid_x - center_pose.position.x, centroid_y - center_pose.position.y) > max_radius_m:
                continue

            best_cell = min(
                cluster,
                key=lambda cell: self._candidate_cell_score(
                    cell=cell,
                    centroid_x=centroid_x,
                    centroid_y=centroid_y,
                    occupancy_grid=occupancy_grid,
                    cost_map=cost_map,
                ),
            )
            goal_x, goal_y = _cell_to_world(
                row=best_cell[0],
                col=best_cell[1],
                origin_x=float(occupancy_grid.origin.position.x),
                origin_y=float(occupancy_grid.origin.position.y),
                resolution=float(occupancy_grid.resolution_m),
            )
            if any(
                math.hypot(goal_x - attempted.position.x, goal_y - attempted.position.y) < revisit_separation_m
                for attempted in attempted_poses
            ):
                continue

            yaw_rad = math.atan2(goal_y - current_pose.position.y, goal_x - current_pose.position.x)
            goal_pose = Pose(
                frame_id=current_pose.frame_id,
                position=Vector3(x=goal_x, y=goal_y, z=center_pose.position.z),
                orientation=_quaternion_from_yaw(yaw_rad),
            )
            plan = planner.plan_preview(
                current_pose=current_pose,
                target_pose=goal_pose,
                occupancy_grid=occupancy_grid,
                cost_map=cost_map,
            )
            if plan is None or plan.planning_mode != "goal":
                continue

            average_cost = self._cluster_average_cost(cluster=cluster, cost_map=cost_map)
            heading_error = abs(_normalize_angle(yaw_rad - current_yaw))
            heading_bonus = max(0.0, math.cos(heading_error))
            info_gain_cells = len(cluster)
            score = (
                float(self.config.frontier_info_gain_weight) * float(info_gain_cells)
                - float(self.config.frontier_distance_weight) * float(plan.total_distance_m)
                - float(self.config.frontier_cost_weight) * (average_cost / 100.0)
                + float(self.config.frontier_heading_weight) * heading_bonus
            )
            candidates.append(
                Go2FrontierCandidate(
                    pose=goal_pose,
                    score=score,
                    cluster_size=len(cluster),
                    info_gain_cells=info_gain_cells,
                    path_distance_m=float(plan.total_distance_m),
                    average_cost=average_cost,
                )
            )

        candidates.sort(
            key=lambda item: (
                -float(item.score),
                -int(item.info_gain_cells),
                float(item.path_distance_m),
            )
        )
        return candidates

    def count_known_cells(self, occupancy_grid: Optional[OccupancyGrid]) -> int:
        """统计当前地图已知栅格数量。"""

        if occupancy_grid is None:
            return 0
        return int(sum(1 for value in occupancy_grid.data if int(value) >= 0))

    def _build_frontier_mask(
        self,
        *,
        free_mask: np.ndarray,
        unknown_mask: np.ndarray,
        cost_map: Optional[CostMap],
    ) -> np.ndarray:
        height, width = free_mask.shape
        padded_unknown = np.pad(unknown_mask.astype(np.uint8), 1, constant_values=0)
        unknown_neighbors = np.zeros((height, width), dtype=np.uint8)
        for row_offset, col_offset in self._NEIGHBORS:
            unknown_neighbors += padded_unknown[
                1 + row_offset : 1 + row_offset + height,
                1 + col_offset : 1 + col_offset + width,
            ]

        frontier_mask = free_mask & (unknown_neighbors >= int(self.config.frontier_min_unknown_neighbors))
        if cost_map is not None and int(cost_map.width) == width and int(cost_map.height) == height:
            costs = np.asarray(cost_map.data, dtype=np.float32).reshape(height, width)
            frontier_mask &= costs <= float(self.config.max_goal_cost)
        return frontier_mask

    def _cluster_frontiers(self, frontier_mask: np.ndarray) -> List[List[Cell2D]]:
        height, width = frontier_mask.shape
        visited = np.zeros((height, width), dtype=np.uint8)
        clusters: List[List[Cell2D]] = []
        for row in range(height):
            for col in range(width):
                if not frontier_mask[row, col] or visited[row, col] == 1:
                    continue
                cluster: List[Cell2D] = []
                queue = [(row, col)]
                visited[row, col] = 1
                while queue:
                    current_row, current_col = queue.pop()
                    cluster.append((current_row, current_col))
                    for row_offset, col_offset in self._NEIGHBORS:
                        next_row = current_row + row_offset
                        next_col = current_col + col_offset
                        if next_row < 0 or next_row >= height or next_col < 0 or next_col >= width:
                            continue
                        if visited[next_row, next_col] == 1 or not frontier_mask[next_row, next_col]:
                            continue
                        visited[next_row, next_col] = 1
                        queue.append((next_row, next_col))
                clusters.append(cluster)
        return clusters

    def _candidate_cell_score(
        self,
        *,
        cell: Cell2D,
        centroid_x: float,
        centroid_y: float,
        occupancy_grid: OccupancyGrid,
        cost_map: Optional[CostMap],
    ) -> float:
        world_x, world_y = _cell_to_world(
            row=cell[0],
            col=cell[1],
            origin_x=float(occupancy_grid.origin.position.x),
            origin_y=float(occupancy_grid.origin.position.y),
            resolution=float(occupancy_grid.resolution_m),
        )
        centroid_distance = math.hypot(world_x - centroid_x, world_y - centroid_y)
        cell_cost = 0.0
        if cost_map is not None and int(cost_map.width) == int(occupancy_grid.width) and int(cost_map.height) == int(occupancy_grid.height):
            index = cell[0] * int(cost_map.width) + cell[1]
            if 0 <= index < len(cost_map.data):
                cell_cost = float(cost_map.data[index])
        return centroid_distance + (cell_cost / 100.0)

    def _cluster_average_cost(self, *, cluster: Sequence[Cell2D], cost_map: Optional[CostMap]) -> float:
        if cost_map is None:
            return 0.0
        costs: List[float] = []
        width = int(cost_map.width)
        for row, col in cluster:
            index = row * width + col
            if 0 <= index < len(cost_map.data):
                costs.append(float(cost_map.data[index]))
        if not costs:
            return 0.0
        return float(sum(costs) / float(len(costs)))


def _cell_to_world(*, row: int, col: int, origin_x: float, origin_y: float, resolution: float) -> Tuple[float, float]:
    return (
        origin_x + (float(col) + 0.5) * resolution,
        origin_y + (float(row) + 0.5) * resolution,
    )


def _yaw_from_pose(pose: Pose) -> float:
    rotation = pose.orientation
    siny_cosp = 2.0 * (rotation.w * rotation.z + rotation.x * rotation.y)
    cosy_cosp = 1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quaternion_from_yaw(yaw_rad: float) -> Quaternion:
    half_yaw = yaw_rad * 0.5
    return Quaternion(z=math.sin(half_yaw), w=math.cos(half_yaw))


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))
