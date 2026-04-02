from __future__ import annotations

from math import atan2, cos, radians, sin
from typing import Iterable, List, Optional, Sequence, Tuple

from contracts.geometry import Pose, Quaternion, Vector3
from contracts.maps import CostMap, OccupancyGrid


class InspectionPosePlanner:
    """为语义区域或对象实例生成可达、可观察的巡检位姿。"""

    def __init__(
        self,
        *,
        occupied_threshold: int = 70,
        cost_threshold: float = 60.0,
        candidate_radii_m: Sequence[float] = (0.8, 1.2, 1.6),
        candidate_angles_deg: Sequence[float] = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0),
    ) -> None:
        self._occupied_threshold = max(0, min(100, int(occupied_threshold)))
        self._cost_threshold = float(cost_threshold)
        self._candidate_radii_m = tuple(float(item) for item in candidate_radii_m if float(item) > 0.0)
        self._candidate_angles_deg = tuple(float(item) for item in candidate_angles_deg)

    def plan_for_target(
        self,
        *,
        target_pose: Pose,
        occupancy_grid: Optional[OccupancyGrid] = None,
        cost_map: Optional[CostMap] = None,
        existing_poses: Optional[Iterable[Pose]] = None,
        max_candidates: int = 4,
    ) -> List[Pose]:
        """根据目标中心位姿生成一组巡检位姿。"""

        if existing_poses:
            deduplicated = self._deduplicate(existing_poses, max_candidates=max_candidates)
            if deduplicated:
                return deduplicated

        candidates: List[Tuple[float, Pose]] = []
        for radius_m in self._candidate_radii_m:
            for angle_deg in self._candidate_angles_deg:
                pose = self._build_candidate_pose(target_pose, radius_m=radius_m, angle_deg=angle_deg)
                traversable, penalty = self._evaluate_pose(
                    pose,
                    occupancy_grid=occupancy_grid,
                    cost_map=cost_map,
                )
                if not traversable:
                    continue
                score = (radius_m * 0.05) + penalty
                candidates.append((score, pose))

        candidates.sort(key=lambda item: item[0])
        resolved = [pose for _, pose in candidates[: max(1, max_candidates)]]
        if resolved:
            return self._deduplicate(resolved, max_candidates=max_candidates)
        return [target_pose]

    def _build_candidate_pose(self, target_pose: Pose, *, radius_m: float, angle_deg: float) -> Pose:
        angle_rad = radians(angle_deg)
        candidate_x = float(target_pose.position.x) + (radius_m * cos(angle_rad))
        candidate_y = float(target_pose.position.y) + (radius_m * sin(angle_rad))
        yaw = atan2(
            float(target_pose.position.y) - candidate_y,
            float(target_pose.position.x) - candidate_x,
        )
        return Pose(
            frame_id=target_pose.frame_id,
            position=Vector3(x=candidate_x, y=candidate_y, z=float(target_pose.position.z)),
            orientation=self._yaw_to_quaternion(yaw),
        )

    def _evaluate_pose(
        self,
        pose: Pose,
        *,
        occupancy_grid: Optional[OccupancyGrid],
        cost_map: Optional[CostMap],
    ) -> Tuple[bool, float]:
        occupancy_penalty = 0.0
        if occupancy_grid is not None:
            occupancy_value = self._sample_grid_value(occupancy_grid, pose)
            if occupancy_value is None or occupancy_value < 0 or occupancy_value >= self._occupied_threshold:
                return False, 1.0
            occupancy_penalty = occupancy_value / 100.0

        cost_penalty = 0.0
        if cost_map is not None:
            cost_value = self._sample_grid_value(cost_map, pose)
            if cost_value is None or cost_value >= self._cost_threshold:
                return False, 1.0
            cost_penalty = float(cost_value) / max(1.0, self._cost_threshold)

        return True, occupancy_penalty + cost_penalty

    def _sample_grid_value(self, grid, pose: Pose):
        resolution = float(grid.resolution_m)
        origin_x = float(grid.origin.position.x)
        origin_y = float(grid.origin.position.y)
        offset_x = float(pose.position.x) - origin_x
        offset_y = float(pose.position.y) - origin_y
        cell_x = int(offset_x / resolution)
        cell_y = int(offset_y / resolution)
        if cell_x < 0 or cell_y < 0 or cell_x >= int(grid.width) or cell_y >= int(grid.height):
            return None
        index = (cell_y * int(grid.width)) + cell_x
        if index < 0 or index >= len(grid.data):
            return None
        return grid.data[index]

    def _deduplicate(self, poses: Iterable[Pose], *, max_candidates: int) -> List[Pose]:
        result: List[Pose] = []
        for pose in poses:
            if any(self._distance(existing, pose) <= 0.15 for existing in result):
                continue
            result.append(pose)
            if len(result) >= max(1, max_candidates):
                break
        return result

    def _distance(self, left: Pose, right: Pose) -> float:
        delta_x = float(left.position.x) - float(right.position.x)
        delta_y = float(left.position.y) - float(right.position.y)
        delta_z = float(left.position.z) - float(right.position.z)
        return float((delta_x * delta_x + delta_y * delta_y + delta_z * delta_z) ** 0.5)

    def _yaw_to_quaternion(self, yaw_rad: float) -> Quaternion:
        half_yaw = yaw_rad * 0.5
        return Quaternion(z=sin(half_yaw), w=cos(half_yaw))
