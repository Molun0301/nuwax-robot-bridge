from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from contracts.frame_semantics import frame_ids_semantically_equal
from contracts.geometry import FrameTree, Pose, Quaternion, Transform, Vector3


@dataclass(frozen=True)
class ResolvedFrameTransform:
    """从源坐标系到目标坐标系的解析结果。"""

    source_frame_id: str
    target_frame_id: str
    matrix: np.ndarray


def transform_pose_to_frame(
    frame_tree: Optional[FrameTree],
    pose: Pose,
    *,
    target_frame_id: str,
) -> Optional[Pose]:
    """把位姿从源坐标系变换到目标坐标系。"""

    resolved = resolve_frame_transform(
        frame_tree,
        source_frame_id=pose.frame_id,
        target_frame_id=target_frame_id,
    )
    if resolved is None:
        return None
    position_vector = np.array(
        [
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
            1.0,
        ],
        dtype=np.float64,
    )
    rotated_position = resolved.matrix @ position_vector
    rotation_matrix = _pose_rotation_matrix(pose.orientation)
    composed_rotation = resolved.matrix[:3, :3] @ rotation_matrix
    return Pose(
        frame_id=resolved.target_frame_id,
        position=Vector3(
            x=float(rotated_position[0]),
            y=float(rotated_position[1]),
            z=float(rotated_position[2]),
        ),
        orientation=_matrix_to_quaternion(composed_rotation),
    )


def transform_points_to_frame(
    frame_tree: Optional[FrameTree],
    points: Iterable[Vector3],
    *,
    source_frame_id: str,
    target_frame_id: str,
) -> Optional[Tuple[Vector3, ...]]:
    """把一组点从源坐标系变换到目标坐标系。"""

    resolved = resolve_frame_transform(
        frame_tree,
        source_frame_id=source_frame_id,
        target_frame_id=target_frame_id,
    )
    if resolved is None:
        return None
    transformed: List[Vector3] = []
    for point in points:
        position_vector = np.array(
            [float(point.x), float(point.y), float(point.z), 1.0],
            dtype=np.float64,
        )
        mapped = resolved.matrix @ position_vector
        transformed.append(
            Vector3(
                x=float(mapped[0]),
                y=float(mapped[1]),
                z=float(mapped[2]),
            )
        )
    return tuple(transformed)


def resolve_frame_transform(
    frame_tree: Optional[FrameTree],
    *,
    source_frame_id: str,
    target_frame_id: str,
) -> Optional[ResolvedFrameTransform]:
    """解析从源坐标系到目标坐标系的刚体变换。"""

    normalized_source = str(source_frame_id or "").strip()
    normalized_target = str(target_frame_id or "").strip()
    if not normalized_source or not normalized_target:
        return None
    if frame_ids_semantically_equal(normalized_source, normalized_target):
        return ResolvedFrameTransform(
            source_frame_id=normalized_source,
            target_frame_id=normalized_target,
            matrix=np.identity(4, dtype=np.float64),
        )
    if frame_tree is None:
        return None

    available_frames = _collect_available_frames(frame_tree.transforms)
    resolved_source = _resolve_frame_alias(normalized_source, available_frames)
    resolved_target = _resolve_frame_alias(normalized_target, available_frames)
    if resolved_source is None or resolved_target is None:
        return None
    if resolved_source == resolved_target:
        return ResolvedFrameTransform(
            source_frame_id=normalized_source,
            target_frame_id=normalized_target,
            matrix=np.identity(4, dtype=np.float64),
        )

    adjacency: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for transform in frame_tree.transforms:
        parent_matrix = _transform_to_matrix(transform)
        adjacency.setdefault(transform.child_frame_id, []).append((transform.parent_frame_id, parent_matrix))
        adjacency.setdefault(transform.parent_frame_id, []).append((transform.child_frame_id, np.linalg.inv(parent_matrix)))

    visited = {resolved_source}
    queue: List[Tuple[str, np.ndarray]] = [(resolved_source, np.identity(4, dtype=np.float64))]
    while queue:
        current_frame_id, current_matrix = queue.pop(0)
        if current_frame_id == resolved_target:
            return ResolvedFrameTransform(
                source_frame_id=normalized_source,
                target_frame_id=normalized_target,
                matrix=current_matrix,
            )
        for next_frame_id, next_matrix in adjacency.get(current_frame_id, []):
            if next_frame_id in visited:
                continue
            visited.add(next_frame_id)
            queue.append((next_frame_id, next_matrix @ current_matrix))
    return None


def _collect_available_frames(transforms: Iterable[Transform]) -> Tuple[str, ...]:
    frames = []
    for transform in transforms:
        for frame_id in (transform.parent_frame_id, transform.child_frame_id):
            if frame_id not in frames:
                frames.append(frame_id)
    return tuple(frames)


def _resolve_frame_alias(frame_id: str, available_frames: Iterable[str]) -> Optional[str]:
    for candidate in available_frames:
        if candidate == frame_id:
            return candidate
    for candidate in available_frames:
        if frame_ids_semantically_equal(candidate, frame_id):
            return candidate
    return None


def _transform_to_matrix(transform: Transform) -> np.ndarray:
    matrix = np.identity(4, dtype=np.float64)
    matrix[:3, :3] = _pose_rotation_matrix(transform.rotation)
    matrix[:3, 3] = np.array(
        [
            float(transform.translation.x),
            float(transform.translation.y),
            float(transform.translation.z),
        ],
        dtype=np.float64,
    )
    return matrix


def _pose_rotation_matrix(rotation: Quaternion) -> np.ndarray:
    x = float(rotation.x)
    y = float(rotation.y)
    z = float(rotation.z)
    w = float(rotation.w)
    norm = sqrt((x * x) + (y * y) + (z * z) + (w * w))
    if norm <= 1e-12:
        return np.identity(3, dtype=np.float64)
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return np.array(
        [
            [1.0 - (2.0 * ((y * y) + (z * z))), 2.0 * ((x * y) - (z * w)), 2.0 * ((x * z) + (y * w))],
            [2.0 * ((x * y) + (z * w)), 1.0 - (2.0 * ((x * x) + (z * z))), 2.0 * ((y * z) - (x * w))],
            [2.0 * ((x * z) - (y * w)), 2.0 * ((y * z) + (x * w)), 1.0 - (2.0 * ((x * x) + (y * y)))],
        ],
        dtype=np.float64,
    )


def _matrix_to_quaternion(rotation_matrix: np.ndarray) -> Quaternion:
    matrix = np.asarray(rotation_matrix, dtype=np.float64)
    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    if trace > 0.0:
        scale = sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
        return Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))
    if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
        return Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))
    if matrix[1, 1] > matrix[2, 2]:
        scale = sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
        return Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))
    scale = sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
    w = (matrix[1, 0] - matrix[0, 1]) / scale
    x = (matrix[0, 2] + matrix[2, 0]) / scale
    y = (matrix[1, 2] + matrix[2, 1]) / scale
    z = 0.25 * scale
    return Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))


__all__ = [
    "ResolvedFrameTransform",
    "resolve_frame_transform",
    "transform_points_to_frame",
    "transform_pose_to_frame",
]
