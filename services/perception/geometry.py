from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from contracts.frame_semantics import frame_ids_semantically_equal
from contracts.geometry import FrameTree, Pose, Vector3
from contracts.image import CameraInfo, ImageFrame
from contracts.perception import Detection2D
from contracts.pointcloud import PointCloudFrame
from services.localization.frame_tree_ops import transform_points_to_frame


class DetectionGeometryAugmentor:
    """基于点云为二维检测结果补充几何样本。"""

    _MAP_POINT_KEYS = ("map_points", "point_samples_map", "map_point", "point_sample_map")
    _CAMERA_POINT_KEYS = ("camera_points", "point_samples_camera", "camera_point", "point_sample_camera")
    _PIXEL_CENTER_KEYS = ("pixel_center_px", "mask_center_px", "center_px")

    def __init__(
        self,
        *,
        max_points_per_detection: int = 24,
        min_depth_m: float = 0.05,
    ) -> None:
        self._max_points_per_detection = max(1, int(max_points_per_detection))
        self._min_depth_m = max(0.001, float(min_depth_m))

    def augment_detections(
        self,
        detections_2d: Tuple[Detection2D, ...],
        *,
        image_frame: ImageFrame,
        camera_info: CameraInfo,
        point_cloud: Optional[PointCloudFrame],
        frame_tree: Optional[FrameTree] = None,
        current_pose: Optional[Pose] = None,
    ) -> Tuple[Tuple[Detection2D, ...], Dict[str, object]]:
        if not detections_2d or point_cloud is None or not point_cloud.points:
            return detections_2d, {}

        prepared = self._prepare_projectable_points(
            image_frame=image_frame,
            camera_info=camera_info,
            point_cloud=point_cloud,
            frame_tree=frame_tree,
            map_frame_id=str(current_pose.frame_id or "").strip() or None if current_pose is not None else None,
        )
        if prepared is None:
            return detections_2d, {}

        projected_pixels, camera_points, map_points, map_frame_id = prepared
        augmented = []
        augmented_count = 0
        support_point_total = 0
        for detection in detections_2d:
            updated_detection, support_count = self._augment_single_detection(
                detection=detection,
                camera_info=camera_info,
                projected_pixels=projected_pixels,
                camera_points=camera_points,
                map_points=map_points,
                map_frame_id=map_frame_id,
            )
            if updated_detection is not detection:
                augmented_count += 1
            support_point_total += support_count
            augmented.append(updated_detection)

        return tuple(augmented), {
            "geometry_augmented_count": augmented_count,
            "geometry_support_point_total": support_point_total,
            "geometry_point_cloud_frame_id": point_cloud.frame_id,
            "geometry_point_cloud_point_count": int(point_cloud.point_count),
            "geometry_point_cloud_source_topic": point_cloud.metadata.get("source_topic"),
        }

    def _prepare_projectable_points(
        self,
        *,
        image_frame: ImageFrame,
        camera_info: CameraInfo,
        point_cloud: PointCloudFrame,
        frame_tree: Optional[FrameTree],
        map_frame_id: Optional[str],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[str]]]:
        point_samples = tuple(point_cloud.points)
        if not point_samples:
            return None

        camera_points = self._resolve_points_to_frame(
            points=point_samples,
            source_frame_id=point_cloud.frame_id,
            target_frame_id=camera_info.frame_id,
            frame_tree=frame_tree,
        )
        if camera_points is None:
            return None

        camera_array = np.asarray([[float(item.x), float(item.y), float(item.z)] for item in camera_points], dtype=np.float64)
        if camera_array.size == 0:
            return None

        valid_depth_mask = camera_array[:, 2] > float(self._min_depth_m)
        if not np.any(valid_depth_mask):
            return None
        camera_array = camera_array[valid_depth_mask]

        pixel_u = (camera_array[:, 0] * float(camera_info.fx) / camera_array[:, 2]) + float(camera_info.cx)
        pixel_v = (camera_array[:, 1] * float(camera_info.fy) / camera_array[:, 2]) + float(camera_info.cy)
        projected_pixels = np.stack((pixel_u, pixel_v), axis=1)

        width_px = int(image_frame.width_px)
        height_px = int(image_frame.height_px)
        in_view_mask = (
            (projected_pixels[:, 0] >= 0.0)
            & (projected_pixels[:, 0] < float(width_px))
            & (projected_pixels[:, 1] >= 0.0)
            & (projected_pixels[:, 1] < float(height_px))
        )
        if not np.any(in_view_mask):
            return None

        projected_pixels = projected_pixels[in_view_mask]
        camera_array = camera_array[in_view_mask]

        map_array = None
        if map_frame_id:
            resolved_map_points = self._resolve_points_to_frame(
                points=point_samples,
                source_frame_id=point_cloud.frame_id,
                target_frame_id=map_frame_id,
                frame_tree=frame_tree,
            )
            if resolved_map_points is not None:
                map_array = np.asarray(
                    [[float(item.x), float(item.y), float(item.z)] for item in resolved_map_points],
                    dtype=np.float64,
                )[valid_depth_mask][in_view_mask]
        return projected_pixels, camera_array, map_array, map_frame_id

    def _augment_single_detection(
        self,
        *,
        detection: Detection2D,
        camera_info: CameraInfo,
        projected_pixels: np.ndarray,
        camera_points: np.ndarray,
        map_points: Optional[np.ndarray],
        map_frame_id: Optional[str],
    ) -> Tuple[Detection2D, int]:
        bbox = detection.bbox
        x_min = float(bbox.x_px)
        y_min = float(bbox.y_px)
        x_max = x_min + float(bbox.width_px)
        y_max = y_min + float(bbox.height_px)

        support_mask = (
            (projected_pixels[:, 0] >= x_min)
            & (projected_pixels[:, 0] <= x_max)
            & (projected_pixels[:, 1] >= y_min)
            & (projected_pixels[:, 1] <= y_max)
        )
        support_indices = np.flatnonzero(support_mask)
        if support_indices.size == 0:
            return detection, 0

        ordered_indices = support_indices[np.argsort(camera_points[support_indices, 2])]
        selected_indices = ordered_indices[: self._max_points_per_detection]
        selected_camera = camera_points[selected_indices]
        selected_map = map_points[selected_indices] if map_points is not None else None
        selected_pixels = projected_pixels[selected_indices]

        attributes = dict(detection.attributes)
        attributes_updated = False
        if not self._has_any_geometry_key(attributes, self._CAMERA_POINT_KEYS):
            attributes["camera_points"] = {
                "frame_id": camera_info.frame_id,
                "points": self._serialize_points(selected_camera),
            }
            attributes_updated = True
        if map_frame_id and selected_map is not None and not self._has_any_geometry_key(attributes, self._MAP_POINT_KEYS):
            attributes["map_points"] = {
                "frame_id": map_frame_id,
                "points": self._serialize_points(selected_map),
            }
            attributes_updated = True
        if self._float_or_none(attributes.get("depth_m")) is None:
            attributes["depth_m"] = round(float(np.median(selected_camera[:, 2])), 6)
            attributes_updated = True
        if not self._has_any_geometry_key(attributes, self._PIXEL_CENTER_KEYS):
            pixel_center = selected_pixels.mean(axis=0)
            attributes["pixel_center_px"] = [round(float(pixel_center[0]), 4), round(float(pixel_center[1]), 4)]
            attributes_updated = True

        attributes["geometry_support_point_count"] = int(selected_indices.size)
        attributes["geometry_projection_method"] = "pointcloud_bbox_projection"
        if not attributes_updated:
            return detection.model_copy(update={"attributes": attributes}, deep=True), int(selected_indices.size)
        return detection.model_copy(update={"attributes": attributes}, deep=True), int(selected_indices.size)

    def _resolve_points_to_frame(
        self,
        *,
        points: Iterable[Vector3],
        source_frame_id: str,
        target_frame_id: str,
        frame_tree: Optional[FrameTree],
    ) -> Optional[Tuple[Vector3, ...]]:
        if frame_ids_semantically_equal(source_frame_id, target_frame_id):
            if source_frame_id == target_frame_id:
                return tuple(points)
            return tuple(
                Vector3(
                    x=float(item.x),
                    y=float(item.y),
                    z=float(item.z),
                )
                for item in points
            )
        return transform_points_to_frame(
            frame_tree,
            points,
            source_frame_id=source_frame_id,
            target_frame_id=target_frame_id,
        )

    def _serialize_points(self, points: np.ndarray) -> Tuple[Dict[str, float], ...]:
        return tuple(
            {
                "x": round(float(item[0]), 6),
                "y": round(float(item[1]), 6),
                "z": round(float(item[2]), 6),
            }
            for item in points
        )

    def _has_any_geometry_key(self, attributes: Dict[str, object], keys: Tuple[str, ...]) -> bool:
        return any(attributes.get(key) is not None for key in keys)

    def _float_or_none(self, value: object) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
