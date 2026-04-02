from __future__ import annotations

from dataclasses import dataclass
import itertools

from contracts.image import CameraInfo, ImageFrame
from contracts.perception import BoundingBox2D, Detection2D, Detection3D, Track, TrackState
from services.perception.base import TrackerBackend, TrackerBackendSpec
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class TrackLifecyclePolicy:
    """二维轨迹生命周期策略。"""

    iou_match_threshold: float = 0.3
    tentative_hits_required: int = 2
    lost_ttl_frames: int = 2


@dataclass
class _TrackRecord:
    """内存中的活动轨迹。"""

    track_id: str
    camera_id: str
    label: str
    bbox: Optional[BoundingBox2D]
    score: float
    hits: int = 1
    misses: int = 0
    state: TrackState = TrackState.TENTATIVE
    attributes: Optional[Dict[str, object]] = None

    def to_contract(self) -> Track:
        return Track(
            track_id=self.track_id,
            label=self.label,
            state=self.state,
            score=self.score,
            bbox=self.bbox,
            attributes=dict(self.attributes or {}),
        )


def _bbox_iou(left: Optional[BoundingBox2D], right: Optional[BoundingBox2D]) -> float:
    if left is None or right is None:
        return 0.0
    left_x1 = left.x_px
    left_y1 = left.y_px
    left_x2 = left.x_px + left.width_px
    left_y2 = left.y_px + left.height_px
    right_x1 = right.x_px
    right_y1 = right.y_px
    right_x2 = right.x_px + right.width_px
    right_y2 = right.y_px + right.height_px

    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    if intersection <= 0.0:
        return 0.0
    left_area = left.width_px * left.height_px
    right_area = right.width_px * right.height_px
    union = max(1.0, left_area + right_area - intersection)
    return intersection / union


class Basic2DTrackerBackend(TrackerBackend):
    """基于标签和 IoU 的首版二维跟踪器。"""

    def __init__(
        self,
        *,
        name: str = "basic_2d_tracker",
        lifecycle_policy: Optional[TrackLifecyclePolicy] = None,
    ) -> None:
        self.spec = TrackerBackendSpec(
            name=name,
            backend_kind="2d_iou_tracker",
            metadata={"future_extension": "预留 3D 轨迹关联接口"},
        )
        self._policy = lifecycle_policy or TrackLifecyclePolicy()
        self._records_by_camera: Dict[str, Dict[str, _TrackRecord]] = {}
        self._track_counter = itertools.count(1)

    def update(
        self,
        image_frame: ImageFrame,
        *,
        detections_2d: Tuple[Detection2D, ...] = (),
        detections_3d: Tuple[Detection3D, ...] = (),
        camera_info: Optional[CameraInfo] = None,
    ) -> Tuple[Track, ...]:
        del detections_3d, camera_info
        camera_id = image_frame.camera_id
        records = self._records_by_camera.setdefault(camera_id, {})
        matched_track_ids: Set[str] = set()
        emitted: List[Track] = []

        for detection in sorted(detections_2d, key=lambda item: item.score, reverse=True):
            matched = self._match_record(records, detection, matched_track_ids)
            if matched is None:
                matched = self._create_record(camera_id, detection)
                records[matched.track_id] = matched
            else:
                matched.hits += 1
                matched.misses = 0
                matched.bbox = detection.bbox
                matched.score = detection.score
                matched.attributes = {
                    **dict(matched.attributes or {}),
                    **dict(detection.attributes),
                }

            matched.state = (
                TrackState.TRACKED
                if matched.hits >= self._policy.tentative_hits_required
                else TrackState.TENTATIVE
            )
            matched_track_ids.add(matched.track_id)
            emitted.append(matched.to_contract())

        removed_track_ids: List[str] = []
        for track_id, record in records.items():
            if track_id in matched_track_ids:
                continue
            record.misses += 1
            if record.misses > self._policy.lost_ttl_frames:
                record.state = TrackState.REMOVED
                emitted.append(record.to_contract())
                removed_track_ids.append(track_id)
                continue
            record.state = TrackState.LOST
            emitted.append(record.to_contract())

        for track_id in removed_track_ids:
            records.pop(track_id, None)

        return tuple(emitted)

    def reset(self, camera_id: Optional[str] = None) -> None:
        if camera_id is None:
            self._records_by_camera.clear()
            return
        self._records_by_camera.pop(camera_id, None)

    def _match_record(
        self,
        records: Dict[str, _TrackRecord],
        detection: Detection2D,
        matched_track_ids: Set[str],
    ) -> Optional[_TrackRecord]:
        best_record: Optional[_TrackRecord] = None
        best_iou = 0.0
        for record in records.values():
            if record.track_id in matched_track_ids:
                continue
            if record.label != detection.label:
                continue
            if record.state == TrackState.REMOVED:
                continue
            iou = _bbox_iou(record.bbox, detection.bbox)
            if iou < self._policy.iou_match_threshold:
                continue
            if iou > best_iou:
                best_iou = iou
                best_record = record
        return best_record

    def _create_record(self, camera_id: str, detection: Detection2D) -> _TrackRecord:
        suffix = next(self._track_counter)
        safe_camera = self._normalize_fragment(camera_id)
        safe_label = self._normalize_fragment(detection.label)
        track_id = f"trk_{safe_camera}_{safe_label}_{suffix:04d}"
        return _TrackRecord(
            track_id=track_id,
            camera_id=camera_id,
            label=detection.label,
            bbox=detection.bbox,
            score=detection.score,
            attributes=dict(detection.attributes),
        )

    def _normalize_fragment(self, value: str) -> str:
        result = []
        for char in value.lower():
            if char.isalnum():
                result.append(char)
            else:
                result.append("_")
        normalized = "".join(result).strip("_")
        return normalized or "item"
