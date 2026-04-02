from __future__ import annotations

from typing import Dict, Optional, Any

from adapters.base import AdapterBase, AdapterCategory, AdapterConfig
from adapters.helpers import build_pose, resolve_timestamp
from contracts.naming import build_observation_id, validate_observation_id
from contracts.perception import BoundingBox2D, Detection2D, Detection3D, Observation, Track, TrackState


class ExternalPerceptionServiceAdapter(AdapterBase[Dict[str, Any], Observation]):
    """外部感知服务适配器模板。"""

    @classmethod
    def build_default_config(
        cls,
        name: str = "external_perception_service",
        *,
        source_ref: Optional[str] = "perception-service",
    ) -> AdapterConfig:
        """构造默认配置。"""

        return AdapterConfig(
            name=name,
            category=AdapterCategory.PERCEPTION,
            source_kind="service",
            contract_type="Observation",
            source_ref=source_ref,
            settings={"default_frame_id": "base", "default_camera_id": "front_camera"},
        )

    def convert_payload(self, payload: Dict[str, Any]) -> Observation:
        """把外部感知结果转换成观察契约。"""

        timestamp = resolve_timestamp(payload)
        frame_id = str(payload.get("frame_id") or self.config.settings.get("default_frame_id") or "base")
        metadata = dict(payload.get("metadata", {}))
        raw_observation_id = str(payload.get("observation_id", "")).strip()
        try:
            observation_id = validate_observation_id(raw_observation_id) if raw_observation_id else ""
        except ValueError:
            observation_id = ""
        if not observation_id:
            observation_id = build_observation_id(self._normalize_source_id(), timestamp=timestamp)
            if raw_observation_id:
                metadata["source_observation_id"] = raw_observation_id
        return Observation(
            timestamp=timestamp,
            observation_id=observation_id,
            frame_id=frame_id,
            summary=payload.get("summary"),
            detections_2d=[self._build_detection_2d(item, timestamp) for item in payload.get("detections_2d", [])],
            detections_3d=[self._build_detection_3d(item, frame_id, timestamp) for item in payload.get("detections_3d", [])],
            tracks=[self._build_track(item, frame_id, timestamp) for item in payload.get("tracks", [])],
            artifact_ids=[str(item) for item in payload.get("artifact_ids", [])],
            metadata=metadata,
        )

    def _build_detection_2d(self, payload: Dict[str, Any], timestamp) -> Detection2D:
        bbox_payload = payload.get("bbox") or {}
        return Detection2D(
            timestamp=timestamp,
            label=str(payload["label"]),
            score=float(payload.get("score", 0.0)),
            bbox=BoundingBox2D(
                timestamp=timestamp,
                x_px=float(bbox_payload.get("x_px", 0.0)),
                y_px=float(bbox_payload.get("y_px", 0.0)),
                width_px=float(bbox_payload.get("width_px", 1.0)),
                height_px=float(bbox_payload.get("height_px", 1.0)),
            ),
            camera_id=str(payload.get("camera_id") or self.config.settings.get("default_camera_id") or "front_camera"),
            track_id=payload.get("track_id"),
            attributes=dict(payload.get("attributes", {})),
        )

    def _build_detection_3d(self, payload: Dict[str, Any], frame_id: str, timestamp) -> Detection3D:
        return Detection3D(
            timestamp=timestamp,
            label=str(payload["label"]),
            score=float(payload.get("score", 0.0)),
            pose=build_pose(payload.get("pose") or {}, frame_id=frame_id, timestamp=timestamp),
            size_x_m=float(payload["size_x_m"]) if payload.get("size_x_m") is not None else None,
            size_y_m=float(payload["size_y_m"]) if payload.get("size_y_m") is not None else None,
            size_z_m=float(payload["size_z_m"]) if payload.get("size_z_m") is not None else None,
            attributes=dict(payload.get("attributes", {})),
        )

    def _build_track(self, payload: Dict[str, Any], frame_id: str, timestamp) -> Track:
        raw_state = str(payload.get("state", TrackState.TENTATIVE)).lower()
        state = TrackState(raw_state)
        pose_payload = payload.get("pose")
        bbox_payload = payload.get("bbox")
        return Track(
            timestamp=timestamp,
            track_id=str(payload["track_id"]),
            label=str(payload["label"]),
            state=state,
            score=float(payload.get("score", 0.0)),
            pose=build_pose(pose_payload, frame_id=frame_id, timestamp=timestamp) if pose_payload else None,
            bbox=(
                BoundingBox2D(
                    timestamp=timestamp,
                    x_px=float(bbox_payload.get("x_px", 0.0)),
                    y_px=float(bbox_payload.get("y_px", 0.0)),
                    width_px=float(bbox_payload.get("width_px", 1.0)),
                    height_px=float(bbox_payload.get("height_px", 1.0)),
                )
                if bbox_payload
                else None
            ),
            attributes=dict(payload.get("attributes", {})),
        )

    def _normalize_source_id(self) -> str:
        raw_source = str(self.config.source_ref or self.adapter_name).lower()
        normalized = []
        for char in raw_source:
            if char.isalnum():
                normalized.append(char)
            else:
                normalized.append("_")
        value = "".join(normalized).strip("_")
        return value or "observation"
