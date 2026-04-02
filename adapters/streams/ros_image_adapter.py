from __future__ import annotations

from typing import Dict, Optional, Any

from adapters.base import AdapterBase, AdapterCategory, AdapterConfig
from adapters.helpers import build_camera_info, normalize_bytes, normalize_image_encoding, resolve_timestamp
from contracts.image import CameraInfo, ImageFrame


class RosImageAdapter(AdapterBase[Dict[str, Any], ImageFrame]):
    """ROS 图像流适配器模板。"""

    @classmethod
    def build_default_config(
        cls,
        name: str = "ros_image_stream",
        *,
        source_ref: Optional[str] = "/camera/image_raw",
    ) -> AdapterConfig:
        """构造默认配置。"""

        return AdapterConfig(
            name=name,
            category=AdapterCategory.STREAMS,
            source_kind="ros2",
            contract_type="ImageFrame",
            source_ref=source_ref,
            settings={"default_camera_id": "front_camera", "default_frame_id": "camera_front"},
        )

    def convert_payload(self, payload: Dict[str, Any]) -> ImageFrame:
        """把外部 ROS 图像消息转换成图像契约。"""

        timestamp = resolve_timestamp(payload)
        camera_id = str(payload.get("camera_id") or self.config.settings.get("default_camera_id") or "front_camera")
        frame_id = str(payload.get("frame_id") or self.config.settings.get("default_frame_id") or "camera_front")
        image_data = normalize_bytes(payload.get("data"))
        return ImageFrame(
            timestamp=timestamp,
            camera_id=camera_id,
            frame_id=frame_id,
            width_px=int(payload.get("width_px", payload.get("width", 1))),
            height_px=int(payload.get("height_px", payload.get("height", 1))),
            encoding=normalize_image_encoding(payload.get("encoding")),
            data=image_data,
            uri=payload.get("uri"),
            artifact_id=payload.get("artifact_id"),
            metadata={
                "topic": payload.get("topic", self.config.source_ref),
                "sequence_id": payload.get("sequence_id"),
                **dict(payload.get("metadata", {})),
            },
        )

    def build_camera_info_contract(self, payload: Dict[str, Any]) -> CameraInfo:
        """把外部相机内参转换成统一契约。"""

        timestamp = resolve_timestamp(payload)
        camera_id = str(payload.get("camera_id") or self.config.settings.get("default_camera_id") or "front_camera")
        frame_id = str(payload.get("frame_id") or self.config.settings.get("default_frame_id") or "camera_front")
        return build_camera_info(payload.get("camera_info") or payload, camera_id=camera_id, frame_id=frame_id, timestamp=timestamp)
