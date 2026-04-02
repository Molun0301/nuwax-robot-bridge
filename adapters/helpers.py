from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Any, Mapping

from contracts.base import utc_now
from contracts.geometry import Pose, Quaternion, Transform, Vector3
from contracts.image import CameraInfo, ImageEncoding


def resolve_timestamp(payload: Optional[Mapping[str, Any]] = None, raw_value: Optional[Any] = None) -> datetime:
    """解析适配器输入里的时间戳。"""

    value = raw_value
    if value is None and payload is not None:
        value = payload.get("timestamp")

    if value is None:
        return utc_now()
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    raise ValueError(f"无法解析时间戳: {value!r}")


def build_vector3(payload: Optional[Mapping[str, Any]] = None) -> Vector3:
    """从字典构造三维向量。"""

    payload = payload or {}
    return Vector3(
        x=float(payload.get("x", 0.0)),
        y=float(payload.get("y", 0.0)),
        z=float(payload.get("z", 0.0)),
    )


def build_quaternion(payload: Optional[Mapping[str, Any]] = None) -> Quaternion:
    """从字典构造四元数。"""

    payload = payload or {}
    return Quaternion(
        x=float(payload.get("x", 0.0)),
        y=float(payload.get("y", 0.0)),
        z=float(payload.get("z", 0.0)),
        w=float(payload.get("w", 1.0)),
    )


def build_pose(
    payload: Optional[Mapping[str, Any]],
    *,
    frame_id: str,
    timestamp: Optional[datetime] = None,
) -> Pose:
    """从字典构造位姿。"""

    if payload is None:
        payload = {}
    return Pose(
        timestamp=timestamp or resolve_timestamp(payload),
        frame_id=frame_id,
        position=build_vector3(payload.get("position")),
        orientation=build_quaternion(payload.get("orientation")),
    )


def build_transform(
    payload: Optional[Mapping[str, Any]],
    *,
    parent_frame_id: str,
    child_frame_id: str,
    timestamp: Optional[datetime] = None,
) -> Transform:
    """从字典构造坐标变换。"""

    if payload is None:
        payload = {}
    return Transform(
        timestamp=timestamp or resolve_timestamp(payload),
        parent_frame_id=parent_frame_id,
        child_frame_id=child_frame_id,
        translation=build_vector3(payload.get("translation") or payload.get("position")),
        rotation=build_quaternion(payload.get("rotation") or payload.get("orientation")),
        authority=payload.get("authority"),
    )


def normalize_image_encoding(raw_encoding: Any, *, mime_type: Optional[str] = None) -> ImageEncoding:
    """把外部图像编码转换成统一枚举。"""

    if isinstance(raw_encoding, ImageEncoding):
        return raw_encoding

    if isinstance(raw_encoding, str):
        normalized = raw_encoding.strip().lower()
        aliases = {
            "jpg": ImageEncoding.JPEG,
            "jpeg": ImageEncoding.JPEG,
            "image/jpeg": ImageEncoding.JPEG,
            "png": ImageEncoding.PNG,
            "image/png": ImageEncoding.PNG,
            "rgb8": ImageEncoding.RGB8,
            "bgr8": ImageEncoding.BGR8,
            "mono8": ImageEncoding.MONO8,
            "depth16": ImageEncoding.DEPTH16,
        }
        if normalized in aliases:
            return aliases[normalized]

    if mime_type:
        return normalize_image_encoding(mime_type)

    raise ValueError(f"无法识别图像编码: {raw_encoding!r}")


def normalize_bytes(payload: Any) -> Optional[bytes]:
    """把外部字节载荷转换成 bytes。"""

    if payload is None:
        return None
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if isinstance(payload, list):
        return bytes(payload)
    raise ValueError(f"无法把类型 {type(payload)!r} 转成图像字节。")


def build_camera_info(
    payload: Optional[Mapping[str, Any]],
    *,
    camera_id: str,
    frame_id: str,
    timestamp: Optional[datetime] = None,
) -> CameraInfo:
    """从字典构造相机信息。"""

    payload = payload or {}
    return CameraInfo(
        timestamp=timestamp or resolve_timestamp(payload),
        camera_id=camera_id,
        frame_id=frame_id,
        width_px=int(payload.get("width_px", payload.get("width", 1))),
        height_px=int(payload.get("height_px", payload.get("height", 1))),
        fx=float(payload.get("fx", 0.0)),
        fy=float(payload.get("fy", 0.0)),
        cx=float(payload.get("cx", 0.0)),
        cy=float(payload.get("cy", 0.0)),
        distortion_model=payload.get("distortion_model"),
        distortion_coefficients=[float(value) for value in payload.get("distortion_coefficients", [])],
    )
