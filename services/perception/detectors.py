from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple

from contracts.image import CameraInfo, ImageFrame
from contracts.perception import BoundingBox2D, Detection2D, Detection3D
from gateways.errors import GatewayError
from services.perception.base import DetectionBundle, DetectorBackend, DetectorBackendSpec
from services.perception.image_utils import decode_image_frame_to_bgr


def _clamp_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(1.0, score))


def _normalize_label(value: object) -> str:
    label = str(value or "unknown").strip().lower()
    return label or "unknown"


def _clamp_bbox(
    *,
    x_px: object,
    y_px: object,
    width_px: object,
    height_px: object,
    image_frame: ImageFrame,
) -> BoundingBox2D:
    frame_width = max(1.0, float(image_frame.width_px))
    frame_height = max(1.0, float(image_frame.height_px))
    x = max(0.0, min(float(x_px), frame_width - 1.0))
    y = max(0.0, min(float(y_px), frame_height - 1.0))
    width = max(1.0, min(float(width_px), frame_width - x))
    height = max(1.0, min(float(height_px), frame_height - y))
    return BoundingBox2D(x_px=x, y_px=y, width_px=width, height_px=height)


def normalize_detection_2d(raw_detection: object, image_frame: ImageFrame) -> Detection2D:
    """把检测后端的输出统一规整到标准二维检测契约。"""

    if isinstance(raw_detection, Detection2D):
        bbox = _clamp_bbox(
            x_px=raw_detection.bbox.x_px,
            y_px=raw_detection.bbox.y_px,
            width_px=raw_detection.bbox.width_px,
            height_px=raw_detection.bbox.height_px,
            image_frame=image_frame,
        )
        return raw_detection.model_copy(
            update={
                "label": _normalize_label(raw_detection.label),
                "score": _clamp_score(raw_detection.score),
                "bbox": bbox,
                "camera_id": raw_detection.camera_id or image_frame.camera_id,
                "track_id": raw_detection.track_id,
                "attributes": dict(raw_detection.attributes),
            },
            deep=True,
        )

    if not isinstance(raw_detection, dict):
        raise GatewayError(f"检测结果类型非法：{type(raw_detection)!r}")

    bbox_payload = dict(raw_detection.get("bbox") or {})
    bbox = _clamp_bbox(
        x_px=bbox_payload.get("x_px", raw_detection.get("x_px", 0.0)),
        y_px=bbox_payload.get("y_px", raw_detection.get("y_px", 0.0)),
        width_px=bbox_payload.get("width_px", raw_detection.get("width_px", image_frame.width_px)),
        height_px=bbox_payload.get("height_px", raw_detection.get("height_px", image_frame.height_px)),
        image_frame=image_frame,
    )
    return Detection2D(
        label=_normalize_label(raw_detection.get("label")),
        score=_clamp_score(raw_detection.get("score", 0.0)),
        bbox=bbox,
        camera_id=str(raw_detection.get("camera_id") or image_frame.camera_id),
        track_id=str(raw_detection.get("track_id")).strip() or None if raw_detection.get("track_id") else None,
        attributes=dict(raw_detection.get("attributes") or {}),
    )


def normalize_detection_3d(raw_detection: object) -> Detection3D:
    """把检测后端的输出统一规整到标准三维检测契约。"""

    if isinstance(raw_detection, Detection3D):
        return raw_detection.model_copy(
            update={
                "label": _normalize_label(raw_detection.label),
                "score": _clamp_score(raw_detection.score),
                "attributes": dict(raw_detection.attributes),
            },
            deep=True,
        )
    if not isinstance(raw_detection, dict):
        raise GatewayError(f"三维检测结果类型非法：{type(raw_detection)!r}")
    return Detection3D.model_validate(
        {
            **raw_detection,
            "label": _normalize_label(raw_detection.get("label")),
            "score": _clamp_score(raw_detection.get("score", 0.0)),
            "attributes": dict(raw_detection.get("attributes") or {}),
        }
    )


class MetadataDrivenDetectorBackend(DetectorBackend):
    """基于图像元数据的基础检测后端。

    该后端用于平台首版管线验证：
    1. 统一多来源图像输入后的检测结构；
    2. 在没有重模型依赖时仍能验证检测标准化和上层链路；
    3. 后续接入 YOLO、GroundingDINO、VLM 时只需替换后端。
    """

    def __init__(
        self,
        *,
        name: str = "metadata_detector",
        backend_kind: str = "metadata",
    ) -> None:
        self.spec = DetectorBackendSpec(
            name=name,
            backend_kind=backend_kind,
            metadata={"input_contract": "ImageFrame/CameraInfo", "supports_runtime_switch": True},
        )

    def detect(
        self,
        image_frame: ImageFrame,
        camera_info: Optional[CameraInfo] = None,
    ) -> DetectionBundle:
        del camera_info
        raw_2d = self._iter_raw_items(image_frame.metadata.get("detections_2d"))
        raw_3d = self._iter_raw_items(image_frame.metadata.get("detections_3d"))
        detections_2d = tuple(normalize_detection_2d(item, image_frame) for item in raw_2d)
        detections_3d = tuple(normalize_detection_3d(item) for item in raw_3d)
        return DetectionBundle(
            detections_2d=detections_2d,
            detections_3d=detections_3d,
            metadata={
                "backend_name": self.spec.name,
                "backend_kind": self.spec.backend_kind,
                "raw_detection_count": len(detections_2d) + len(detections_3d),
            },
        )

    def _iter_raw_items(self, payload: object) -> Tuple[object, ...]:
        if payload is None:
            return ()
        if isinstance(payload, tuple):
            return payload
        if isinstance(payload, list):
            return tuple(payload)
        if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes, dict)):
            return tuple(payload)
        raise GatewayError("检测元数据必须是列表或可迭代对象。")


class UltralyticsYoloDetectorBackend(DetectorBackend):
    """基于 Ultralytics YOLO 的生产级本地检测后端。"""

    def __init__(
        self,
        *,
        name: str = "yolo_local",
        weights: str = "yolo26n.pt",
        device: str = "",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        max_detections: int = 100,
    ) -> None:
        self._weights = str(weights).strip() or "yolo26n.pt"
        self._device = str(device).strip()
        self._confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))
        self._iou_threshold = max(0.0, min(1.0, float(iou_threshold)))
        self._image_size = max(32, int(image_size))
        self._max_detections = max(1, int(max_detections))
        self._model = None
        self.spec = DetectorBackendSpec(
            name=name,
            backend_kind="ultralytics_yolo",
            metadata={
                "weights": self._weights,
                "device": self._device or "auto",
                "supports_runtime_switch": True,
                "input_contract": "ImageFrame/CameraInfo",
            },
        )

    def detect(
        self,
        image_frame: ImageFrame,
        camera_info: Optional[CameraInfo] = None,
    ) -> DetectionBundle:
        del camera_info
        image_bgr = decode_image_frame_to_bgr(image_frame)
        results = self._predict(image_bgr)
        first_result = results[0] if results else None
        detections_2d = self._normalize_yolo_result(first_result, image_frame)
        return DetectionBundle(
            detections_2d=detections_2d,
            detections_3d=(),
            metadata={
                "backend_name": self.spec.name,
                "backend_kind": self.spec.backend_kind,
                "raw_detection_count": len(detections_2d),
                "weights": self._weights,
                "device": self._device or "auto",
            },
        )

    def _predict(self, image_bgr) -> List[Any]:
        model = self._get_model()
        kwargs = {
            "conf": self._confidence_threshold,
            "iou": self._iou_threshold,
            "imgsz": self._image_size,
            "max_det": self._max_detections,
            "verbose": False,
        }
        if self._device:
            kwargs["device"] = self._device
        return list(model.predict(image_bgr, **kwargs))

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "当前环境未安装 ultralytics，无法启用本地 YOLO 检测。"
            ) from exc
        self._model = YOLO(self._weights)
        return self._model

    def _normalize_yolo_result(
        self,
        result: Optional[Any],
        image_frame: ImageFrame,
    ) -> Tuple[Detection2D, ...]:
        if result is None:
            return ()
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return ()
        names = getattr(result, "names", None)
        if not isinstance(names, dict):
            names = getattr(self._get_model(), "names", {})

        xyxy_items = self._to_nested_list(getattr(boxes, "xyxy", ()))
        conf_items = self._to_list(getattr(boxes, "conf", ()))
        class_items = self._to_list(getattr(boxes, "cls", ()))

        detections = []
        for index, xyxy in enumerate(xyxy_items):
            if len(xyxy) < 4:
                continue
            x1, y1, x2, y2 = [float(value) for value in xyxy[:4]]
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            class_id = int(class_items[index]) if index < len(class_items) else -1
            score = float(conf_items[index]) if index < len(conf_items) else 0.0
            label = self._resolve_label(names, class_id)
            attributes = {
                "backend_name": self.spec.name,
                "backend_kind": self.spec.backend_kind,
                "model_name": self._weights,
                "class_id": class_id,
                "relative_area": round((width * height) / float(image_frame.width_px * image_frame.height_px), 6),
                "center_x_ratio": round(((x1 + x2) * 0.5) / float(image_frame.width_px), 6),
                "center_y_ratio": round(((y1 + y2) * 0.5) / float(image_frame.height_px), 6),
            }
            detections.append(
                normalize_detection_2d(
                    {
                        "label": label,
                        "score": score,
                        "bbox": {
                            "x_px": x1,
                            "y_px": y1,
                            "width_px": width,
                            "height_px": height,
                        },
                        "camera_id": image_frame.camera_id,
                        "attributes": attributes,
                    },
                    image_frame,
                )
            )
        return tuple(detections)

    def _resolve_label(self, names: Any, class_id: int) -> str:
        if isinstance(names, dict) and class_id in names:
            return str(names[class_id])
        return "class_%s" % class_id

    def _to_list(self, payload: Any) -> List[float]:
        if payload is None:
            return []
        if hasattr(payload, "tolist"):
            payload = payload.tolist()
        if isinstance(payload, tuple):
            return [float(item) for item in payload]
        if isinstance(payload, list):
            return [float(item) for item in payload]
        return [float(payload)]

    def _to_nested_list(self, payload: Any) -> List[List[float]]:
        if payload is None:
            return []
        if hasattr(payload, "tolist"):
            payload = payload.tolist()
        if not isinstance(payload, list):
            return []
        if payload and not isinstance(payload[0], list):
            return [[float(item) for item in payload]]
        return [[float(item) for item in row] for row in payload]


class DetectorPipeline:
    """检测后端选择与调度器。"""

    def __init__(
        self,
        backends: Tuple[DetectorBackend, ...],
        *,
        default_backend_name: Optional[str] = None,
    ) -> None:
        if not backends:
            raise ValueError("DetectorPipeline 至少需要一个检测后端。")
        self._backends = {backend.spec.name: backend for backend in backends}
        self._default_backend_name = default_backend_name or backends[0].spec.name
        if self._default_backend_name not in self._backends:
            raise ValueError("默认检测后端未注册。")

    def list_backends(self) -> Tuple[DetectorBackendSpec, ...]:
        """返回当前所有已注册检测后端规范。"""

        return tuple(backend.spec for backend in self._backends.values())

    def detect(
        self,
        image_frame: ImageFrame,
        camera_info: Optional[CameraInfo] = None,
        *,
        backend_name: Optional[str] = None,
    ) -> Tuple[DetectionBundle, DetectorBackendSpec]:
        """使用指定后端处理图像。"""

        resolved_backend_name = backend_name or self._default_backend_name
        backend = self._backends.get(resolved_backend_name)
        if backend is None:
            raise GatewayError(f"未注册检测后端：{resolved_backend_name}")
        bundle = backend.detect(image_frame, camera_info)
        return bundle, backend.spec
