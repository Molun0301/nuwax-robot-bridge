from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from contracts.base import MetadataDict
from contracts.image import CameraInfo, ImageFrame
from contracts.perception import Detection2D, Detection3D, Track
from contracts.runtime_views import SceneSummary
from typing import Optional, Tuple


@dataclass(frozen=True)
class DetectorBackendSpec:
    """检测后端规范。"""

    name: str
    backend_kind: str
    version: str = "0.1.0"
    supports_2d: bool = True
    supports_3d: bool = False
    metadata: MetadataDict = field(default_factory=dict)


@dataclass(frozen=True)
class DetectionBundle:
    """检测处理结果。"""

    detections_2d: Tuple[Detection2D, ...] = field(default_factory=tuple)
    detections_3d: Tuple[Detection3D, ...] = field(default_factory=tuple)
    metadata: MetadataDict = field(default_factory=dict)


class DetectorBackend(ABC):
    """检测后端接口。"""

    spec: DetectorBackendSpec

    @abstractmethod
    def detect(
        self,
        image_frame: ImageFrame,
        camera_info: Optional[CameraInfo] = None,
    ) -> DetectionBundle:
        """处理单帧图像并返回标准化检测结果。"""


@dataclass(frozen=True)
class TrackerBackendSpec:
    """跟踪后端规范。"""

    name: str
    backend_kind: str
    version: str = "0.1.0"
    supports_2d: bool = True
    supports_3d: bool = False
    metadata: MetadataDict = field(default_factory=dict)


class TrackerBackend(ABC):
    """跟踪后端接口。"""

    spec: TrackerBackendSpec

    @abstractmethod
    def update(
        self,
        image_frame: ImageFrame,
        *,
        detections_2d: Tuple[Detection2D, ...] = (),
        detections_3d: Tuple[Detection3D, ...] = (),
        camera_info: Optional[CameraInfo] = None,
    ) -> Tuple[Track, ...]:
        """基于当前帧检测结果更新轨迹。"""

    def reset(self, camera_id: Optional[str] = None) -> None:
        """重置一个或全部相机的跟踪状态。"""

        del camera_id


@dataclass(frozen=True)
class SceneDescriptionBackendSpec:
    """场景摘要后端规范。"""

    name: str
    backend_kind: str
    version: str = "0.1.0"
    metadata: MetadataDict = field(default_factory=dict)


class SceneDescriptionBackend(ABC):
    """场景摘要后端接口。"""

    spec: SceneDescriptionBackendSpec

    @abstractmethod
    def describe(
        self,
        *,
        camera_id: str,
        detections_2d: Tuple[Detection2D, ...],
        detections_3d: Tuple[Detection3D, ...],
        tracks: Tuple[Track, ...],
        image_frame: ImageFrame,
        camera_info: Optional[CameraInfo] = None,
    ) -> SceneSummary:
        """把检测与跟踪结果聚合成中文场景摘要。"""
