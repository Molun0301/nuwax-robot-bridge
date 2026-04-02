from __future__ import annotations

from pathlib import Path
import time

import pytest

from contracts.artifacts import ArtifactRetentionPolicy
from contracts.geometry import Pose, Quaternion, Vector3
from contracts.image import CameraInfo, ImageEncoding, ImageFrame
from contracts.runtime_views import LocalizationSnapshot
from contracts.perception import TrackState
from contracts.runtime_views import SceneObjectSummary, SceneSummary
from core import EventBus, StateNamespace, StateStore
from gateways.artifacts import LocalArtifactStore
from providers.image import ImageProvider
from services import ArtifactService
from services.perception import (
    Basic2DTrackerBackend,
    DetectorPipeline,
    HybridSceneDescriptionBackend,
    MetadataDrivenDetectorBackend,
    PerceptionService,
    PerceptionVideoRuntime,
    SceneDescriptionBackend,
    SceneDescriptionBackendSpec,
    SimpleSceneDescriptionBackend,
    TrackLifecyclePolicy,
    UltralyticsYoloDetectorBackend,
)
from typing import Optional, Tuple


class _ProviderOwner:
    """测试用提供器宿主。"""

    def __init__(self, providers: ImageProvider) -> None:
        self.providers = providers


class UsbCameraBundle(ImageProvider):
    """模拟 USB 相机输入源。"""

    provider_name = "usb_camera_bundle"
    provider_version = "0.1.0"

    def is_available(self) -> bool:
        return True

    def capture_image(self, camera_id: Optional[str] = None) -> ImageFrame:
        camera = camera_id or "front_camera"
        return ImageFrame(
            camera_id=camera,
            frame_id=f"world/test/{camera}",
            width_px=320,
            height_px=240,
            encoding=ImageEncoding.JPEG,
            data=b"usb-image",
            metadata={
                "source_kind": "usb",
                "detections_2d": [
                    {
                        "label": "person",
                        "score": 0.95,
                        "bbox": {"x_px": 20, "y_px": 30, "width_px": 80, "height_px": 120},
                    }
                ],
            },
        )

    def get_camera_info(self, camera_id: Optional[str] = None) -> CameraInfo:
        return CameraInfo(
            camera_id=camera_id or "front_camera",
            frame_id=f"world/test/{camera_id or 'front_camera'}",
            width_px=320,
            height_px=240,
            fx=120.0,
            fy=120.0,
            cx=160.0,
            cy=120.0,
        )


class RtspCameraBundle(ImageProvider):
    """模拟 RTSP 相机输入源。"""

    provider_name = "rtsp_camera_bundle"
    provider_version = "0.1.0"

    def is_available(self) -> bool:
        return True

    def capture_image(self, camera_id: Optional[str] = None) -> ImageFrame:
        camera = camera_id or "front_camera"
        return ImageFrame(
            camera_id=camera,
            frame_id=f"world/test/{camera}",
            width_px=320,
            height_px=240,
            encoding=ImageEncoding.JPEG,
            data=b"rtsp-image",
            metadata={
                "source_kind": "rtsp",
                "detections_2d": [
                    {
                        "label": "person",
                        "score": 0.95,
                        "bbox": {"x_px": 20, "y_px": 30, "width_px": 80, "height_px": 120},
                    }
                ],
            },
        )

    def get_camera_info(self, camera_id: Optional[str] = None) -> CameraInfo:
        return CameraInfo(
            camera_id=camera_id or "front_camera",
            frame_id=f"world/test/{camera_id or 'front_camera'}",
            width_px=320,
            height_px=240,
            fx=120.0,
            fy=120.0,
            cx=160.0,
            cy=120.0,
        )


def _build_perception_service(
    tmp_path: Path,
    provider_owner,
    *,
    tracker_backend: Optional[Basic2DTrackerBackend] = None,
    detector_pipeline: Optional[DetectorPipeline] = None,
    scene_description_backend: Optional[SceneDescriptionBackend] = None,
) -> Tuple[PerceptionService, StateStore]:
    artifact_service = ArtifactService(
        LocalArtifactStore(str(tmp_path / "artifacts"), "http://testserver"),
        retention_policy=ArtifactRetentionPolicy(
            retention_days=7,
            max_count=20,
            max_total_bytes=1024 * 1024,
            cleanup_batch_size=10,
        ),
    )
    state_store = StateStore()
    service = PerceptionService(
        provider_owner=provider_owner,
        artifact_service=artifact_service,
        state_store=state_store,
        detector_pipeline=detector_pipeline or DetectorPipeline((MetadataDrivenDetectorBackend(),)),
        tracker_backend=tracker_backend,
        scene_description_backend=scene_description_backend,
        event_bus=EventBus(),
        history_limit=10,
        pipeline_name="test_perception_pipeline",
    )
    return service, state_store


class _FakeYoloDetectorBackend(UltralyticsYoloDetectorBackend):
    """通过伪造推理结果验证 YOLO 标准化逻辑。"""

    def _predict(self, image_bgr):
        del image_bgr

        class _Boxes:
            xyxy = [[18.0, 24.0, 90.0, 160.0]]
            conf = [0.93]
            cls = [0]

        class _Result:
            boxes = _Boxes()
            names = {0: "Person"}

        return [_Result()]


class _StubCloudSceneBackend(SceneDescriptionBackend):
    """测试用云端语义后端。"""

    def __init__(self) -> None:
        self.spec = SceneDescriptionBackendSpec(
            name="stub_cloud_scene",
            backend_kind="openai_compatible_multimodal",
        )

    def describe(
        self,
        *,
        camera_id: str,
        detections_2d,
        detections_3d,
        tracks,
        image_frame,
        camera_info=None,
    ) -> SceneSummary:
        del detections_2d, detections_3d, tracks, image_frame, camera_info
        return SceneSummary(
            headline="画面里有一名人员，靠近红色补给架。",
            details=["人员位于画面左侧。", "补给架位于人员右侧。"],
            objects=[
                SceneObjectSummary(
                    label="person",
                    count=1,
                    tracked_count=0,
                    max_score=0.82,
                    camera_ids=[camera_id],
                    track_ids=[],
                    attributes={"display_name_zh": "人员", "relative_location": "left"},
                ),
                SceneObjectSummary(
                    label="charger",
                    count=1,
                    tracked_count=0,
                    max_score=0.74,
                    camera_ids=[camera_id],
                    track_ids=[],
                    attributes={"display_name_zh": "补给架", "color": "red"},
                ),
            ],
            detection_count=1,
            active_track_count=1,
            metadata={
                "semantic_tags": ["红色补给架", "人员附近"],
                "semantic_relations": ["person_beside_charger"],
                "visual_labels": ["person", "charger"],
                "cloud_vision_model": "fake-vision-model",
            },
        )


class _StaticLocalizationService:
    """测试用定位服务。"""

    def __init__(self) -> None:
        self.pose = Pose(
            frame_id="map",
            position=Vector3(x=0.0, y=0.0, z=0.0),
            orientation=Quaternion(w=1.0),
        )

    def get_latest_snapshot(self):
        return LocalizationSnapshot(
            source_name="test_localization",
            current_pose=self.pose,
        )

    def is_available(self) -> bool:
        return True

    def refresh(self):
        return self.get_latest_snapshot()


class _FakeMemoryService:
    """测试用记忆服务。"""

    def __init__(self) -> None:
        self.calls = []

    def remember_current_scene(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {"ok": True}


def test_perception_service_requires_standard_imageframe_and_camerainfo(tmp_path: Path) -> None:
    """感知服务必须只接受标准 ImageFrame/CameraInfo。"""

    provider_owner = _ProviderOwner(UsbCameraBundle())
    service, state_store = _build_perception_service(tmp_path, provider_owner)
    image_frame = provider_owner.providers.capture_image("front_camera")
    camera_info = provider_owner.providers.get_camera_info("front_camera")

    context = service.process_image(image_frame, camera_info, requested_by="tester")
    latest_entry = state_store.read(StateNamespace.PERCEPTION, "front_camera")

    assert context.camera_info is not None
    assert context.camera_info.camera_id == "front_camera"
    assert latest_entry is not None
    assert latest_entry.value.camera_id == "front_camera"

    with pytest.raises(TypeError):
        service.process_image("not-an-image-frame", camera_info)


def test_metadata_detector_normalizes_detection_results() -> None:
    """检测标准化应修正标签、分数与越界框。"""

    image_frame = ImageFrame(
        camera_id="front_camera",
        frame_id="world/test/front_camera",
        width_px=100,
        height_px=80,
        encoding=ImageEncoding.JPEG,
        data=b"image",
        metadata={
            "detections_2d": [
                {
                    "label": " Person ",
                    "score": 1.7,
                    "bbox": {"x_px": -10, "y_px": -5, "width_px": 200, "height_px": 120},
                }
            ]
        },
    )

    bundle = MetadataDrivenDetectorBackend().detect(image_frame)
    detection = bundle.detections_2d[0]

    assert detection.label == "person"
    assert detection.score == pytest.approx(1.0)
    assert detection.bbox.x_px == pytest.approx(0.0)
    assert detection.bbox.y_px == pytest.approx(0.0)
    assert detection.bbox.width_px == pytest.approx(100.0)
    assert detection.bbox.height_px == pytest.approx(80.0)


def test_yolo_detector_backend_normalizes_ultralytics_results() -> None:
    """YOLO 后端应把 Ultralytics 结果规整成标准 Detection2D。"""

    image_frame = ImageFrame(
        camera_id="front_camera",
        frame_id="world/test/front_camera",
        width_px=160,
        height_px=120,
        encoding=ImageEncoding.BGR8,
        data=bytes([0, 0, 0] * 160 * 120),
    )

    backend = _FakeYoloDetectorBackend(weights="yolo26n.pt")
    bundle = backend.detect(image_frame)

    assert bundle.detections_2d
    detection = bundle.detections_2d[0]
    assert detection.label == "person"
    assert detection.score == pytest.approx(0.93)
    assert detection.attributes["model_name"] == "yolo26n.pt"
    assert detection.attributes["class_id"] == 0


def test_yolo_detector_backend_prefers_engine_when_available(tmp_path: Path) -> None:
    """当 TensorRT 引擎存在时，YOLO 后端应优先加载 .engine。"""

    weights_path = tmp_path / "yolo26n.pt"
    engine_path = tmp_path / "yolo26n.engine"
    weights_path.write_bytes(b"fake-pt")
    engine_path.write_bytes(b"fake-engine")

    backend = UltralyticsYoloDetectorBackend(
        weights=str(weights_path),
        engine_path=str(engine_path),
        runtime_preference="tensorrt",
    )
    resolved_source, backend_kind = backend._resolve_model_source()

    assert resolved_source == str(engine_path)
    assert backend_kind == "ultralytics_yolo_tensorrt"


def test_perception_video_runtime_uses_keyframes_and_burst_without_memory_flooding(tmp_path: Path) -> None:
    """关键帧应写记忆，burst 复核默认不应把每帧都写入记忆。"""

    provider_owner = _ProviderOwner(UsbCameraBundle())
    service, state_store = _build_perception_service(tmp_path, provider_owner)
    localization_service = _StaticLocalizationService()
    memory_service = _FakeMemoryService()
    runtime = PerceptionVideoRuntime(
        perception_service=service,
        state_store=state_store,
        localization_service=localization_service,
        mapping_service=None,
        memory_service=memory_service,
        enabled=True,
        auto_start=False,
        camera_id="front_camera",
        interval_sec=0.1,
        store_artifact=False,
        keyframe_min_interval_sec=0.1,
        keyframe_max_interval_sec=2.0,
        keyframe_translation_threshold_m=0.3,
        keyframe_yaw_threshold_deg=10.0,
        remember_keyframes=True,
        remember_min_interval_sec=0.1,
        burst_frame_count=2,
        burst_interval_sec=0.01,
    )

    first_message = runtime.run_once()
    assert first_message is not None
    assert len(memory_service.calls) == 1

    skipped_message = runtime.run_once()
    assert skipped_message is None
    assert len(memory_service.calls) == 1

    time.sleep(0.12)
    localization_service.pose = Pose(
        frame_id="map",
        position=Vector3(x=0.5, y=0.0, z=0.0),
        orientation=Quaternion(w=1.0),
    )
    second_message = runtime.run_once()
    assert second_message is not None
    assert len(memory_service.calls) == 2

    burst_messages = runtime.run_burst(frame_count=2, interval_sec=0.01, remember_result=False)
    assert len(burst_messages) == 2
    assert len(memory_service.calls) == 2

    status = runtime.get_status()
    assert status.processed_frames >= 4
    assert status.metadata["camera_id"] == "front_camera"


def test_tracker_lifecycle_flows_from_tentative_to_removed(tmp_path: Path) -> None:
    """首版二维跟踪器应明确 tentative/tracked/lost/removed 语义。"""

    tracker_backend = Basic2DTrackerBackend(
        lifecycle_policy=TrackLifecyclePolicy(
            iou_match_threshold=0.3,
            tentative_hits_required=2,
            lost_ttl_frames=1,
        )
    )
    service, _ = _build_perception_service(tmp_path, _ProviderOwner(UsbCameraBundle()), tracker_backend=tracker_backend)

    def _frame(with_detection: bool) -> ImageFrame:
        metadata = {}
        if with_detection:
            metadata["detections_2d"] = [
                {
                    "label": "person",
                    "score": 0.92,
                    "bbox": {"x_px": 20, "y_px": 30, "width_px": 80, "height_px": 120},
                }
            ]
        return ImageFrame(
            camera_id="front_camera",
            frame_id="world/test/front_camera",
            width_px=320,
            height_px=240,
            encoding=ImageEncoding.JPEG,
            data=b"frame",
            metadata=metadata,
        )

    camera_info = CameraInfo(
        camera_id="front_camera",
        frame_id="world/test/front_camera",
        width_px=320,
        height_px=240,
        fx=120.0,
        fy=120.0,
        cx=160.0,
        cy=120.0,
    )

    first = service.process_image(_frame(True), camera_info)
    second = service.process_image(_frame(True), camera_info)
    third = service.process_image(_frame(False), camera_info)
    fourth = service.process_image(_frame(False), camera_info)

    assert first.observation.tracks[0].state == TrackState.TENTATIVE
    assert second.observation.tracks[0].state == TrackState.TRACKED
    assert third.observation.tracks[0].state == TrackState.LOST
    assert fourth.observation.tracks[0].state == TrackState.REMOVED


def test_source_switch_does_not_change_top_level_perception_logic(tmp_path: Path) -> None:
    """切换相机来源时，上层感知逻辑不应改动。"""

    usb_service, _ = _build_perception_service(tmp_path / "usb", _ProviderOwner(UsbCameraBundle()))
    rtsp_service, _ = _build_perception_service(tmp_path / "rtsp", _ProviderOwner(RtspCameraBundle()))

    usb_context = usb_service.perceive_current_scene(camera_id="front_camera", requested_by="tester")
    rtsp_context = rtsp_service.perceive_current_scene(camera_id="front_camera", requested_by="tester")

    assert usb_context.scene_summary.headline == rtsp_context.scene_summary.headline
    assert usb_context.observation.detections_2d[0].label == "person"
    assert rtsp_context.observation.detections_2d[0].track_id is not None
    assert usb_context.detector_backend == rtsp_context.detector_backend


def test_hybrid_scene_backend_merges_local_and_cloud_semantics(tmp_path: Path) -> None:
    """本地检测与云端语义识别应合并为统一场景摘要。"""

    provider_owner = _ProviderOwner(UsbCameraBundle())
    service, _ = _build_perception_service(
        tmp_path,
        provider_owner,
        scene_description_backend=HybridSceneDescriptionBackend(
            local_backend=SimpleSceneDescriptionBackend(),
            cloud_backend=_StubCloudSceneBackend(),
        ),
    )

    context = service.perceive_current_scene(camera_id="front_camera", requested_by="tester")

    assert context.scene_summary.headline == "画面里有一名人员，靠近红色补给架。"
    assert "红色补给架" in context.scene_summary.metadata["semantic_tags"]
    assert "person_beside_charger" in context.scene_summary.metadata["semantic_relations"]
    assert context.scene_summary.objects[0].label == "person"
