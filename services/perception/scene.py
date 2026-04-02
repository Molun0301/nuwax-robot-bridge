from __future__ import annotations

from collections import defaultdict
import json
import re

from contracts.image import CameraInfo, ImageFrame
from contracts.perception import Detection2D, Detection3D, Track, TrackState
from contracts.runtime_views import SceneObjectSummary, SceneSummary
from services.perception.base import SceneDescriptionBackend, SceneDescriptionBackendSpec
from services.perception.image_utils import image_frame_to_data_url
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _normalize_scene_label(value: object) -> str:
    label = str(value or "").strip().lower()
    label = re.sub(r"[^0-9a-z_]+", "_", label)
    return label.strip("_") or "unknown"


def _dedupe_strings(values: Iterable[object]) -> List[str]:
    items = []
    for raw_value in values:
        value = str(raw_value or "").strip()
        if not value:
            continue
        if value not in items:
            items.append(value)
    return items


def _merge_scene_objects(
    left_objects: Tuple[SceneObjectSummary, ...],
    right_objects: Tuple[SceneObjectSummary, ...],
) -> Tuple[SceneObjectSummary, ...]:
    merged: Dict[str, SceneObjectSummary] = {}
    for item in list(left_objects) + list(right_objects):
        key = _normalize_scene_label(item.label)
        current = merged.get(key)
        if current is None:
            merged[key] = item.model_copy(deep=True)
            continue
        merged[key] = current.model_copy(
            update={
                "count": max(int(current.count), int(item.count)),
                "tracked_count": max(int(current.tracked_count), int(item.tracked_count)),
                "max_score": max(float(current.max_score), float(item.max_score)),
                "camera_ids": _dedupe_strings(list(current.camera_ids) + list(item.camera_ids)),
                "track_ids": _dedupe_strings(list(current.track_ids) + list(item.track_ids)),
                "attributes": {
                    **dict(current.attributes),
                    **dict(item.attributes),
                },
            },
            deep=True,
        )
    return tuple(
        sorted(
            merged.values(),
            key=lambda item: (-int(item.count), -float(item.max_score), item.label),
        )
    )


class SimpleSceneDescriptionBackend(SceneDescriptionBackend):
    """首版中文场景摘要器。"""

    def __init__(
        self,
        *,
        name: str = "simple_scene_describer",
        max_labels: int = 5,
    ) -> None:
        self.spec = SceneDescriptionBackendSpec(
            name=name,
            backend_kind="rule_based_summary",
            metadata={"max_labels": max_labels},
        )
        self._max_labels = max(1, max_labels)

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
        del detections_3d, camera_info
        grouped: Dict[str, Dict[str, object]] = defaultdict(
            lambda: {
                "count": 0,
                "tracked_count": 0,
                "max_score": 0.0,
                "camera_ids": set(),
                "track_ids": set(),
                "attributes": {},
            }
        )

        for detection in detections_2d:
            bucket = grouped[detection.label]
            bucket["count"] = int(bucket["count"]) + 1
            bucket["max_score"] = max(float(bucket["max_score"]), float(detection.score))
            bucket["camera_ids"].add(detection.camera_id or camera_id)
            bucket["attributes"].update(dict(detection.attributes))

        active_track_states = {TrackState.TENTATIVE, TrackState.TRACKED}
        active_tracks = [track for track in tracks if track.state in active_track_states]
        for track in active_tracks:
            bucket = grouped[track.label]
            bucket["tracked_count"] = int(bucket["tracked_count"]) + 1
            bucket["track_ids"].add(track.track_id)
            if track.bbox is not None:
                bucket["camera_ids"].add(camera_id)
            bucket["max_score"] = max(float(bucket["max_score"]), float(track.score))
            bucket["attributes"].update(dict(track.attributes))

        objects = sorted(
            (
                SceneObjectSummary(
                    label=label,
                    count=int(bucket["count"]),
                    tracked_count=int(bucket["tracked_count"]),
                    max_score=float(bucket["max_score"]),
                    camera_ids=sorted(bucket["camera_ids"]),
                    track_ids=sorted(bucket["track_ids"]),
                    attributes=dict(bucket["attributes"]),
                )
                for label, bucket in grouped.items()
            ),
            key=lambda item: (-item.count, -item.max_score, item.label),
        )

        if not objects:
            headline = f"相机 {camera_id} 当前未发现明确目标。"
            details = [
                f"图像分辨率为 {image_frame.width_px}x{image_frame.height_px}。",
                "检测结果为空，当前更适合继续观察或切换视角。",
            ]
        else:
            top_objects = objects[: self._max_labels]
            headline = "当前画面检测到 " + "、".join(f"{item.label} {item.count} 个" for item in top_objects) + "。"
            details = [
                f"相机 {camera_id} 当前共有 {len(detections_2d)} 个二维检测。",
                f"当前活跃轨迹数量为 {len(active_tracks)}。",
            ]
            details.extend(
                f"{item.label} 共 {item.count} 个，活跃轨迹 {item.tracked_count} 条，最高置信度 {item.max_score:.2f}。"
                for item in top_objects
            )

        return SceneSummary(
            headline=headline,
            details=details,
            objects=objects,
            detection_count=len(detections_2d),
            active_track_count=len(active_tracks),
            metadata={
                "camera_id": camera_id,
                "image_width_px": image_frame.width_px,
                "image_height_px": image_frame.height_px,
                "visual_labels": [item.label for item in objects],
                "semantic_tags": [item.label for item in objects],
            },
        )


class OpenAICompatibleVisionSceneDescriptionBackend(SceneDescriptionBackend):
    """使用 OpenAI SDK 兼容接口的云端视觉语义后端。"""

    def __init__(
        self,
        *,
        name: str = "openai_vision_scene",
        model: str,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        timeout_sec: float = 20.0,
        max_tokens: int = 700,
        temperature: float = 0.0,
    ) -> None:
        self._model_name = str(model).strip()
        self._api_key = str(api_key).strip() or "EMPTY"
        self._base_url = str(base_url).strip() or "https://api.openai.com/v1"
        self._timeout_sec = max(1.0, float(timeout_sec))
        self._max_tokens = max(128, int(max_tokens))
        self._temperature = max(0.0, float(temperature))
        self._client = None
        self.spec = SceneDescriptionBackendSpec(
            name=name,
            backend_kind="openai_compatible_multimodal",
            metadata={
                "model_name": self._model_name,
                "base_url": self._base_url,
            },
        )

    def is_enabled(self) -> bool:
        return bool(self._model_name)

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
        del detections_3d
        if not self.is_enabled():
            raise RuntimeError("当前未配置云端视觉模型。")

        client = self._get_client()
        prompt = self._build_prompt(
            camera_id=camera_id,
            detections_2d=detections_2d,
            tracks=tracks,
            camera_info=camera_info,
            image_frame=image_frame,
        )
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_frame_to_data_url(image_frame)}},
        ]
        kwargs = {
            "model": self._model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是机器人宿主机的云端视觉语义后端。"
                        "你必须严格输出一个 JSON 对象，不要输出 Markdown，不要补充解释。"
                    ),
                },
                {"role": "user", "content": content},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        response = None
        try:
            response = client.chat.completions.create(
                response_format={"type": "json_object"},
                **kwargs,
            )
        except Exception:
            response = client.chat.completions.create(**kwargs)

        payload = self._parse_json_payload(self._extract_message_text(response))
        objects = tuple(self._build_scene_objects(payload.get("objects"), camera_id))
        details = _dedupe_strings(payload.get("details") or [])
        tags = _dedupe_strings(payload.get("tags") or [])
        relations = _dedupe_strings(payload.get("relations") or [])
        headline = str(payload.get("headline") or "").strip()
        if not headline:
            if objects:
                headline = "画面中主要可见 " + "、".join(item.label for item in objects[:3]) + "。"
            else:
                headline = "云端语义分析未识别到明确目标。"
        return SceneSummary(
            headline=headline,
            details=details,
            objects=list(objects),
            detection_count=len(detections_2d),
            active_track_count=len([item for item in tracks if item.state in {TrackState.TENTATIVE, TrackState.TRACKED}]),
            metadata={
                "camera_id": camera_id,
                "cloud_vision_model": self._model_name,
                "cloud_vision_base_url": self._base_url,
                "semantic_tags": tags,
                "semantic_relations": relations,
                "visual_labels": [item.label for item in objects],
            },
        )

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "当前环境未安装 openai，无法启用云端视觉语义后端。"
            ) from exc
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout_sec,
        )
        return self._client

    def _build_prompt(
        self,
        *,
        camera_id: str,
        detections_2d: Tuple[Detection2D, ...],
        tracks: Tuple[Track, ...],
        camera_info: Optional[CameraInfo],
        image_frame: ImageFrame,
    ) -> str:
        detection_hints = [
            {
                "label": item.label,
                "score": round(float(item.score), 4),
                "bbox": {
                    "x_px": round(float(item.bbox.x_px), 2),
                    "y_px": round(float(item.bbox.y_px), 2),
                    "width_px": round(float(item.bbox.width_px), 2),
                    "height_px": round(float(item.bbox.height_px), 2),
                },
            }
            for item in detections_2d[:12]
        ]
        track_hints = [
            {
                "track_id": item.track_id,
                "label": item.label,
                "state": item.state.value,
                "score": round(float(item.score), 4),
            }
            for item in tracks[:12]
        ]
        camera_hint = {
            "camera_id": camera_id,
            "resolution": "%sx%s" % (image_frame.width_px, image_frame.height_px),
            "fx": round(float(camera_info.fx), 4) if camera_info is not None else None,
            "fy": round(float(camera_info.fy), 4) if camera_info is not None else None,
        }
        return (
            "请基于这张机器人当前图像，生成适合具身记忆与导航复核使用的结构化识别结果。"
            "要求：\n"
            "1. headline 和 details 用中文。\n"
            "2. objects[].label 使用英文小写下划线标签，尽量稳定。\n"
            "3. tags 给出适合写入向量记忆的中文关键词。\n"
            "4. relations 给出与导航或复核有关的相对关系，例如 near_left / on_table / beside_dock。\n"
            "5. 只输出 JSON，对象结构为 "
            '{"headline":"", "details":[""], "tags":[""], "relations":[""], '
            '"objects":[{"label":"", "count":1, "confidence":0.0, "attributes":{"display_name_zh":"","color":"","state":"","relative_location":""}}]}.'
            "\n以下是本地检测提示，可参考但不要被它限制：\n"
            "camera=%s\n"
            "detections=%s\n"
            "tracks=%s\n"
            % (
                json.dumps(camera_hint, ensure_ascii=False),
                json.dumps(detection_hints, ensure_ascii=False),
                json.dumps(track_hints, ensure_ascii=False),
            )
        )

    def _extract_message_text(self, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            values = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    values.append(str(item.get("text", "")))
            return "\n".join(value for value in values if value)
        return str(content or "")

    def _parse_json_payload(self, content: str) -> Dict[str, Any]:
        raw_content = str(content or "").strip()
        if not raw_content:
            return {}
        try:
            payload = json.loads(raw_content)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        start = raw_content.find("{")
        end = raw_content.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(raw_content[start : end + 1])
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                return {}
        return {}

    def _build_scene_objects(self, payload: Any, camera_id: str) -> List[SceneObjectSummary]:
        objects: List[SceneObjectSummary] = []
        if not isinstance(payload, list):
            return objects
        for raw_item in payload:
            if not isinstance(raw_item, dict):
                continue
            label = _normalize_scene_label(raw_item.get("label"))
            attributes = raw_item.get("attributes")
            objects.append(
                SceneObjectSummary(
                    label=label,
                    count=max(0, int(raw_item.get("count", 1) or 1)),
                    tracked_count=0,
                    max_score=max(0.0, min(1.0, float(raw_item.get("confidence", 0.0) or 0.0))),
                    camera_ids=[camera_id],
                    track_ids=[],
                    attributes=dict(attributes or {}),
                )
            )
        return objects


class HybridSceneDescriptionBackend(SceneDescriptionBackend):
    """融合本地检测摘要与云端语义识别结果。"""

    def __init__(
        self,
        *,
        local_backend: SceneDescriptionBackend,
        cloud_backend: Optional[OpenAICompatibleVisionSceneDescriptionBackend] = None,
        name: str = "hybrid_scene_describer",
    ) -> None:
        self._local_backend = local_backend
        self._cloud_backend = cloud_backend
        self.spec = SceneDescriptionBackendSpec(
            name=name,
            backend_kind="hybrid_local_cloud_scene",
            metadata={
                "local_backend": local_backend.spec.name,
                "cloud_backend": cloud_backend.spec.name if cloud_backend is not None else "",
            },
        )

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
        local_summary = self._local_backend.describe(
            camera_id=camera_id,
            detections_2d=detections_2d,
            detections_3d=detections_3d,
            tracks=tracks,
            image_frame=image_frame,
            camera_info=camera_info,
        )
        cloud_enabled = self._cloud_backend is not None and (
            not hasattr(self._cloud_backend, "is_enabled") or bool(self._cloud_backend.is_enabled())
        )
        if not cloud_enabled:
            return local_summary.model_copy(
                update={
                    "metadata": {
                        **dict(local_summary.metadata),
                        "scene_backend_chain": [self._local_backend.spec.name],
                    }
                },
                deep=True,
            )

        try:
            cloud_summary = self._cloud_backend.describe(
                camera_id=camera_id,
                detections_2d=detections_2d,
                detections_3d=detections_3d,
                tracks=tracks,
                image_frame=image_frame,
                camera_info=camera_info,
            )
        except Exception as exc:
            return local_summary.model_copy(
                update={
                    "metadata": {
                        **dict(local_summary.metadata),
                        "scene_backend_chain": [self._local_backend.spec.name, self._cloud_backend.spec.name],
                        "cloud_vision_error": str(exc),
                    }
                },
                deep=True,
            )

        merged_objects = _merge_scene_objects(
            tuple(local_summary.objects),
            tuple(cloud_summary.objects),
        )
        merged_details = _dedupe_strings(list(cloud_summary.details) + list(local_summary.details))
        merged_tags = _dedupe_strings(
            list(local_summary.metadata.get("semantic_tags") or [])
            + list(cloud_summary.metadata.get("semantic_tags") or [])
            + [item.label for item in merged_objects]
        )
        merged_relations = _dedupe_strings(cloud_summary.metadata.get("semantic_relations") or [])
        merged_headline = str(cloud_summary.headline or "").strip() or local_summary.headline
        return SceneSummary(
            headline=merged_headline,
            details=merged_details,
            objects=list(merged_objects),
            detection_count=local_summary.detection_count,
            active_track_count=local_summary.active_track_count,
            metadata={
                **dict(local_summary.metadata),
                **dict(cloud_summary.metadata),
                "visual_labels": [item.label for item in merged_objects],
                "semantic_tags": merged_tags,
                "semantic_relations": merged_relations,
                "scene_backend_chain": [self._local_backend.spec.name, self._cloud_backend.spec.name],
                "local_scene_headline": local_summary.headline,
                "cloud_scene_headline": cloud_summary.headline,
            },
        )
