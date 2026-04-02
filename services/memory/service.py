from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import logging
import re
from typing import Deque, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from contracts.base import utc_now
from contracts.events import RuntimeEventCategory
from contracts.geometry import Pose, Vector3
from contracts.maps import SemanticMap, SemanticRegion
from contracts.memory import (
    MemoryArrivalVerification,
    MemoryNavigationCandidate,
    MemoryPayloadFilter,
    MemoryQueryMatch,
    MemoryQueryResult,
    MemoryRecordKind,
    MemoryStoreSummary,
    SemanticMemoryEntry,
    TaggedLocation,
)
from contracts.naming import build_location_id, build_memory_id, build_observation_event_id
from contracts.spatial_memory import (
    GroundingQueryPlan,
    NavigationCandidate,
    ObservationEvent,
    SemanticInstance,
    VerificationResult,
)
from core import EventBus, StateNamespace, StateStore
from gateways.artifacts import LocalArtifactStore
from gateways.errors import GatewayError
from services.localization import LocalizationService
from services.mapping import MappingService
from services.memory.grounding_query_planner import GroundingQueryPlanner
from services.memory.instance_association_service import AssociationOutcome, InstanceAssociationService
from services.memory.inspection_pose_planner import InspectionPosePlanner
from services.memory.repository import MemoryRepository
from services.memory.semantic_map_builder import SemanticMapBuildResult, SemanticMapBuilder
from services.memory.vector_store import (
    EPISODIC_OBSERVATIONS_COLLECTION,
    IMAGE_VECTOR_NAME,
    OBJECT_INSTANCES_COLLECTION,
    PLACE_NODES_COLLECTION,
    TEXT_VECTOR_NAME,
    NamedVector,
    SQLiteNamedVectorStore,
    VectorPoint,
    VectorSearchHit,
)
from services.memory.vectorizer import (
    HashingTextEmbedder,
    TextEmbedder,
    VisionLanguageEmbedder,
    build_text_embedder,
    build_vision_language_embedder,
)
from services.memory.vision_to_map_projection_service import ProjectedObservation, VisionToMapProjectionService
from services.observation_service import ObservationService
from services.perception import PerceptionService

LOGGER = logging.getLogger(__name__)


@dataclass
class _ScoredCandidate:
    """检索阶段中的中间候选。"""

    point: VectorPoint
    record_kind: MemoryRecordKind
    tagged_location: Optional[TaggedLocation]
    semantic_memory: Optional[SemanticMemoryEntry]
    semantic_instance: Optional[SemanticInstance]
    observation_event: Optional[ObservationEvent]
    text_recall_score: float
    lexical_score: float
    lexical_reason: str
    image_score: float
    spatial_score: float
    recency_score: float
    distance_m: Optional[float]
    final_score: float


class MemoryService:
    """统一地点记忆、语义记忆与多模态向量检索服务。"""

    def __init__(
        self,
        *,
        localization_service: LocalizationService,
        mapping_service: MappingService,
        observation_service: ObservationService,
        perception_service: PerceptionService,
        state_store: StateStore,
        event_bus: Optional[EventBus] = None,
        artifact_store: Optional[LocalArtifactStore] = None,
        history_limit: int = 200,
        memory_db_path: str = ":memory:",
        embedding_model: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        text_embedding_model: Optional[str] = None,
        text_embedding_dimension: Optional[int] = None,
        image_embedding_model: str = "disabled",
        image_embedding_dimension: int = 512,
        text_embedder: Optional[TextEmbedder] = None,
        vision_language_embedder: Optional[VisionLanguageEmbedder] = None,
        vector_store: Optional[SQLiteNamedVectorStore] = None,
    ) -> None:
        self._localization_service = localization_service
        self._mapping_service = mapping_service
        self._observation_service = observation_service
        self._perception_service = perception_service
        self._state_store = state_store
        self._event_bus = event_bus
        self._artifact_store = artifact_store
        self._repository = MemoryRepository(memory_db_path)
        self._vector_store = vector_store or SQLiteNamedVectorStore(memory_db_path)

        resolved_text_model = (
            text_embedding_model
            or embedding_model
            or "sentence-transformers:BAAI/bge-m3"
        )
        resolved_text_dimension = int(text_embedding_dimension or embedding_dimension or 1024)
        self._text_embedder = text_embedder or self._build_text_embedder_with_fallback(
            resolved_text_model,
            resolved_text_dimension,
        )
        self._image_embedder = (
            vision_language_embedder
            or self._build_image_embedder_with_fallback(
                image_embedding_model,
                image_embedding_dimension,
            )
        )
        self._inspection_pose_planner = InspectionPosePlanner()
        self._semantic_map_builder = SemanticMapBuilder(
            inspection_pose_planner=self._inspection_pose_planner,
        )
        self._vision_projection_service = VisionToMapProjectionService()
        self._instance_association_service = InstanceAssociationService(
            text_embedder=self._text_embedder,
            inspection_pose_planner=self._inspection_pose_planner,
        )
        self._grounding_query_planner = GroundingQueryPlanner()

        self._tagged_locations_by_id: Dict[str, TaggedLocation] = {}
        self._semantic_entries_by_id: Dict[str, SemanticMemoryEntry] = {}
        self._semantic_instances_by_id: Dict[str, SemanticInstance] = {}
        self._observation_events_by_id: Dict[str, ObservationEvent] = {}
        self._latest_semantic_map_context = SemanticMapBuildResult(
            map_version_id=None,
            frame_id="map",
            semantic_regions=(),
            anchors_by_id={},
            topology_nodes_by_id={},
        )
        self._tagged_location_history: Deque[TaggedLocation] = deque(maxlen=max(1, history_limit))
        self._semantic_history: Deque[SemanticMemoryEntry] = deque(maxlen=max(1, history_limit))
        self._semantic_instance_history: Deque[SemanticInstance] = deque(maxlen=max(1, history_limit))
        self._observation_event_history: Deque[ObservationEvent] = deque(maxlen=max(1, history_limit))
        self._load_persisted_memory()

    def _build_text_embedder_with_fallback(
        self,
        model_name: str,
        dimension: int,
    ) -> TextEmbedder:
        """优先构造配置指定的文本向量器，失败时回退到 hashing。"""

        try:
            return build_text_embedder(model_name, dimension)
        except Exception as exc:
            LOGGER.warning(
                "文本向量模型初始化失败，已回退到 hashing-v1。requested=%s error=%s",
                model_name,
                exc,
            )
            fallback_dimension = dimension if dimension > 0 else 256
            return HashingTextEmbedder(
                model_name="hashing-v1",
                dimension=fallback_dimension,
            )

    def _build_image_embedder_with_fallback(
        self,
        model_name: str,
        dimension: int,
    ) -> Optional[VisionLanguageEmbedder]:
        """优先构造图文共享向量器，失败时关闭图像向量链路。"""

        try:
            return build_vision_language_embedder(model_name, dimension)
        except Exception as exc:
            LOGGER.warning(
                "图像向量模型初始化失败，已关闭图像向量链路。requested=%s error=%s",
                model_name,
                exc,
            )
            return None

    def tag_location(
        self,
        name: str,
        *,
        aliases: Union[List[str], Tuple[str, ...], None] = None,
        description: Optional[str] = None,
        camera_id: Optional[str] = None,
        attach_latest_observation: bool = True,
        attach_latest_perception: bool = True,
        auto_create_memory: bool = True,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Tuple[TaggedLocation, Optional[SemanticMemoryEntry]]:
        """把当前位置写成带空间锚点的命名地点。"""

        normalized_name = self._normalize_text(name)
        if not normalized_name:
            raise GatewayError(
                "地点名称不能为空。",
                error_code="invalid_params",
                http_status=422,
                jsonrpc_code=-32602,
            )

        localization_snapshot = self._get_or_refresh_localization()
        if localization_snapshot.current_pose is None:
            raise GatewayError("当前没有可用定位结果，无法标记地点。")

        map_snapshot = self._get_map_snapshot()
        observation_context = None
        perception_context = None
        if attach_latest_observation:
            observation_context = self._observation_service.get_latest_observation(camera_id)
        if attach_latest_perception:
            perception_context = self._perception_service.get_latest_perception(camera_id)

        semantic_region_id, semantic_labels = self._resolve_semantic_region(
            localization_snapshot.current_pose,
            map_snapshot.semantic_map if map_snapshot is not None else None,
        )
        topo_node_id = self._resolve_topology_node_id(
            localization_snapshot.current_pose,
            map_snapshot.metadata if map_snapshot is not None else {},
            metadata,
        )
        self._refresh_semantic_map_context(map_snapshot)
        anchor_id = self._resolve_anchor_id_for_pose(
            localization_snapshot.current_pose,
            semantic_region_id=semantic_region_id,
            topo_node_id=topo_node_id,
        )
        merged_metadata = {
            "description": description or "",
            "camera_id": camera_id or "",
            "anchor_id": anchor_id or "",
            **dict(metadata or {}),
        }
        if perception_context is not None:
            merged_metadata.update(self._build_scene_memory_metadata(perception_context))
        tagged_location = TaggedLocation(
            location_id=build_location_id("tag"),
            name=name.strip(),
            normalized_name=normalized_name,
            aliases=self._normalize_aliases(aliases),
            pose=localization_snapshot.current_pose,
            map_version_id=map_snapshot.version_id if map_snapshot is not None else None,
            localization_source_name=localization_snapshot.source_name,
            topo_node_id=topo_node_id,
            semantic_region_id=semantic_region_id,
            semantic_labels=semantic_labels,
            observation_id=(
                observation_context.observation.observation_id if observation_context is not None else None
            ),
            perception_headline=(
                perception_context.scene_summary.headline if perception_context is not None else description
            ),
            metadata=merged_metadata,
        )

        memory_entry = None
        if auto_create_memory:
            scene_labels = (
                self._collect_scene_visual_labels(perception_context.scene_summary)
                if perception_context is not None
                else []
            )
            scene_tags = (
                self._collect_scene_vision_tags(perception_context.scene_summary)
                if perception_context is not None
                else []
            )
            scene_summary = (
                description
                or (
                    perception_context.scene_summary.headline
                    if perception_context is not None
                    else ""
                )
                or "记录了一个可导航地点。"
            )
            memory_entry = self._create_scene_memory_entry(
                title="地点记忆：" + tagged_location.name,
                summary=scene_summary,
                tags=[
                    tagged_location.name,
                    *tagged_location.aliases,
                    *scene_labels,
                    *scene_tags,
                    *tagged_location.semantic_labels,
                ],
                linked_location=tagged_location,
                camera_id=camera_id,
                metadata={"source": "tag_location", **dict(merged_metadata)},
            )
            tagged_location = tagged_location.model_copy(
                update={"memory_id": memory_entry.memory_id},
                deep=True,
            )

        self._tagged_locations_by_id[tagged_location.location_id] = tagged_location
        self._tagged_location_history.append(tagged_location)
        self._write_tagged_location_state(tagged_location)
        self._repository.upsert_tagged_location(tagged_location)
        self._upsert_vector_point_for_location(tagged_location)
        self._publish_memory_event(
            "memory.location_tagged",
            "地点 %s 已写入记忆。" % tagged_location.name,
            payload={"location_id": tagged_location.location_id, "name": tagged_location.name},
        )
        return tagged_location, memory_entry

    def remember_current_scene(
        self,
        *,
        title: Optional[str] = None,
        camera_id: Optional[str] = None,
        summary_override: Optional[str] = None,
        tags: Union[List[str], Tuple[str, ...], None] = None,
        linked_location_id: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> SemanticMemoryEntry:
        """把当前观察或感知结果写入语义记忆。"""

        linked_location = None
        if linked_location_id:
            linked_location = self._tagged_locations_by_id.get(linked_location_id)
            if linked_location is None:
                raise GatewayError("未找到地点记忆：%s" % linked_location_id)

        perception_context = self._perception_service.get_latest_perception(camera_id)
        observation_context = self._observation_service.get_latest_observation(camera_id)
        if perception_context is None and observation_context is None:
            raise GatewayError("当前没有可写入记忆的观察或感知结果。")

        if title is None:
            title = (
                perception_context.scene_summary.headline
                if perception_context is not None
                else "观察记录：%s" % observation_context.camera_id
            )
        summary = summary_override or (
            perception_context.scene_summary.headline
            if perception_context is not None
            else observation_context.observation.summary
        )
        collected_tags = list(tags or [])
        if perception_context is not None:
            collected_tags.extend(self._collect_scene_visual_labels(perception_context.scene_summary))
            collected_tags.extend(self._collect_scene_vision_tags(perception_context.scene_summary))
        if linked_location is not None:
            collected_tags.append(linked_location.name)
            collected_tags.extend(linked_location.aliases)
            collected_tags.extend(linked_location.semantic_labels)

        merged_metadata = dict(metadata or {})
        if perception_context is not None:
            merged_metadata.update(self._build_scene_memory_metadata(perception_context))

        return self._create_scene_memory_entry(
            title=title,
            summary=summary,
            tags=collected_tags,
            linked_location=linked_location,
            camera_id=camera_id,
            metadata=merged_metadata,
        )

    def query_location(
        self,
        query: str,
        *,
        similarity_threshold: float = 0.55,
        limit: int = 5,
        payload_filter: Optional[MemoryPayloadFilter] = None,
        max_age_sec: Optional[float] = None,
        near_pose: Optional[Pose] = None,
        max_distance_m: Optional[float] = None,
    ) -> MemoryQueryResult:
        """查询命名地点。"""

        resolved_filter = self._merge_payload_filter(
            payload_filter,
            record_kinds=[MemoryRecordKind.TAGGED_LOCATION],
            max_age_sec=max_age_sec,
            near_pose=near_pose,
            max_distance_m=max_distance_m,
        )
        result = self._query_records(
            query,
            allowed_record_kinds=(MemoryRecordKind.TAGGED_LOCATION,),
            similarity_threshold=similarity_threshold,
            limit=limit,
            payload_filter=resolved_filter,
        )
        self._state_store.write(
            StateNamespace.MEMORY,
            "last_location_query",
            result,
            source="memory_service",
            metadata={"kind": "location_query"},
        )
        return result

    def query_semantic_memory(
        self,
        query: str,
        *,
        similarity_threshold: float = 0.25,
        limit: int = 5,
        payload_filter: Optional[MemoryPayloadFilter] = None,
        max_age_sec: Optional[float] = None,
        near_pose: Optional[Pose] = None,
        max_distance_m: Optional[float] = None,
    ) -> MemoryQueryResult:
        """查询语义记忆。"""

        resolved_filter = self._merge_payload_filter(
            payload_filter,
            record_kinds=[
                MemoryRecordKind.SCENE,
                MemoryRecordKind.NOTE,
                MemoryRecordKind.OBJECT_INSTANCE,
                MemoryRecordKind.OBSERVATION_EVENT,
            ],
            max_age_sec=max_age_sec,
            near_pose=near_pose,
            max_distance_m=max_distance_m,
        )
        result = self._query_records(
            query,
            allowed_record_kinds=(
                MemoryRecordKind.SCENE,
                MemoryRecordKind.NOTE,
                MemoryRecordKind.OBJECT_INSTANCE,
                MemoryRecordKind.OBSERVATION_EVENT,
            ),
            similarity_threshold=similarity_threshold,
            limit=limit,
            payload_filter=resolved_filter,
        )
        self._state_store.write(
            StateNamespace.MEMORY,
            "last_semantic_query",
            result,
            source="memory_service",
            metadata={"kind": "semantic_query"},
        )
        return result

    def resolve_tagged_location(
        self,
        query: str,
        *,
        similarity_threshold: float = 0.55,
    ) -> Optional[TaggedLocation]:
        """返回最优地点命中。"""

        result = self.query_location(query, similarity_threshold=similarity_threshold, limit=1)
        if not result.matches:
            return None
        return result.matches[0].tagged_location

    def resolve_navigation_candidate(
        self,
        query: str,
        *,
        camera_id: Optional[str] = None,
        similarity_threshold: float = 0.25,
        max_age_sec: Optional[float] = None,
    ) -> Optional[MemoryNavigationCandidate]:
        """把自由文本查询解析成可导航的记忆候选。"""

        current_pose = self._get_current_pose()
        map_snapshot = self._get_map_snapshot()
        self._refresh_semantic_map_context(map_snapshot)
        known_labels = self._collect_known_grounding_labels()
        grounding_plan = self._grounding_query_planner.plan(query, known_labels=known_labels)
        payload_filter = self._grounding_query_planner.build_payload_filter(
            grounding_plan,
            map_version_id=map_snapshot.version_id if map_snapshot is not None else None,
        )
        if max_age_sec is not None:
            payload_filter.max_age_sec = max_age_sec
        if current_pose is not None:
            payload_filter.near_pose = current_pose
        result = self._query_records(
            query,
            allowed_record_kinds=(
                MemoryRecordKind.OBJECT_INSTANCE,
                MemoryRecordKind.OBSERVATION_EVENT,
                MemoryRecordKind.SCENE,
                MemoryRecordKind.NOTE,
            ),
            similarity_threshold=similarity_threshold,
            limit=5,
            payload_filter=payload_filter,
        )
        for match in result.matches:
            if match.navigation_candidate is not None:
                return self._to_legacy_navigation_candidate(match.navigation_candidate)
        tagged_location = self.resolve_tagged_location(query)
        if tagged_location is not None:
            spatial_candidate = self._build_navigation_candidate(
                query,
                record_kind=MemoryRecordKind.TAGGED_LOCATION,
                record_id=tagged_location.location_id,
                anchor_id=self._resolve_location_anchor_id(tagged_location),
                inspection_pose=tagged_location.pose,
                anchor_pose=tagged_location.pose,
                target_name=tagged_location.name,
                linked_location_id=tagged_location.location_id,
                map_version_id=tagged_location.map_version_id,
                topo_node_id=tagged_location.topo_node_id,
                semantic_region_id=tagged_location.semantic_region_id,
                artifact_ids=[],
                current_pose=current_pose,
                verification_memory_id=tagged_location.memory_id,
                verification_text=tagged_location.perception_headline or query,
                camera_id=camera_id,
            )
            return self._to_legacy_navigation_candidate(spatial_candidate)
        return None

    def verify_arrival(
        self,
        query: str,
        *,
        navigation_candidate: Optional[Union[MemoryNavigationCandidate, NavigationCandidate]] = None,
        camera_id: Optional[str] = None,
        similarity_threshold: float = 0.55,
    ) -> MemoryArrivalVerification:
        """到点后用当前场景与目标记忆做一次复核。"""

        legacy_candidate = self._coerce_legacy_navigation_candidate(navigation_candidate)
        effective_query = (
            legacy_candidate.verification_query
            if legacy_candidate is not None and legacy_candidate.verification_query
            else query
        )
        perception_context = self._perception_service.get_latest_perception(camera_id)
        if perception_context is None or (utc_now() - perception_context.timestamp) > timedelta(seconds=2):
            perception_context = self._perception_service.describe_current_scene(
                camera_id=camera_id,
                refresh=True,
                requested_by="memory_arrival_verify",
            )
        scene_text = " ".join(
            [
                perception_context.scene_summary.headline,
                " ".join(item.label for item in perception_context.scene_summary.objects),
                " ".join(self._collect_scene_vision_tags(perception_context.scene_summary)),
            ]
        ).strip()
        lexical_score, lexical_reason = self._score_text_candidates(
            self._normalize_text(effective_query),
            [
                scene_text,
                *[item.label for item in perception_context.scene_summary.objects],
            ],
            fallback_reason="当前场景与目标文本存在一定相似度。",
        )
        matched_labels = self._normalize_aliases(
            [item.label for item in perception_context.scene_summary.objects]
            + self._collect_scene_vision_tags(perception_context.scene_summary)
        )
        image_score = 0.0
        target_point = None
        if legacy_candidate is not None:
            target_point = self._vector_store.get_point(legacy_candidate.record_id)
        if self._image_embedder is not None and perception_context.image_artifact is not None:
            current_image_bytes = self._load_image_bytes(perception_context.image_artifact.artifact_id)
            if current_image_bytes is not None:
                current_image_vector = self._image_embedder.embed_image_bytes(current_image_bytes)
                query_image_vector = self._image_embedder.embed_text(effective_query)
                image_score = max(image_score, max(0.0, self._image_embedder.cosine_similarity(query_image_vector, current_image_vector)))
                if target_point is not None and IMAGE_VECTOR_NAME in target_point.vectors:
                    image_score = max(
                        image_score,
                        max(
                            0.0,
                            self._image_embedder.cosine_similarity(
                                current_image_vector,
                                target_point.vectors[IMAGE_VECTOR_NAME].vector,
                            ),
                        ),
                    )
        score = self._clamp_score((0.45 * lexical_score) + (0.55 * image_score if image_score > 0.0 else 0.0) + (0.35 * lexical_score if image_score <= 0.0 else 0.0))
        verified = score >= similarity_threshold or self._normalize_text(effective_query) in {self._normalize_text(item) for item in matched_labels}
        if verified and image_score >= lexical_score + 0.05:
            reason = "到点后图像语义与目标记忆高度一致。"
        elif verified:
            reason = lexical_reason or "到点后目标标签与场景摘要匹配。"
        else:
            reason = "到点后尚未观察到足够强的目标证据。"
        result = MemoryArrivalVerification(
            query=effective_query,
            verified=verified,
            score=score,
            matched_labels=matched_labels,
            matched_memory_id=(
                legacy_candidate.verification_memory_id if legacy_candidate is not None else None
            ),
            reason=reason,
            metadata={
                "lexical_score": round(lexical_score, 6),
                "image_score": round(image_score, 6),
                "camera_id": perception_context.camera_id,
                "requested_query": query,
                "verification_query": effective_query,
            },
        )
        self._state_store.write(
            StateNamespace.MEMORY,
            "last_arrival_verification",
            result,
            source="memory_service",
            metadata={"kind": "arrival_verification"},
        )
        return result

    def get_summary(self) -> MemoryStoreSummary:
        """返回记忆摘要。"""

        last_location = self._tagged_location_history[-1] if self._tagged_location_history else None
        last_memory = self._semantic_history[-1] if self._semantic_history else None
        vector_counts = self._vector_store.count_vectors_by_name()
        return MemoryStoreSummary(
            tagged_location_count=len(self._tagged_locations_by_id),
            semantic_memory_count=len(self._semantic_entries_by_id),
            last_location_id=last_location.location_id if last_location is not None else None,
            last_memory_id=last_memory.memory_id if last_memory is not None else None,
            metadata={
                "store_mode": "multimodal_named_vector",
                "db_path": self._repository.db_path,
                "vector_store_backend": self._vector_store.backend_name,
                "text_embedding_model": self._text_embedder.model_name,
                "text_embedding_dimension": self._text_embedder.dimension,
                "image_embedding_model": self._image_embedder.model_name if self._image_embedder is not None else "disabled",
                "image_embedding_dimension": self._image_embedder.dimension if self._image_embedder is not None else 0,
                "vector_point_count": self._vector_store.count_points(),
                "collection_counts": self._vector_store.count_points_by_collection(),
                "named_vector_counts": vector_counts,
                "semantic_instance_count": len(self._semantic_instances_by_id),
                "observation_event_count": len(self._observation_events_by_id),
            },
        )

    def list_tagged_locations(self, *, limit: Optional[int] = None) -> Tuple[TaggedLocation, ...]:
        """列出地点记忆。"""

        items = list(self._tagged_location_history)
        if limit is not None:
            items = items[-max(0, limit) :]
        return tuple(items)

    def list_semantic_memories(self, *, limit: Optional[int] = None) -> Tuple[SemanticMemoryEntry, ...]:
        """列出语义记忆。"""

        items = list(self._semantic_history)
        if limit is not None:
            items = items[-max(0, limit) :]
        return tuple(items)

    def _create_scene_memory_entry(
        self,
        *,
        title: str,
        summary: Optional[str],
        tags: Union[List[str], Tuple[str, ...], None],
        linked_location: Optional[TaggedLocation],
        camera_id: Optional[str],
        metadata: Optional[Dict[str, object]],
    ) -> SemanticMemoryEntry:
        localization_snapshot = self._get_or_refresh_localization(allow_missing=True)
        map_snapshot = self._get_map_snapshot()
        observation_context = self._observation_service.get_latest_observation(camera_id)
        perception_context = self._perception_service.get_latest_perception(camera_id)
        artifact_ids: List[str] = []
        if observation_context is not None and observation_context.image_artifact is not None:
            artifact_ids.append(observation_context.image_artifact.artifact_id)
        if perception_context is not None and perception_context.image_artifact is not None:
            if perception_context.image_artifact.artifact_id not in artifact_ids:
                artifact_ids.append(perception_context.image_artifact.artifact_id)

        pose = None
        if linked_location is not None:
            pose = linked_location.pose
        elif localization_snapshot is not None:
            pose = localization_snapshot.current_pose
        semantic_region_id = None
        semantic_labels: List[str] = []
        topo_node_id = None
        if linked_location is not None:
            semantic_region_id = linked_location.semantic_region_id
            semantic_labels = list(linked_location.semantic_labels)
            topo_node_id = linked_location.topo_node_id
        elif pose is not None:
            semantic_region_id, semantic_labels = self._resolve_semantic_region(
                pose,
                map_snapshot.semantic_map if map_snapshot is not None else None,
            )
            topo_node_id = self._resolve_topology_node_id(
                pose,
                map_snapshot.metadata if map_snapshot is not None else {},
                metadata,
            )

        merged_metadata = {"camera_id": camera_id or "", **dict(metadata or {})}
        if perception_context is not None:
            merged_metadata.update(self._build_scene_memory_metadata(perception_context))

        entry = SemanticMemoryEntry(
            memory_id=build_memory_id("scene"),
            kind=MemoryRecordKind.SCENE if linked_location is None else MemoryRecordKind.NOTE,
            title=title.strip(),
            summary=(summary or title).strip(),
            tags=self._normalize_tags(tags),
            linked_location_id=linked_location.location_id if linked_location is not None else None,
            pose=pose,
            map_version_id=(
                linked_location.map_version_id
                if linked_location is not None
                else map_snapshot.version_id if map_snapshot is not None else None
            ),
            topo_node_id=topo_node_id,
            semantic_region_id=semantic_region_id,
            semantic_labels=semantic_labels,
            observation_id=(
                observation_context.observation.observation_id if observation_context is not None else None
            ),
            perception_headline=(
                perception_context.scene_summary.headline if perception_context is not None else None
            ),
            artifact_ids=artifact_ids,
            metadata=merged_metadata,
        )
        self._semantic_entries_by_id[entry.memory_id] = entry
        self._semantic_history.append(entry)
        self._write_semantic_memory_state(entry)
        self._repository.upsert_semantic_memory(entry)
        self._upsert_vector_point_for_memory(entry)
        self._ingest_spatial_semantic_observation(
            memory_entry=entry,
            perception_context=perception_context,
            observation_context=observation_context,
            map_snapshot=map_snapshot,
            current_pose=pose,
            camera_id=camera_id,
        )
        self._publish_memory_event(
            "memory.entry_created",
            "语义记忆 %s 已写入。" % entry.title,
            payload={"memory_id": entry.memory_id, "title": entry.title},
        )
        return entry

    def _query_records(
        self,
        query: str,
        *,
        allowed_record_kinds: Tuple[MemoryRecordKind, ...],
        similarity_threshold: float,
        limit: int,
        payload_filter: Optional[MemoryPayloadFilter],
    ) -> MemoryQueryResult:
        normalized_query = self._normalize_text(query)
        if not normalized_query:
            raise GatewayError(
                "查询文本不能为空。",
                error_code="invalid_params",
                http_status=422,
                jsonrpc_code=-32602,
            )
        current_pose = self._get_current_pose()
        query_text_vector = self._text_embedder.embed_text(query)
        query_image_vector = self._image_embedder.embed_text(query) if self._image_embedder is not None else None
        recall_limit = max(limit * 12, 20)
        vector_hits = self._vector_store.query(
            vector_name=TEXT_VECTOR_NAME,
            query_vector=query_text_vector,
            limit=recall_limit,
            payload_filter=payload_filter,
            collection_names=self._collections_for_record_kinds(allowed_record_kinds),
        )
        recall_by_id = {hit.point.point_id: hit.score for hit in vector_hits}
        candidate_ids = set(recall_by_id.keys())
        for record_kind in allowed_record_kinds:
            for candidate_id, lexical_score in self._iter_lexical_candidate_ids(normalized_query, record_kind):
                if lexical_score >= max(0.18, similarity_threshold * 0.5):
                    candidate_ids.add(candidate_id)

        scored_candidates: List[_ScoredCandidate] = []
        for candidate_id in sorted(candidate_ids):
            point = self._vector_store.get_point(candidate_id)
            if point is None:
                continue
            if payload_filter is not None and not self._point_matches_filter(point, payload_filter):
                continue
            candidate = self._score_candidate(
                query=query,
                normalized_query=normalized_query,
                point=point,
                allowed_record_kinds=allowed_record_kinds,
                current_pose=current_pose,
                text_recall_score=recall_by_id.get(candidate_id, 0.0),
                query_image_vector=query_image_vector,
            )
            if candidate is None:
                continue
            if candidate.final_score < similarity_threshold:
                continue
            scored_candidates.append(candidate)

        scored_candidates.sort(key=lambda item: item.final_score, reverse=True)
        matches = [self._build_query_match(query, item, current_pose) for item in scored_candidates[: max(1, limit)]]
        return MemoryQueryResult(
            query=query,
            similarity_threshold=similarity_threshold,
            matches=matches,
            metadata={
                "candidate_count": len(scored_candidates),
                "retrieval_mode": "text_recall_then_filter_then_multimodal_rerank",
                "vector_store_backend": self._vector_store.backend_name,
                "text_embedding_model": self._text_embedder.model_name,
                "text_embedding_dimension": self._text_embedder.dimension,
                "image_embedding_model": self._image_embedder.model_name if self._image_embedder is not None else "disabled",
                "image_embedding_dimension": self._image_embedder.dimension if self._image_embedder is not None else 0,
                "filters": payload_filter.model_dump(mode="json") if payload_filter is not None else {},
            },
        )

    def _score_candidate(
        self,
        *,
        query: str,
        normalized_query: str,
        point: VectorPoint,
        allowed_record_kinds: Tuple[MemoryRecordKind, ...],
        current_pose: Optional[Pose],
        text_recall_score: float,
        query_image_vector: Optional[np.ndarray],
    ) -> Optional[_ScoredCandidate]:
        if point.record_kind not in {item.value for item in allowed_record_kinds}:
            return None
        tagged_location = self._tagged_locations_by_id.get(point.point_id)
        semantic_memory = self._semantic_entries_by_id.get(point.point_id)
        semantic_instance = self._semantic_instances_by_id.get(point.point_id)
        observation_event = self._observation_events_by_id.get(point.point_id)
        if tagged_location is None and semantic_memory is None and semantic_instance is None and observation_event is None:
            return None
        if tagged_location is not None:
            record_kind = MemoryRecordKind.TAGGED_LOCATION
            lexical_score, lexical_reason = self._score_location(normalized_query, tagged_location)
            location_or_memory_pose = tagged_location.pose
            timestamp = tagged_location.timestamp
        elif semantic_instance is not None:
            record_kind = MemoryRecordKind.OBJECT_INSTANCE
            lexical_score, lexical_reason = self._score_semantic_instance(normalized_query, semantic_instance)
            location_or_memory_pose = semantic_instance.pose
            timestamp = semantic_instance.last_seen_ts
        elif observation_event is not None:
            record_kind = MemoryRecordKind.OBSERVATION_EVENT
            lexical_score, lexical_reason = self._score_observation_event(normalized_query, observation_event)
            location_or_memory_pose = observation_event.pose
            timestamp = observation_event.timestamp
        else:
            assert semantic_memory is not None
            record_kind = semantic_memory.kind
            lexical_score, lexical_reason = self._score_semantic_entry(normalized_query, semantic_memory)
            location_or_memory_pose = semantic_memory.pose
            timestamp = semantic_memory.timestamp

        image_score = 0.0
        if (
            query_image_vector is not None
            and IMAGE_VECTOR_NAME in point.vectors
            and self._image_embedder is not None
        ):
            image_score = max(
                0.0,
                self._image_embedder.cosine_similarity(
                    query_image_vector,
                    point.vectors[IMAGE_VECTOR_NAME].vector,
                ),
            )
        spatial_score, distance_m = self._compute_spatial_score(current_pose, location_or_memory_pose)
        recency_score = self._compute_recency_score(timestamp)
        final_score = self._combine_candidate_scores(
            record_kind=record_kind,
            lexical_score=lexical_score,
            text_recall_score=text_recall_score,
            image_score=image_score,
            spatial_score=spatial_score,
            recency_score=recency_score,
        )
        return _ScoredCandidate(
            point=point,
            record_kind=record_kind,
            tagged_location=tagged_location,
            semantic_memory=semantic_memory,
            semantic_instance=semantic_instance,
            observation_event=observation_event,
            text_recall_score=text_recall_score,
            lexical_score=lexical_score,
            lexical_reason=lexical_reason,
            image_score=image_score,
            spatial_score=spatial_score,
            recency_score=recency_score,
            distance_m=distance_m,
            final_score=final_score,
        )

    def _build_query_match(
        self,
        query: str,
        candidate: _ScoredCandidate,
        current_pose: Optional[Pose],
    ) -> MemoryQueryMatch:
        navigation_candidate = None
        if candidate.tagged_location is not None:
            navigation_candidate = self._build_navigation_candidate(
                query,
                record_kind=MemoryRecordKind.TAGGED_LOCATION,
                record_id=candidate.tagged_location.location_id,
                anchor_id=self._resolve_location_anchor_id(candidate.tagged_location),
                inspection_pose=candidate.tagged_location.pose,
                anchor_pose=candidate.tagged_location.pose,
                target_name=candidate.tagged_location.name,
                linked_location_id=candidate.tagged_location.location_id,
                map_version_id=candidate.tagged_location.map_version_id,
                topo_node_id=candidate.tagged_location.topo_node_id,
                semantic_region_id=candidate.tagged_location.semantic_region_id,
                artifact_ids=[],
                current_pose=current_pose,
                verification_memory_id=candidate.tagged_location.memory_id,
                verification_text=candidate.tagged_location.perception_headline or query,
                camera_id=None,
            )
        elif candidate.semantic_memory is not None and candidate.semantic_memory.pose is not None:
            navigation_candidate = self._build_navigation_candidate(
                query,
                record_kind=candidate.semantic_memory.kind,
                record_id=candidate.semantic_memory.memory_id,
                anchor_id=self._resolve_memory_anchor_id(candidate.semantic_memory),
                inspection_pose=self._resolve_inspection_pose(
                    anchor_id=self._resolve_memory_anchor_id(candidate.semantic_memory),
                    fallback_pose=candidate.semantic_memory.pose,
                ),
                anchor_pose=candidate.semantic_memory.pose,
                target_name=candidate.semantic_memory.title,
                linked_location_id=candidate.semantic_memory.linked_location_id,
                map_version_id=candidate.semantic_memory.map_version_id,
                topo_node_id=candidate.semantic_memory.topo_node_id,
                semantic_region_id=candidate.semantic_memory.semantic_region_id,
                artifact_ids=candidate.semantic_memory.artifact_ids,
                current_pose=current_pose,
                verification_memory_id=candidate.semantic_memory.memory_id,
                verification_text=(
                    candidate.semantic_memory.perception_headline
                    or candidate.semantic_memory.summary
                    or query
                ),
                camera_id=str(candidate.point.payload.get("camera_id") or "") or None,
            )
        elif candidate.semantic_instance is not None:
            navigation_candidate = self._build_navigation_candidate(
                query,
                record_kind=MemoryRecordKind.OBJECT_INSTANCE,
                record_id=candidate.semantic_instance.instance_id,
                anchor_id=candidate.semantic_instance.anchor_id,
                inspection_pose=self._resolve_inspection_pose(
                    anchor_id=candidate.semantic_instance.anchor_id,
                    fallback_pose=candidate.semantic_instance.pose,
                    override_poses=candidate.semantic_instance.inspection_poses,
                ),
                anchor_pose=candidate.semantic_instance.pose,
                target_name=candidate.semantic_instance.display_name or candidate.semantic_instance.label,
                linked_location_id=None,
                map_version_id=candidate.semantic_instance.map_version_id,
                topo_node_id=candidate.semantic_instance.topo_node_id,
                semantic_region_id=candidate.semantic_instance.semantic_region_id,
                artifact_ids=candidate.semantic_instance.artifact_ids,
                current_pose=current_pose,
                verification_memory_id=None,
                verification_text=" ".join(
                    [
                        candidate.semantic_instance.label,
                        " ".join(candidate.semantic_instance.semantic_labels),
                        " ".join(candidate.semantic_instance.vision_tags),
                    ]
                ).strip()
                or query,
                camera_id=str(candidate.semantic_instance.metadata.get("last_camera_id", "") or "") or None,
            )
        elif candidate.observation_event is not None and candidate.observation_event.pose is not None:
            navigation_candidate = self._build_navigation_candidate(
                query,
                record_kind=MemoryRecordKind.OBSERVATION_EVENT,
                record_id=candidate.observation_event.event_id,
                anchor_id=candidate.observation_event.anchor_id or self._resolve_anchor_id_for_pose(
                    candidate.observation_event.pose,
                    semantic_region_id=candidate.observation_event.semantic_region_id,
                    topo_node_id=candidate.observation_event.topo_node_id,
                ),
                inspection_pose=self._resolve_inspection_pose(
                    anchor_id=candidate.observation_event.anchor_id,
                    fallback_pose=candidate.observation_event.pose,
                ),
                anchor_pose=candidate.observation_event.pose,
                target_name=candidate.observation_event.title,
                linked_location_id=None,
                map_version_id=candidate.observation_event.map_version_id,
                topo_node_id=candidate.observation_event.topo_node_id,
                semantic_region_id=candidate.observation_event.semantic_region_id,
                artifact_ids=candidate.observation_event.artifact_ids,
                current_pose=current_pose,
                verification_memory_id=candidate.observation_event.source_memory_id,
                verification_text=candidate.observation_event.summary or query,
                camera_id=candidate.observation_event.camera_id,
            )

        reason = self._build_match_reason(candidate)
        return MemoryQueryMatch(
            record_kind=candidate.record_kind,
            record_id=candidate.point.point_id,
            score=self._clamp_score(candidate.final_score),
            reason=reason,
            tagged_location=candidate.tagged_location,
            semantic_memory=candidate.semantic_memory,
            semantic_instance=candidate.semantic_instance,
            observation_event=candidate.observation_event,
            navigation_candidate=navigation_candidate,
            metadata={
                "retrieval_mode": "text_recall_then_filter_then_multimodal_rerank",
                "text_recall_score": round(candidate.text_recall_score, 6),
                "lexical_score": round(candidate.lexical_score, 6),
                "image_score": round(candidate.image_score, 6),
                "spatial_score": round(candidate.spatial_score, 6),
                "recency_score": round(candidate.recency_score, 6),
                "type_prior": round(self._type_prior(candidate.record_kind), 6),
                "distance_m": round(candidate.distance_m, 4) if candidate.distance_m is not None else None,
                "vector_names": sorted(candidate.point.vectors.keys()),
            },
        )

    def _build_navigation_candidate(
        self,
        query: str,
        *,
        record_kind: MemoryRecordKind,
        record_id: str,
        anchor_id: Optional[str],
        inspection_pose: Pose,
        anchor_pose: Optional[Pose],
        target_name: str,
        linked_location_id: Optional[str],
        map_version_id: Optional[str],
        topo_node_id: Optional[str],
        semantic_region_id: Optional[str],
        artifact_ids: List[str],
        current_pose: Optional[Pose],
        verification_memory_id: Optional[str],
        verification_text: Optional[str],
        camera_id: Optional[str],
    ) -> NavigationCandidate:
        _, distance_m = self._compute_spatial_score(current_pose, inspection_pose)
        verification_artifact_id = artifact_ids[0] if artifact_ids else None
        return NavigationCandidate(
            candidate_id=self._build_navigation_candidate_id(record_id),
            anchor_id=anchor_id or self._fallback_anchor_id(record_id),
            source_collection=self._collection_for_record_kind(record_kind),
            record_id=record_id,
            target_name=target_name,
            inspection_pose=inspection_pose,
            anchor_pose=anchor_pose,
            map_version_id=map_version_id,
            topo_node_id=topo_node_id,
            semantic_region_id=semantic_region_id,
            distance_m=distance_m,
            score=1.0,
            verification_query=verification_text or query,
            verification_artifact_id=verification_artifact_id,
            metadata={
                "camera_id": camera_id or "",
                "record_kind": record_kind.value,
                "linked_location_id": linked_location_id or "",
                "verification_memory_id": verification_memory_id or "",
            },
        )

    def _load_persisted_memory(self) -> None:
        persisted_locations = self._repository.load_tagged_locations()
        persisted_memories = self._repository.load_semantic_memories()
        persisted_instances = self._repository.load_semantic_instances()
        persisted_events = self._repository.load_observation_events()

        for location in persisted_locations:
            self._tagged_locations_by_id[location.location_id] = location
            self._tagged_location_history.append(location)
            self._write_tagged_location_state(location)
            self._upsert_vector_point_for_location(location)

        for entry in persisted_memories:
            self._semantic_entries_by_id[entry.memory_id] = entry
            self._semantic_history.append(entry)
            self._write_semantic_memory_state(entry)
            self._upsert_vector_point_for_memory(entry)

        for instance in persisted_instances:
            self._semantic_instances_by_id[instance.instance_id] = instance
            self._semantic_instance_history.append(instance)
            self._write_semantic_instance_state(instance)
            self._upsert_vector_point_for_instance(instance)

        for event in persisted_events:
            self._observation_events_by_id[event.event_id] = event
            self._observation_event_history.append(event)
            self._write_observation_event_state(event)
            self._upsert_vector_point_for_observation_event(event)

        for location in persisted_locations:
            if location.memory_id:
                self._upsert_vector_point_for_location(location)

    def _upsert_vector_point_for_location(self, location: TaggedLocation) -> None:
        payload = self._build_location_payload(location)
        vectors = {
            TEXT_VECTOR_NAME: NamedVector(
                vector_name=TEXT_VECTOR_NAME,
                model_name=self._text_embedder.model_name,
                dimension=self._text_embedder.dimension,
                vector=self._text_embedder.embed_text(payload["retrieval_text"]),
            )
        }
        image_bytes = self._load_primary_image_bytes([])
        if location.memory_id is not None:
            linked_memory = self._semantic_entries_by_id.get(location.memory_id)
            if linked_memory is not None:
                image_bytes = self._load_primary_image_bytes(linked_memory.artifact_ids) or image_bytes
        if image_bytes is not None and self._image_embedder is not None:
            vectors[IMAGE_VECTOR_NAME] = NamedVector(
                vector_name=IMAGE_VECTOR_NAME,
                model_name=self._image_embedder.model_name,
                dimension=self._image_embedder.dimension,
                vector=self._image_embedder.embed_image_bytes(image_bytes),
            )
        self._vector_store.upsert_point(
            point_id=location.location_id,
            collection_name=PLACE_NODES_COLLECTION,
            record_kind=MemoryRecordKind.TAGGED_LOCATION.value,
            payload=payload,
            vectors=vectors,
            updated_at=location.timestamp.isoformat(),
        )

    def _upsert_vector_point_for_memory(self, entry: SemanticMemoryEntry) -> None:
        payload = self._build_memory_payload(entry)
        vectors = {
            TEXT_VECTOR_NAME: NamedVector(
                vector_name=TEXT_VECTOR_NAME,
                model_name=self._text_embedder.model_name,
                dimension=self._text_embedder.dimension,
                vector=self._text_embedder.embed_text(payload["retrieval_text"]),
            )
        }
        image_bytes = self._load_primary_image_bytes(entry.artifact_ids)
        if image_bytes is not None and self._image_embedder is not None:
            vectors[IMAGE_VECTOR_NAME] = NamedVector(
                vector_name=IMAGE_VECTOR_NAME,
                model_name=self._image_embedder.model_name,
                dimension=self._image_embedder.dimension,
                vector=self._image_embedder.embed_image_bytes(image_bytes),
            )
        self._vector_store.upsert_point(
            point_id=entry.memory_id,
            collection_name=EPISODIC_OBSERVATIONS_COLLECTION,
            record_kind=entry.kind.value,
            payload=payload,
            vectors=vectors,
            updated_at=entry.timestamp.isoformat(),
        )

    def _upsert_vector_point_for_instance(self, instance: SemanticInstance) -> None:
        payload = self._build_instance_payload(instance)
        vectors = {
            TEXT_VECTOR_NAME: NamedVector(
                vector_name=TEXT_VECTOR_NAME,
                model_name=self._text_embedder.model_name,
                dimension=self._text_embedder.dimension,
                vector=self._text_embedder.embed_text(payload["retrieval_text"]),
            )
        }
        image_bytes = self._load_primary_image_bytes(instance.artifact_ids)
        if image_bytes is not None and self._image_embedder is not None:
            vectors[IMAGE_VECTOR_NAME] = NamedVector(
                vector_name=IMAGE_VECTOR_NAME,
                model_name=self._image_embedder.model_name,
                dimension=self._image_embedder.dimension,
                vector=self._image_embedder.embed_image_bytes(image_bytes),
            )
        self._vector_store.upsert_point(
            point_id=instance.instance_id,
            collection_name=OBJECT_INSTANCES_COLLECTION,
            record_kind=MemoryRecordKind.OBJECT_INSTANCE.value,
            payload=payload,
            vectors=vectors,
            updated_at=instance.last_seen_ts.isoformat(),
        )

    def _upsert_vector_point_for_observation_event(self, event: ObservationEvent) -> None:
        payload = self._build_observation_event_payload(event)
        vectors = {
            TEXT_VECTOR_NAME: NamedVector(
                vector_name=TEXT_VECTOR_NAME,
                model_name=self._text_embedder.model_name,
                dimension=self._text_embedder.dimension,
                vector=self._text_embedder.embed_text(payload["retrieval_text"]),
            )
        }
        image_bytes = self._load_primary_image_bytes(event.artifact_ids)
        if image_bytes is not None and self._image_embedder is not None:
            vectors[IMAGE_VECTOR_NAME] = NamedVector(
                vector_name=IMAGE_VECTOR_NAME,
                model_name=self._image_embedder.model_name,
                dimension=self._image_embedder.dimension,
                vector=self._image_embedder.embed_image_bytes(image_bytes),
            )
        self._vector_store.upsert_point(
            point_id=event.event_id,
            collection_name=EPISODIC_OBSERVATIONS_COLLECTION,
            record_kind=MemoryRecordKind.OBSERVATION_EVENT.value,
            payload=payload,
            vectors=vectors,
            updated_at=event.timestamp.isoformat(),
        )

    def _build_location_payload(self, location: TaggedLocation) -> Dict[str, object]:
        description = str(location.metadata.get("description", "")).strip()
        visual_labels = self._normalize_tags(location.metadata.get("visual_labels"))
        vision_tags = self._normalize_tags(location.metadata.get("vision_tags"))
        retrieval_text = self._build_location_search_text(location)
        return {
            "record_id": location.location_id,
            "record_kind": MemoryRecordKind.TAGGED_LOCATION.value,
            "name": location.name,
            "normalized_name": location.normalized_name,
            "aliases": list(location.aliases),
            "linked_location_id": location.location_id,
            "map_version_id": location.map_version_id,
            "localization_source_name": location.localization_source_name,
            "topo_node_id": location.topo_node_id,
            "semantic_region_id": location.semantic_region_id,
            "anchor_id": self._resolve_location_anchor_id(location),
            "semantic_labels": list(location.semantic_labels),
            "observation_id": location.observation_id,
            "perception_headline": location.perception_headline or "",
            "camera_id": "",
            "memory_id": location.memory_id,
            "timestamp": location.timestamp.isoformat(),
            "pose_x": float(location.pose.position.x),
            "pose_y": float(location.pose.position.y),
            "pose_z": float(location.pose.position.z),
            "description": description,
            "visual_labels": visual_labels,
            "vision_tags": vision_tags,
            "verification_text": location.perception_headline or description or location.name,
            "retrieval_text": retrieval_text,
        }

    def _build_memory_payload(self, entry: SemanticMemoryEntry) -> Dict[str, object]:
        linked_location_name = ""
        if entry.linked_location_id:
            linked_location = self._tagged_locations_by_id.get(entry.linked_location_id)
            if linked_location is not None:
                linked_location_name = linked_location.name
        camera_id = str(entry.metadata.get("camera_id", "") or "")
        visual_labels = self._normalize_tags(entry.metadata.get("visual_labels"))
        vision_tags = self._normalize_tags(entry.metadata.get("vision_tags"))
        pose_x = float(entry.pose.position.x) if entry.pose is not None else None
        pose_y = float(entry.pose.position.y) if entry.pose is not None else None
        pose_z = float(entry.pose.position.z) if entry.pose is not None else None
        retrieval_text = self._build_semantic_search_text(entry)
        return {
            "record_id": entry.memory_id,
            "record_kind": entry.kind.value,
            "title": entry.title,
            "summary": entry.summary,
            "tags": list(entry.tags),
            "linked_location_id": entry.linked_location_id,
            "linked_location_name": linked_location_name,
            "map_version_id": entry.map_version_id,
            "topo_node_id": entry.topo_node_id,
            "semantic_region_id": entry.semantic_region_id,
            "anchor_id": self._resolve_memory_anchor_id(entry),
            "semantic_labels": list(entry.semantic_labels),
            "observation_id": entry.observation_id,
            "perception_headline": entry.perception_headline or "",
            "artifact_ids": list(entry.artifact_ids),
            "camera_id": camera_id,
            "timestamp": entry.timestamp.isoformat(),
            "pose_x": pose_x,
            "pose_y": pose_y,
            "pose_z": pose_z,
            "visual_labels": visual_labels,
            "vision_tags": vision_tags,
            "verification_text": entry.perception_headline or entry.summary or entry.title,
            "retrieval_text": retrieval_text,
        }

    def _build_instance_payload(self, instance: SemanticInstance) -> Dict[str, object]:
        retrieval_text = self._build_instance_search_text(instance)
        return {
            "record_id": instance.instance_id,
            "record_kind": MemoryRecordKind.OBJECT_INSTANCE.value,
            "anchor_id": instance.anchor_id,
            "name": instance.display_name or instance.label,
            "label": instance.label,
            "instance_type": instance.instance_type,
            "movability": instance.movability.value,
            "map_version_id": instance.map_version_id,
            "topo_node_id": instance.topo_node_id,
            "semantic_region_id": instance.semantic_region_id,
            "semantic_labels": list(instance.semantic_labels),
            "visual_labels": list(instance.visual_labels),
            "vision_tags": list(instance.vision_tags),
            "artifact_ids": list(instance.artifact_ids),
            "timestamp": instance.first_seen_ts.isoformat(),
            "last_seen_ts": instance.last_seen_ts.isoformat(),
            "pose_x": float(instance.pose.position.x),
            "pose_y": float(instance.pose.position.y),
            "pose_z": float(instance.pose.position.z),
            "retrieval_text": retrieval_text,
            "verification_text": " ".join([instance.label, *instance.semantic_labels, *instance.vision_tags]).strip(),
        }

    def _build_observation_event_payload(self, event: ObservationEvent) -> Dict[str, object]:
        retrieval_text = self._build_observation_event_search_text(event)
        pose_x = float(event.pose.position.x) if event.pose is not None else None
        pose_y = float(event.pose.position.y) if event.pose is not None else None
        pose_z = float(event.pose.position.z) if event.pose is not None else None
        return {
            "record_id": event.event_id,
            "record_kind": MemoryRecordKind.OBSERVATION_EVENT.value,
            "anchor_id": event.anchor_id,
            "title": event.title,
            "summary": event.summary,
            "camera_id": event.camera_id or "",
            "map_version_id": event.map_version_id,
            "topo_node_id": event.topo_node_id,
            "semantic_region_id": event.semantic_region_id,
            "semantic_labels": list(event.semantic_labels),
            "visual_labels": list(event.visual_labels),
            "vision_tags": list(event.vision_tags),
            "artifact_ids": list(event.artifact_ids),
            "linked_instance_ids": list(event.linked_instance_ids),
            "timestamp": event.timestamp.isoformat(),
            "pose_x": pose_x,
            "pose_y": pose_y,
            "pose_z": pose_z,
            "retrieval_text": retrieval_text,
            "verification_text": event.summary or event.title,
        }

    def _build_location_search_text(self, location: TaggedLocation) -> str:
        segments = [
            location.name,
            location.normalized_name,
            " ".join(location.aliases),
            location.perception_headline or "",
            str(location.metadata.get("description", "")),
            " ".join(location.semantic_labels),
            " ".join(self._normalize_tags(location.metadata.get("visual_labels"))),
            " ".join(self._normalize_tags(location.metadata.get("vision_tags"))),
            location.semantic_region_id or "",
            location.topo_node_id or "",
        ]
        return " ".join(segment for segment in segments if segment).strip()

    def _build_semantic_search_text(self, entry: SemanticMemoryEntry) -> str:
        linked_location_name = ""
        if entry.linked_location_id:
            linked_location = self._tagged_locations_by_id.get(entry.linked_location_id)
            if linked_location is not None:
                linked_location_name = linked_location.name
        segments = [
            entry.title,
            entry.summary,
            " ".join(entry.tags),
            entry.perception_headline or "",
            linked_location_name,
            " ".join(entry.semantic_labels),
            " ".join(self._normalize_tags(entry.metadata.get("visual_labels"))),
            " ".join(self._normalize_tags(entry.metadata.get("vision_tags"))),
            entry.semantic_region_id or "",
            entry.topo_node_id or "",
        ]
        return " ".join(segment for segment in segments if segment).strip()

    def _build_instance_search_text(self, instance: SemanticInstance) -> str:
        segments = [
            instance.label,
            instance.display_name or "",
            " ".join(instance.semantic_labels),
            " ".join(instance.visual_labels),
            " ".join(instance.vision_tags),
            instance.semantic_region_id or "",
            instance.topo_node_id or "",
            " ".join("%s:%s" % (key, value) for key, value in sorted(instance.attributes.items())),
        ]
        return " ".join(segment for segment in segments if segment).strip()

    def _build_observation_event_search_text(self, event: ObservationEvent) -> str:
        segments = [
            event.title,
            event.summary,
            " ".join(event.semantic_labels),
            " ".join(event.visual_labels),
            " ".join(event.vision_tags),
            event.semantic_region_id or "",
            event.topo_node_id or "",
        ]
        return " ".join(segment for segment in segments if segment).strip()

    def _iter_lexical_candidate_ids(
        self,
        normalized_query: str,
        record_kind: MemoryRecordKind,
    ) -> Iterable[Tuple[str, float]]:
        if record_kind == MemoryRecordKind.TAGGED_LOCATION:
            for location in self._tagged_locations_by_id.values():
                lexical_score, _ = self._score_location(normalized_query, location)
                yield location.location_id, lexical_score
            return
        if record_kind == MemoryRecordKind.OBJECT_INSTANCE:
            for instance in self._semantic_instances_by_id.values():
                lexical_score, _ = self._score_semantic_instance(normalized_query, instance)
                yield instance.instance_id, lexical_score
            return
        if record_kind == MemoryRecordKind.OBSERVATION_EVENT:
            for event in self._observation_events_by_id.values():
                lexical_score, _ = self._score_observation_event(normalized_query, event)
                yield event.event_id, lexical_score
            return
        for entry in self._semantic_entries_by_id.values():
            if entry.kind != record_kind:
                continue
            lexical_score, _ = self._score_semantic_entry(normalized_query, entry)
            yield entry.memory_id, lexical_score

    def _score_location(self, normalized_query: str, location: TaggedLocation) -> Tuple[float, str]:
        exact_candidates = [
            location.name,
            location.normalized_name,
            *location.aliases,
            *location.semantic_labels,
        ]
        if normalized_query in {self._normalize_text(item) for item in exact_candidates if item}:
            return 1.0, "命中地点名称、别名或区域语义。"
        texts = [
            location.name,
            location.normalized_name,
            *location.aliases,
            location.perception_headline or "",
            str(location.metadata.get("description", "")),
            *location.semantic_labels,
            *self._normalize_tags(location.metadata.get("visual_labels")),
            *self._normalize_tags(location.metadata.get("vision_tags")),
            location.semantic_region_id or "",
            location.topo_node_id or "",
        ]
        return self._score_text_candidates(normalized_query, texts, fallback_reason="与地点描述或空间标签相似。")

    def _score_semantic_entry(self, normalized_query: str, entry: SemanticMemoryEntry) -> Tuple[float, str]:
        texts = [
            entry.title,
            entry.summary,
            *entry.tags,
            entry.perception_headline or "",
            *entry.semantic_labels,
            *self._normalize_tags(entry.metadata.get("visual_labels")),
            *self._normalize_tags(entry.metadata.get("vision_tags")),
            entry.semantic_region_id or "",
            entry.topo_node_id or "",
        ]
        return self._score_text_candidates(normalized_query, texts, fallback_reason="与语义记忆内容相似。")

    def _score_semantic_instance(self, normalized_query: str, instance: SemanticInstance) -> Tuple[float, str]:
        texts = [
            instance.label,
            instance.display_name or "",
            *instance.semantic_labels,
            *instance.visual_labels,
            *instance.vision_tags,
            instance.semantic_region_id or "",
            instance.topo_node_id or "",
            *["%s:%s" % (key, value) for key, value in sorted(instance.attributes.items())],
        ]
        return self._score_text_candidates(normalized_query, texts, fallback_reason="与空间语义实例相似。")

    def _score_observation_event(self, normalized_query: str, event: ObservationEvent) -> Tuple[float, str]:
        texts = [
            event.title,
            event.summary,
            *event.semantic_labels,
            *event.visual_labels,
            *event.vision_tags,
            event.semantic_region_id or "",
            event.topo_node_id or "",
        ]
        return self._score_text_candidates(normalized_query, texts, fallback_reason="与观察事件内容相似。")

    def _collect_scene_visual_labels(self, scene_summary) -> List[str]:
        return self._normalize_tags([item.label for item in scene_summary.objects])

    def _collect_scene_vision_tags(self, scene_summary) -> List[str]:
        raw_tags: List[str] = []
        raw_tags.extend(scene_summary.metadata.get("semantic_tags") or [])
        raw_tags.extend(scene_summary.metadata.get("semantic_relations") or [])
        raw_tags.extend(scene_summary.metadata.get("visual_labels") or [])
        for item in scene_summary.objects:
            raw_tags.append(item.label)
            raw_tags.extend(self._flatten_metadata_strings(item.attributes))
        return self._normalize_tags(raw_tags)

    def _build_scene_memory_metadata(self, perception_context) -> Dict[str, object]:
        scene_summary = perception_context.scene_summary
        return {
            "visual_labels": self._collect_scene_visual_labels(scene_summary),
            "vision_tags": self._collect_scene_vision_tags(scene_summary),
            "semantic_relations": self._normalize_aliases(scene_summary.metadata.get("semantic_relations")),
            "scene_backend_chain": self._normalize_aliases(scene_summary.metadata.get("scene_backend_chain")),
            "cloud_vision_model": str(scene_summary.metadata.get("cloud_vision_model", "") or ""),
            "detector_backend": perception_context.detector_backend,
            "tracker_backend": perception_context.tracker_backend,
        }

    def _flatten_metadata_strings(self, value: object) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            items: List[str] = []
            for inner_value in value.values():
                items.extend(self._flatten_metadata_strings(inner_value))
            return items
        if isinstance(value, (list, tuple, set)):
            items = []
            for inner_value in value:
                items.extend(self._flatten_metadata_strings(inner_value))
            return items
        return [str(value)]

    def _score_text_candidates(
        self,
        normalized_query: str,
        texts: List[str],
        *,
        fallback_reason: str,
    ) -> Tuple[float, str]:
        if not normalized_query:
            return 0.0, fallback_reason
        query_tokens = self._tokenize(normalized_query)
        best_score = 0.0
        for text in texts:
            normalized_text = self._normalize_text(text)
            if not normalized_text:
                continue
            if normalized_query == normalized_text:
                return 0.99, "文本完全匹配。"
            text_tokens = self._tokenize(normalized_text)
            overlap_score = self._token_overlap_score(query_tokens, text_tokens)
            fuzzy_score = SequenceMatcher(None, normalized_query, normalized_text).ratio()
            score = max(fuzzy_score, overlap_score, (0.6 * fuzzy_score) + (0.4 * overlap_score))
            best_score = max(best_score, score)
        return best_score, fallback_reason

    def _combine_candidate_scores(
        self,
        *,
        record_kind: MemoryRecordKind,
        lexical_score: float,
        text_recall_score: float,
        image_score: float,
        spatial_score: float,
        recency_score: float,
    ) -> float:
        if record_kind == MemoryRecordKind.TAGGED_LOCATION:
            if lexical_score >= 0.99:
                return 1.0
            score = (
                (0.45 * lexical_score)
                + (0.35 * text_recall_score)
                + (0.10 * image_score)
                + (0.05 * spatial_score)
                + (0.05 * recency_score)
            )
            base_score = self._clamp_score(max(score, lexical_score * 0.95, text_recall_score * 0.9))
            return self._apply_type_prior(base_score, record_kind)
        if record_kind == MemoryRecordKind.OBJECT_INSTANCE:
            score = (
                (0.12 * lexical_score)
                + (0.38 * text_recall_score)
                + (0.25 * image_score)
                + (0.15 * spatial_score)
                + (0.10 * recency_score)
            )
            base_score = self._clamp_score(max(score, (0.92 * text_recall_score) + 0.03))
            return self._apply_type_prior(base_score, record_kind)
        if record_kind == MemoryRecordKind.OBSERVATION_EVENT:
            score = (
                (0.18 * lexical_score)
                + (0.42 * text_recall_score)
                + (0.20 * image_score)
                + (0.12 * spatial_score)
                + (0.08 * recency_score)
            )
            base_score = self._clamp_score(max(score, (0.96 * text_recall_score) + 0.02))
            return self._apply_type_prior(base_score, record_kind)
        score = (
            (0.20 * lexical_score)
            + (0.45 * text_recall_score)
            + (0.20 * image_score)
            + (0.10 * spatial_score)
            + (0.05 * recency_score)
        )
        base_score = self._clamp_score(max(score, text_recall_score + 0.03, image_score * 0.9))
        return self._apply_type_prior(base_score, record_kind)

    def _apply_type_prior(self, base_score: float, record_kind: MemoryRecordKind) -> float:
        type_prior = self._type_prior(record_kind)
        return self._clamp_score(base_score * type_prior)

    def _type_prior(self, record_kind: MemoryRecordKind) -> float:
        if record_kind == MemoryRecordKind.OBJECT_INSTANCE:
            return 1.0
        if record_kind == MemoryRecordKind.OBSERVATION_EVENT:
            return 0.85
        if record_kind in {MemoryRecordKind.SCENE, MemoryRecordKind.NOTE}:
            return 0.75
        return 1.0

    def _build_match_reason(self, candidate: _ScoredCandidate) -> str:
        if candidate.lexical_score >= 0.99:
            return candidate.lexical_reason
        if candidate.image_score >= max(candidate.text_recall_score, candidate.lexical_score) + 0.05:
            return "CLIP 图像重排增强了该候选。"
        if candidate.text_recall_score >= candidate.lexical_score + 0.05:
            return "BGE 文本召回命中，随后经空间与时间条件确认。"
        if candidate.spatial_score >= 0.8:
            return "文本相似且位于当前附近区域。"
        return candidate.lexical_reason or "多模态记忆命中。"

    def _compute_spatial_score(
        self,
        current_pose: Optional[Pose],
        candidate_pose: Optional[Pose],
    ) -> Tuple[float, Optional[float]]:
        if current_pose is None or candidate_pose is None:
            return 0.0, None
        delta_x = float(candidate_pose.position.x) - float(current_pose.position.x)
        delta_y = float(candidate_pose.position.y) - float(current_pose.position.y)
        delta_z = float(candidate_pose.position.z) - float(current_pose.position.z)
        distance_m = float(np.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z))
        score = 1.0 / (1.0 + distance_m)
        return self._clamp_score(score), distance_m

    def _compute_recency_score(self, timestamp: datetime) -> float:
        age_sec = max(0.0, (utc_now() - timestamp).total_seconds())
        if age_sec <= 300.0:
            return 1.0
        if age_sec <= 3600.0:
            return 0.85
        if age_sec <= 86400.0:
            return 0.65
        if age_sec <= 604800.0:
            return 0.45
        return 0.25

    def _point_matches_filter(self, point: VectorPoint, payload_filter: MemoryPayloadFilter) -> bool:
        shadow_store = self._vector_store
        return shadow_store._payload_matches(point.payload, payload_filter)  # type: ignore[attr-defined]

    def _merge_payload_filter(
        self,
        payload_filter: Optional[MemoryPayloadFilter],
        *,
        record_kinds: List[MemoryRecordKind],
        max_age_sec: Optional[float],
        near_pose: Optional[Pose],
        max_distance_m: Optional[float],
    ) -> MemoryPayloadFilter:
        if payload_filter is None:
            return MemoryPayloadFilter(
                record_kinds=list(record_kinds),
                max_age_sec=max_age_sec,
                near_pose=near_pose,
                max_distance_m=max_distance_m,
            )
        merged = payload_filter.model_copy(deep=True)
        merged.record_kinds = list(record_kinds)
        if max_age_sec is not None:
            merged.max_age_sec = max_age_sec
        if near_pose is not None:
            merged.near_pose = near_pose
        if max_distance_m is not None:
            merged.max_distance_m = max_distance_m
        return merged

    def _refresh_semantic_map_context(self, map_snapshot) -> SemanticMapBuildResult:
        if map_snapshot is None:
            self._latest_semantic_map_context = self._semantic_map_builder.build(None)
            return self._latest_semantic_map_context
        if self._latest_semantic_map_context.map_version_id == map_snapshot.version_id:
            return self._latest_semantic_map_context
        self._latest_semantic_map_context = self._semantic_map_builder.build(map_snapshot)
        self._state_store.write(
            StateNamespace.MEMORY,
            "semantic_map_context",
            self._latest_semantic_map_context,
            source="memory_service",
            metadata={"kind": "semantic_map_context", "map_version_id": map_snapshot.version_id},
        )
        return self._latest_semantic_map_context

    def _ingest_spatial_semantic_observation(
        self,
        *,
        memory_entry: SemanticMemoryEntry,
        perception_context,
        observation_context,
        map_snapshot,
        current_pose: Optional[Pose],
        camera_id: Optional[str],
    ) -> None:
        if perception_context is None or current_pose is None:
            return
        map_context = self._refresh_semantic_map_context(map_snapshot)
        projected = self._vision_projection_service.project(
            perception_context,
            current_pose=current_pose,
            map_context=map_context,
        )
        if not projected:
            return
        updated_instances, outcomes = self._instance_association_service.associate(
            projected,
            existing_instances=self._semantic_instances_by_id,
            map_version_id=memory_entry.map_version_id,
            occupancy_grid=map_snapshot.occupancy_grid if map_snapshot is not None else None,
            cost_map=map_snapshot.cost_map if map_snapshot is not None else None,
            now=memory_entry.timestamp,
        )
        self._semantic_instances_by_id = dict(updated_instances)
        linked_instance_ids: List[str] = []
        for outcome in outcomes:
            instance = outcome.instance
            linked_instance_ids.append(instance.instance_id)
            self._semantic_instance_history.append(instance)
            self._write_semantic_instance_state(instance)
            self._repository.upsert_semantic_instance(instance)
            self._upsert_vector_point_for_instance(instance)

        observation_event = ObservationEvent(
            event_id=build_observation_event_id("perception", timestamp=memory_entry.timestamp),
            title=memory_entry.title,
            summary=memory_entry.summary,
            camera_id=camera_id or str(memory_entry.metadata.get("camera_id", "") or "") or None,
            pose=current_pose,
            map_version_id=memory_entry.map_version_id,
            topo_node_id=memory_entry.topo_node_id,
            semantic_region_id=memory_entry.semantic_region_id,
            anchor_id=self._resolve_memory_anchor_id(memory_entry),
            source_observation_id=(
                observation_context.observation.observation_id if observation_context is not None else memory_entry.observation_id
            ),
            source_memory_id=memory_entry.memory_id,
            linked_instance_ids=self._normalize_aliases(linked_instance_ids),
            artifact_ids=list(memory_entry.artifact_ids),
            semantic_labels=self._normalize_tags(
                [
                    *memory_entry.semantic_labels,
                    *[item for projection in projected for item in projection.semantic_labels],
                ]
            ),
            visual_labels=self._normalize_tags([item.label for item in perception_context.scene_summary.objects]),
            vision_tags=self._normalize_tags(self._collect_scene_vision_tags(perception_context.scene_summary)),
            metadata={
                "camera_id": camera_id or "",
                "projection_count": len(projected),
                "linked_memory_id": memory_entry.memory_id,
            },
        )
        self._observation_events_by_id[observation_event.event_id] = observation_event
        self._observation_event_history.append(observation_event)
        self._write_observation_event_state(observation_event)
        self._repository.upsert_observation_event(observation_event)
        self._upsert_vector_point_for_observation_event(observation_event)

        for instance_id in linked_instance_ids:
            current_instance = self._semantic_instances_by_id.get(instance_id)
            if current_instance is None:
                continue
            updated_instance = current_instance.model_copy(
                update={
                    "last_observation_event_id": observation_event.event_id,
                    "supporting_observation_event_ids": self._normalize_aliases(
                        [*current_instance.supporting_observation_event_ids, observation_event.event_id]
                    ),
                },
                deep=True,
            )
            self._semantic_instances_by_id[instance_id] = updated_instance
            self._repository.upsert_semantic_instance(updated_instance)
            self._upsert_vector_point_for_instance(updated_instance)

    def _collect_known_grounding_labels(self) -> List[str]:
        labels: List[str] = []
        for location in self._tagged_locations_by_id.values():
            labels.extend(location.semantic_labels)
            labels.append(location.name)
            labels.extend(location.aliases)
        for entry in self._semantic_entries_by_id.values():
            labels.extend(entry.tags)
            labels.extend(entry.semantic_labels)
        for instance in self._semantic_instances_by_id.values():
            labels.append(instance.label)
            labels.extend(instance.semantic_labels)
            labels.extend(instance.visual_labels)
            labels.extend(instance.vision_tags)
        return self._normalize_tags(labels)

    def _collection_for_record_kind(self, record_kind: MemoryRecordKind) -> str:
        if record_kind == MemoryRecordKind.TAGGED_LOCATION:
            return PLACE_NODES_COLLECTION
        if record_kind == MemoryRecordKind.OBJECT_INSTANCE:
            return OBJECT_INSTANCES_COLLECTION
        return EPISODIC_OBSERVATIONS_COLLECTION

    def _collections_for_record_kinds(
        self,
        record_kinds: Tuple[MemoryRecordKind, ...],
    ) -> Tuple[str, ...]:
        collections: List[str] = []
        for record_kind in record_kinds:
            collection_name = self._collection_for_record_kind(record_kind)
            if collection_name not in collections:
                collections.append(collection_name)
        return tuple(collections)

    def _resolve_anchor_id_for_pose(
        self,
        pose: Pose,
        *,
        semantic_region_id: Optional[str],
        topo_node_id: Optional[str],
    ) -> Optional[str]:
        if semantic_region_id:
            for region in self._latest_semantic_map_context.semantic_regions:
                if region.region_id == semantic_region_id:
                    return region.anchor_id
        if topo_node_id:
            node = self._latest_semantic_map_context.topology_nodes_by_id.get(topo_node_id)
            if isinstance(node, dict):
                anchor_id = str(node.get("anchor_id") or "").strip()
                if anchor_id:
                    return anchor_id
        best_anchor = None
        best_distance = None
        for anchor in self._latest_semantic_map_context.anchors_by_id.values():
            distance_m = self._distance_between_poses(pose, anchor.pose)
            if best_distance is None or distance_m < best_distance:
                best_distance = distance_m
                best_anchor = anchor
        return best_anchor.anchor_id if best_anchor is not None else None

    def _resolve_location_anchor_id(self, location: TaggedLocation) -> Optional[str]:
        metadata_anchor_id = str(location.metadata.get("anchor_id", "") or "").strip()
        if metadata_anchor_id:
            return metadata_anchor_id
        return self._resolve_anchor_id_for_pose(
            location.pose,
            semantic_region_id=location.semantic_region_id,
            topo_node_id=location.topo_node_id,
        )

    def _resolve_memory_anchor_id(self, entry: SemanticMemoryEntry) -> Optional[str]:
        metadata_anchor_id = str(entry.metadata.get("anchor_id", "") or "").strip()
        if metadata_anchor_id:
            return metadata_anchor_id
        if entry.pose is None:
            return None
        return self._resolve_anchor_id_for_pose(
            entry.pose,
            semantic_region_id=entry.semantic_region_id,
            topo_node_id=entry.topo_node_id,
        )

    def _resolve_inspection_pose(
        self,
        *,
        anchor_id: Optional[str],
        fallback_pose: Pose,
        override_poses: Optional[List[Pose]] = None,
    ) -> Pose:
        if override_poses:
            return override_poses[0]
        if anchor_id and anchor_id in self._latest_semantic_map_context.anchors_by_id:
            anchor = self._latest_semantic_map_context.anchors_by_id[anchor_id]
            if anchor.inspection_poses:
                return anchor.inspection_poses[0]
        return fallback_pose

    def _build_navigation_candidate_id(self, record_id: str) -> str:
        safe = re.sub(r"[^0-9a-z]+", "_", record_id.lower())
        safe = safe.strip("_") or "memory"
        return "navc_%s_19700101T000000Z_deadbeef" % safe[:24]

    def _fallback_anchor_id(self, record_id: str) -> str:
        safe = re.sub(r"[^0-9a-z]+", "_", record_id.lower())
        safe = safe.strip("_") or "anchor"
        return "anc_%s_19700101T000000Z_deadbeef" % safe[:24]

    def _to_legacy_navigation_candidate(self, candidate: NavigationCandidate) -> MemoryNavigationCandidate:
        record_kind = MemoryRecordKind(str(candidate.metadata.get("record_kind") or MemoryRecordKind.NOTE.value))
        return MemoryNavigationCandidate(
            record_kind=record_kind,
            record_id=candidate.record_id,
            target_pose=candidate.inspection_pose,
            target_name=candidate.target_name,
            linked_location_id=str(candidate.metadata.get("linked_location_id") or "") or None,
            map_version_id=candidate.map_version_id,
            topo_node_id=candidate.topo_node_id,
            semantic_region_id=candidate.semantic_region_id,
            distance_m=candidate.distance_m,
            verification_query=candidate.verification_query,
            verification_artifact_id=candidate.verification_artifact_id,
            verification_memory_id=str(candidate.metadata.get("verification_memory_id") or "") or None,
            metadata={
                "camera_id": str(candidate.metadata.get("camera_id") or ""),
                "anchor_id": candidate.anchor_id,
                "candidate_id": candidate.candidate_id,
            },
        )

    def _coerce_legacy_navigation_candidate(
        self,
        navigation_candidate: Optional[Union[MemoryNavigationCandidate, NavigationCandidate]],
    ) -> Optional[MemoryNavigationCandidate]:
        if navigation_candidate is None:
            return None
        if isinstance(navigation_candidate, MemoryNavigationCandidate):
            return navigation_candidate
        return self._to_legacy_navigation_candidate(navigation_candidate)

    def _write_semantic_instance_state(self, instance: SemanticInstance) -> None:
        self._state_store.write(
            StateNamespace.MEMORY,
            "instance:%s" % instance.instance_id,
            instance,
            source="memory_service",
            metadata={"kind": "semantic_instance"},
        )

    def _write_observation_event_state(self, event: ObservationEvent) -> None:
        self._state_store.write(
            StateNamespace.MEMORY,
            "observation_event:%s" % event.event_id,
            event,
            source="memory_service",
            metadata={"kind": "observation_event"},
        )

    def _distance_between_poses(self, left: Pose, right: Pose) -> float:
        delta_x = float(left.position.x) - float(right.position.x)
        delta_y = float(left.position.y) - float(right.position.y)
        delta_z = float(left.position.z) - float(right.position.z)
        return float(np.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z))

    def _resolve_semantic_region(
        self,
        pose: Pose,
        semantic_map: Optional[SemanticMap],
    ) -> Tuple[Optional[str], List[str]]:
        if semantic_map is None or not semantic_map.regions:
            if self._latest_semantic_map_context.semantic_regions:
                best_region = None
                best_distance = None
                for region in self._latest_semantic_map_context.semantic_regions:
                    if region.centroid is None:
                        continue
                    distance_m = self._distance_between_poses(pose, region.centroid)
                    if best_distance is None or distance_m < best_distance:
                        best_distance = distance_m
                        best_region = region
                if best_region is None:
                    return None, []
                labels = [best_region.label, *best_region.aliases]
                return best_region.region_id, self._normalize_tags(labels)
            return None, []
        best_region = None
        best_distance = None
        for region in semantic_map.regions:
            if region.centroid is None:
                continue
            distance_m = self._distance_between_poses(region.centroid, pose)
            if best_distance is None or distance_m < best_distance:
                best_distance = distance_m
                best_region = region
        if best_region is None:
            return None, []
        labels = [best_region.label]
        aliases = best_region.attributes.get("aliases") if hasattr(best_region, "attributes") else None
        if isinstance(aliases, list):
            labels.extend(str(item).strip() for item in aliases if str(item).strip())
        return best_region.region_id, list(dict.fromkeys(self._normalize_text(label) for label in labels if label))

    def _resolve_topology_node_id(
        self,
        pose: Pose,
        map_metadata: Dict[str, object],
        extra_metadata: Optional[Dict[str, object]],
    ) -> Optional[str]:
        metadata = dict(map_metadata or {})
        metadata.update(dict(extra_metadata or {}))
        if metadata.get("topo_node_id"):
            return str(metadata.get("topo_node_id"))
        topology_nodes = metadata.get("topology_nodes")
        if not isinstance(topology_nodes, list):
            return None
        best_node_id = None
        best_distance = None
        for node in topology_nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("node_id") or node.get("id") or "").strip()
            if not node_id:
                continue
            try:
                node_x = float(node.get("x"))
                node_y = float(node.get("y"))
                node_z = float(node.get("z", 0.0))
            except (TypeError, ValueError):
                continue
            delta_x = node_x - float(pose.position.x)
            delta_y = node_y - float(pose.position.y)
            delta_z = node_z - float(pose.position.z)
            distance_m = float(np.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z))
            if best_distance is None or distance_m < best_distance:
                best_distance = distance_m
                best_node_id = node_id
        return best_node_id

    def _get_current_pose(self) -> Optional[Pose]:
        snapshot = self._localization_service.get_latest_snapshot()
        if snapshot is not None and snapshot.current_pose is not None:
            return snapshot.current_pose
        if self._localization_service.is_available():
            refreshed = self._localization_service.refresh()
            return refreshed.current_pose
        return None

    def _get_or_refresh_localization(self, *, allow_missing: bool = False):
        snapshot = self._localization_service.get_latest_snapshot()
        if snapshot is not None:
            return snapshot
        if self._localization_service.is_available():
            return self._localization_service.refresh()
        if allow_missing:
            return None
        raise GatewayError("当前没有可用定位结果。")

    def _get_map_snapshot(self):
        snapshot = self._mapping_service.get_latest_snapshot()
        if snapshot is not None:
            self._refresh_semantic_map_context(snapshot)
            return snapshot
        if self._mapping_service.is_available():
            snapshot = self._mapping_service.refresh()
            self._refresh_semantic_map_context(snapshot)
            return snapshot
        return None

    def _load_primary_image_bytes(self, artifact_ids: List[str]) -> Optional[bytes]:
        if self._artifact_store is None:
            return None
        for artifact_id in artifact_ids:
            data = self._load_image_bytes(artifact_id)
            if data is not None:
                return data
        return None

    def _load_image_bytes(self, artifact_id: str) -> Optional[bytes]:
        if self._artifact_store is None or not artifact_id:
            return None
        try:
            return self._artifact_store.get_file_path(artifact_id).read_bytes()
        except Exception:
            return None

    def _write_tagged_location_state(self, location: TaggedLocation) -> None:
        self._state_store.write(
            StateNamespace.MEMORY,
            "location:%s" % location.location_id,
            location,
            source="memory_service",
            metadata={"kind": "tagged_location"},
        )

    def _write_semantic_memory_state(self, entry: SemanticMemoryEntry) -> None:
        self._state_store.write(
            StateNamespace.MEMORY,
            "memory:%s" % entry.memory_id,
            entry,
            source="memory_service",
            metadata={"kind": "semantic_memory"},
        )

    def _publish_memory_event(self, event_type: str, message: str, *, payload: Dict[str, object]) -> None:
        if self._event_bus is None:
            return
        self._event_bus.publish(
            self._event_bus.build_event(
                event_type,
                category=RuntimeEventCategory.SYSTEM,
                source="memory_service",
                message=message,
                payload=payload,
            )
        )

    def _normalize_aliases(self, aliases: Union[List[str], Tuple[str, ...], None]) -> List[str]:
        if not aliases:
            return []
        result: List[str] = []
        for item in aliases:
            normalized = str(item).strip()
            if normalized and normalized not in result:
                result.append(normalized)
        return result

    def _normalize_tags(self, tags: Union[List[str], Tuple[str, ...], None]) -> List[str]:
        if not tags:
            return []
        result: List[str] = []
        for item in tags:
            normalized = self._normalize_text(item)
            if normalized and normalized not in result:
                result.append(normalized)
        return result

    def _normalize_text(self, text: Optional[str]) -> str:
        raw = str(text or "").strip().lower()
        if not raw:
            return ""
        normalized = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", raw)
        return " ".join(part for part in normalized.split() if part)

    def _tokenize(self, normalized_text: str) -> List[str]:
        if not normalized_text:
            return []
        tokens = normalized_text.split()
        if tokens:
            return tokens
        return [normalized_text]

    def _token_overlap_score(self, left_tokens: List[str], right_tokens: List[str]) -> float:
        if not left_tokens or not right_tokens:
            return 0.0
        left_set = set(left_tokens)
        right_set = set(right_tokens)
        return len(left_set & right_set) / float(len(left_set | right_set))

    def _clamp_score(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))
