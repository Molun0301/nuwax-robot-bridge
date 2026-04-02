from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

from contracts.base import utc_now
from contracts.geometry import Pose
from contracts.naming import build_anchor_id, build_instance_id
from contracts.spatial_memory import (
    InstanceLifecycleState,
    InstanceMovability,
    SemanticInstance,
)
from services.memory.inspection_pose_planner import InspectionPosePlanner
from services.memory.vectorizer import TextEmbedder
from services.memory.vision_to_map_projection_service import ProjectedObservation


@dataclass(frozen=True)
class AssociationOutcome:
    """一次实例关联输出。"""

    instance: SemanticInstance
    created: bool
    matched_score: float


class InstanceAssociationService:
    """按空间、属性、时间连续性合并语义实例。"""

    _MERGE_DISTANCE_M = {
        InstanceMovability.STATIC: 1.8,
        InstanceMovability.MOVABLE: 0.9,
        InstanceMovability.TRANSIENT: 1.2,
    }
    _MAX_TIME_GAP_SEC = {
        InstanceMovability.STATIC: 7 * 24 * 3600.0,
        InstanceMovability.MOVABLE: 12 * 3600.0,
        InstanceMovability.TRANSIENT: 10 * 60.0,
    }
    _ATTRIBUTE_SIMILARITY_GATE = {
        InstanceMovability.STATIC: 0.18,
        InstanceMovability.MOVABLE: 0.22,
        InstanceMovability.TRANSIENT: 0.10,
    }
    _UNCERTAIN_AFTER_SEC = {
        InstanceMovability.STATIC: 3 * 24 * 3600.0,
        InstanceMovability.MOVABLE: 2 * 3600.0,
        InstanceMovability.TRANSIENT: 2 * 60.0,
    }
    _STALE_AFTER_SEC = {
        InstanceMovability.STATIC: 7 * 24 * 3600.0,
        InstanceMovability.MOVABLE: 24 * 3600.0,
        InstanceMovability.TRANSIENT: 15 * 60.0,
    }
    _REMOVED_AFTER_SEC = {
        InstanceMovability.STATIC: 30 * 24 * 3600.0,
        InstanceMovability.MOVABLE: 7 * 24 * 3600.0,
        InstanceMovability.TRANSIENT: 24 * 3600.0,
    }

    def __init__(
        self,
        *,
        text_embedder: TextEmbedder,
        inspection_pose_planner: Optional[InspectionPosePlanner] = None,
    ) -> None:
        self._text_embedder = text_embedder
        self._inspection_pose_planner = inspection_pose_planner or InspectionPosePlanner()

    def associate(
        self,
        projected_observations: Iterable[ProjectedObservation],
        *,
        existing_instances: Dict[str, SemanticInstance],
        map_version_id: Optional[str],
        occupancy_grid=None,
        cost_map=None,
        now: Optional[datetime] = None,
    ) -> Tuple[Dict[str, SemanticInstance], Tuple[AssociationOutcome, ...]]:
        resolved_now = now or utc_now()
        instances = dict(existing_instances)
        outcomes: List[AssociationOutcome] = []

        for item in projected_observations:
            best_instance = None
            best_score = 0.0
            for candidate in instances.values():
                if candidate.lifecycle_state == InstanceLifecycleState.REMOVED:
                    continue
                score = self._association_score(item, candidate, resolved_now)
                if score > best_score:
                    best_score = score
                    best_instance = candidate

            if best_instance is not None and best_score >= 0.6:
                merged = self._merge_instance(
                    best_instance,
                    item,
                    map_version_id=map_version_id,
                    occupancy_grid=occupancy_grid,
                    cost_map=cost_map,
                    now=resolved_now,
                )
                instances[merged.instance_id] = merged
                outcomes.append(AssociationOutcome(instance=merged, created=False, matched_score=best_score))
                continue

            created = self._create_instance(
                item,
                map_version_id=map_version_id,
                occupancy_grid=occupancy_grid,
                cost_map=cost_map,
                now=resolved_now,
            )
            instances[created.instance_id] = created
            outcomes.append(AssociationOutcome(instance=created, created=True, matched_score=1.0))

        stale_updated: Dict[str, SemanticInstance] = {}
        for instance_id, instance in instances.items():
            stale_updated[instance_id] = self._mark_stale(instance, now=resolved_now)
        return stale_updated, tuple(outcomes)

    def _association_score(
        self,
        observation: ProjectedObservation,
        candidate: SemanticInstance,
        now: datetime,
    ) -> float:
        label_score = 1.0 if observation.label.strip().lower() == candidate.label.strip().lower() else 0.0
        if label_score == 0.0:
            return 0.0
        distance_m = self._distance(observation.pose, candidate.pose)
        max_distance_m = self._MERGE_DISTANCE_M.get(candidate.movability, 1.0)
        if distance_m > max_distance_m:
            return 0.0
        distance_score = 1.0 / (1.0 + distance_m)
        attribute_score = self._attribute_similarity(observation, candidate)
        age_sec = max(0.0, (now - candidate.last_seen_ts).total_seconds())
        max_time_gap_sec = self._MAX_TIME_GAP_SEC.get(candidate.movability, 3600.0)
        if age_sec > max_time_gap_sec:
            return 0.0
        temporal_score = max(0.1, 1.0 - (age_sec / max_time_gap_sec))
        similarity_gate = self._ATTRIBUTE_SIMILARITY_GATE.get(candidate.movability, 0.0)
        if attribute_score < similarity_gate:
            return 0.0
        track_bonus = 0.0
        if observation.track_id and observation.track_id == str(candidate.metadata.get("last_track_id") or ""):
            track_bonus = 0.15
        return min(
            1.0,
            (0.42 * label_score) + (0.26 * distance_score) + (0.17 * attribute_score) + (0.15 * temporal_score) + track_bonus,
        )

    def _merge_instance(
        self,
        current: SemanticInstance,
        observation: ProjectedObservation,
        *,
        map_version_id: Optional[str],
        occupancy_grid,
        cost_map,
        now: datetime,
    ) -> SemanticInstance:
        updated_count = current.observation_count + 1
        blended_pose = self._blend_pose(current.pose, observation.pose, weight=float(current.observation_count) / float(updated_count))
        visual_labels = self._merge_strings(current.visual_labels, observation.visual_labels)
        semantic_labels = self._merge_strings(current.semantic_labels, observation.semantic_labels)
        artifact_ids = self._merge_strings(current.artifact_ids, observation.source_artifact_ids)
        inspection_poses = self._inspection_pose_planner.plan_for_target(
            target_pose=blended_pose,
            occupancy_grid=occupancy_grid,
            cost_map=cost_map,
            existing_poses=current.inspection_poses,
        )
        supporting_event_ids = self._merge_strings(
            current.supporting_observation_event_ids,
            (),
        )
        return current.model_copy(
            update={
                "pose": blended_pose,
                "map_version_id": map_version_id or current.map_version_id,
                "topo_node_id": observation.topo_node_id or current.topo_node_id,
                "semantic_region_id": observation.semantic_region_id or current.semantic_region_id,
                "confidence": max(float(current.confidence), float(observation.score)),
                "observation_count": updated_count,
                "last_seen_ts": now,
                "artifact_ids": artifact_ids,
                "semantic_labels": semantic_labels,
                "visual_labels": visual_labels,
                "vision_tags": self._merge_strings(current.vision_tags, observation.semantic_labels),
                "inspection_poses": inspection_poses,
                "lifecycle_state": InstanceLifecycleState.ACTIVE,
                "metadata": {
                    **dict(current.metadata),
                    "last_track_id": observation.track_id or dict(current.metadata).get("last_track_id", ""),
                    "last_camera_id": observation.camera_id or dict(current.metadata).get("last_camera_id", ""),
                    "association_policy": current.movability.value,
                    "uncertain_after_sec": self._UNCERTAIN_AFTER_SEC[current.movability],
                    "stale_after_sec": self._STALE_AFTER_SEC[current.movability],
                    "removed_after_sec": self._REMOVED_AFTER_SEC[current.movability],
                },
                "supporting_observation_event_ids": supporting_event_ids,
            },
            deep=True,
        )

    def _create_instance(
        self,
        observation: ProjectedObservation,
        *,
        map_version_id: Optional[str],
        occupancy_grid,
        cost_map,
        now: datetime,
    ) -> SemanticInstance:
        anchor_id = build_anchor_id(observation.label[:24] or "object", timestamp=now)
        instance_id = build_instance_id(observation.label[:24] or "object", timestamp=now)
        inspection_poses = self._inspection_pose_planner.plan_for_target(
            target_pose=observation.pose,
            occupancy_grid=occupancy_grid,
            cost_map=cost_map,
        )
        movability = self._infer_movability(observation.label, observation.attributes)
        return SemanticInstance(
            instance_id=instance_id,
            anchor_id=anchor_id,
            label=observation.label.strip().lower(),
            display_name=str(observation.attributes.get("display_name_zh") or observation.label),
            pose=observation.pose,
            map_version_id=map_version_id,
            topo_node_id=observation.topo_node_id,
            semantic_region_id=observation.semantic_region_id,
            instance_type=observation.instance_type,
            movability=movability,
            lifecycle_state=InstanceLifecycleState.ACTIVE,
            confidence=float(observation.score),
            observation_count=1,
            first_seen_ts=now,
            last_seen_ts=now,
            artifact_ids=list(observation.source_artifact_ids),
            semantic_labels=list(observation.semantic_labels),
            visual_labels=list(observation.visual_labels),
            vision_tags=list(observation.semantic_labels),
            inspection_poses=inspection_poses,
            attributes=dict(observation.attributes),
            metadata={
                "last_track_id": observation.track_id or "",
                "last_camera_id": observation.camera_id or "",
                "association_policy": movability.value,
                "uncertain_after_sec": self._UNCERTAIN_AFTER_SEC[movability],
                "stale_after_sec": self._STALE_AFTER_SEC[movability],
                "removed_after_sec": self._REMOVED_AFTER_SEC[movability],
            },
        )

    def _mark_stale(self, instance: SemanticInstance, *, now: datetime) -> SemanticInstance:
        age_sec = max(0.0, (now - instance.last_seen_ts).total_seconds())
        lifecycle_state = InstanceLifecycleState.ACTIVE
        removed_after_sec = self._REMOVED_AFTER_SEC.get(instance.movability, 7 * 24 * 3600.0)
        stale_after_sec = self._STALE_AFTER_SEC.get(instance.movability, 24 * 3600.0)
        uncertain_after_sec = self._UNCERTAIN_AFTER_SEC.get(instance.movability, 3600.0)
        if age_sec >= removed_after_sec:
            lifecycle_state = InstanceLifecycleState.REMOVED
        elif age_sec >= stale_after_sec:
            lifecycle_state = InstanceLifecycleState.STALE
        elif age_sec >= uncertain_after_sec:
            lifecycle_state = InstanceLifecycleState.UNCERTAIN
        if lifecycle_state == instance.lifecycle_state:
            return instance
        return instance.model_copy(
            update={
                "lifecycle_state": lifecycle_state,
                "metadata": {
                    **dict(instance.metadata),
                    "association_policy": instance.movability.value,
                    "uncertain_after_sec": uncertain_after_sec,
                    "stale_after_sec": stale_after_sec,
                    "removed_after_sec": removed_after_sec,
                },
            },
            deep=True,
        )

    def _infer_movability(self, label: str, attributes: Dict[str, object]) -> InstanceMovability:
        lowered_label = str(label).strip().lower()
        combined = " ".join(
            str(item).strip().lower()
            for item in [lowered_label, attributes.get("movability"), attributes.get("state"), attributes.get("category")]
            if str(item or "").strip()
        )
        if any(keyword in combined for keyword in ("wall", "door", "dock", "station", "shelf", "charger", "table")):
            return InstanceMovability.STATIC
        if any(keyword in combined for keyword in ("person", "dog", "cat", "visitor", "worker")):
            return InstanceMovability.TRANSIENT
        return InstanceMovability.MOVABLE

    def _attribute_similarity(self, observation: ProjectedObservation, candidate: SemanticInstance) -> float:
        left_text = " ".join(
            [
                observation.label,
                " ".join(observation.semantic_labels),
                " ".join("%s:%s" % (key, value) for key, value in sorted(observation.attributes.items())),
            ]
        ).strip()
        right_text = " ".join(
            [
                candidate.label,
                " ".join(candidate.semantic_labels),
                " ".join("%s:%s" % (key, value) for key, value in sorted(candidate.attributes.items())),
            ]
        ).strip()
        if not left_text or not right_text:
            return 0.0
        left_vector = self._text_embedder.embed_text(left_text)
        right_vector = self._text_embedder.embed_text(right_text)
        return max(0.0, self._text_embedder.cosine_similarity(left_vector, right_vector))

    def _blend_pose(self, left: Pose, right: Pose, *, weight: float) -> Pose:
        clamped_weight = max(0.0, min(1.0, float(weight)))
        return Pose(
            frame_id=right.frame_id,
            position=left.position.model_copy(
                update={
                    "x": (clamped_weight * float(left.position.x)) + ((1.0 - clamped_weight) * float(right.position.x)),
                    "y": (clamped_weight * float(left.position.y)) + ((1.0 - clamped_weight) * float(right.position.y)),
                    "z": (clamped_weight * float(left.position.z)) + ((1.0 - clamped_weight) * float(right.position.z)),
                }
            ),
            orientation=right.orientation,
        )

    def _distance(self, left: Pose, right: Pose) -> float:
        delta_x = float(left.position.x) - float(right.position.x)
        delta_y = float(left.position.y) - float(right.position.y)
        delta_z = float(left.position.z) - float(right.position.z)
        return float((delta_x * delta_x + delta_y * delta_y + delta_z * delta_z) ** 0.5)

    def _merge_strings(self, left: Iterable[str], right: Iterable[str]) -> List[str]:
        result: List[str] = []
        for item in list(left) + list(right):
            normalized = str(item or "").strip()
            if normalized and normalized not in result:
                result.append(normalized)
        return result
