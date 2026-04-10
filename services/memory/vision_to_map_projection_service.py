from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from contracts.frame_semantics import frame_ids_semantically_equal
from contracts.geometry import Pose, Quaternion, Vector3
from contracts.runtime_views import PerceptionContext
from services.memory.semantic_map_builder import SemanticMapBuildResult


@dataclass(frozen=True)
class ProjectedObservation:
    """视觉目标投影到地图后的标准结果。"""

    label: str
    score: float
    pose: Pose
    camera_id: Optional[str]
    track_id: Optional[str]
    instance_type: str
    source_observation_id: str
    source_artifact_ids: Tuple[str, ...]
    semantic_region_id: Optional[str]
    topo_node_id: Optional[str]
    semantic_labels: Tuple[str, ...]
    visual_labels: Tuple[str, ...]
    attributes: Dict[str, object]


class VisionToMapProjectionService:
    """把关键帧视觉结果投影到 map frame。"""

    def project(
        self,
        perception_context: PerceptionContext,
        *,
        current_pose: Optional[Pose],
        map_context: SemanticMapBuildResult,
    ) -> Tuple[ProjectedObservation, ...]:
        if current_pose is None:
            return ()
        projected: List[ProjectedObservation] = []
        track_pose_by_id = self._build_track_pose_index(perception_context)
        for detection in perception_context.observation.detections_3d:
            projected_pose = self._resolve_detection3d_pose(detection.pose, current_pose=current_pose, map_frame=map_context.frame_id)
            projected.append(
                self._build_projected_observation(
                    label=detection.label,
                    score=detection.score,
                    pose=projected_pose,
                    camera_id=None,
                    track_id=None,
                    instance_type=str(detection.attributes.get("instance_type") or "object"),
                    source_observation_id=perception_context.observation.observation_id,
                    source_artifact_ids=tuple(perception_context.observation.artifact_ids),
                    map_context=map_context,
                    attributes=dict(detection.attributes),
                )
            )

        for detection in perception_context.observation.detections_2d:
            projected_pose = self._resolve_detection2d_pose(
                detection=detection,
                current_pose=current_pose,
                track_pose_by_id=track_pose_by_id,
                map_frame=map_context.frame_id,
            )
            if projected_pose is None:
                continue
            projected.append(
                self._build_projected_observation(
                    label=detection.label,
                    score=detection.score,
                    pose=projected_pose,
                    camera_id=detection.camera_id or perception_context.camera_id,
                    track_id=detection.track_id,
                    instance_type=str(detection.attributes.get("instance_type") or "object"),
                    source_observation_id=perception_context.observation.observation_id,
                    source_artifact_ids=tuple(perception_context.observation.artifact_ids),
                    map_context=map_context,
                    attributes=dict(detection.attributes),
                )
            )
        return tuple(projected)

    def _build_track_pose_index(self, perception_context: PerceptionContext) -> Dict[str, Pose]:
        result: Dict[str, Pose] = {}
        for track in perception_context.observation.tracks:
            if track.track_id and track.pose is not None:
                result[track.track_id] = track.pose
        return result

    def _resolve_detection3d_pose(self, pose: Pose, *, current_pose: Pose, map_frame: str) -> Pose:
        if frame_ids_semantically_equal(pose.frame_id, map_frame):
            if pose.frame_id == map_frame:
                return pose
            return pose.model_copy(update={"frame_id": map_frame}, deep=True)
        return self._relative_pose_to_map(pose, current_pose=current_pose, map_frame=map_frame)

    def _resolve_detection2d_pose(
        self,
        *,
        detection,
        current_pose: Pose,
        track_pose_by_id: Dict[str, Pose],
        map_frame: str,
    ) -> Optional[Pose]:
        projected_pose = self._pose_from_attributes(detection.attributes, map_frame=map_frame)
        if projected_pose is not None:
            return projected_pose
        if detection.track_id and detection.track_id in track_pose_by_id:
            track_pose = track_pose_by_id[detection.track_id]
            if frame_ids_semantically_equal(track_pose.frame_id, map_frame):
                if track_pose.frame_id == map_frame:
                    return track_pose
                return track_pose.model_copy(update={"frame_id": map_frame}, deep=True)
            return self._relative_pose_to_map(track_pose, current_pose=current_pose, map_frame=map_frame)

        depth_m = self._float_or_none(detection.attributes.get("depth_m"))
        lateral_offset_m = self._float_or_none(detection.attributes.get("lateral_offset_m"))
        if depth_m is not None:
            return Pose(
                frame_id=map_frame,
                position=Vector3(
                    x=float(current_pose.position.x) + depth_m,
                    y=float(current_pose.position.y) + (lateral_offset_m or 0.0),
                    z=float(current_pose.position.z),
                ),
                orientation=current_pose.orientation,
            )

        return Pose(
            frame_id=map_frame,
            position=Vector3(
                x=float(current_pose.position.x),
                y=float(current_pose.position.y),
                z=float(current_pose.position.z),
            ),
            orientation=current_pose.orientation,
        )

    def _relative_pose_to_map(self, pose: Pose, *, current_pose: Pose, map_frame: str) -> Pose:
        return Pose(
            frame_id=map_frame,
            position=Vector3(
                x=float(current_pose.position.x) + float(pose.position.x),
                y=float(current_pose.position.y) + float(pose.position.y),
                z=float(current_pose.position.z) + float(pose.position.z),
            ),
            orientation=Quaternion(
                x=float(pose.orientation.x),
                y=float(pose.orientation.y),
                z=float(pose.orientation.z),
                w=float(pose.orientation.w),
            ),
        )

    def _pose_from_attributes(self, attributes: Dict[str, object], *, map_frame: str) -> Optional[Pose]:
        if not isinstance(attributes, dict):
            return None
        if isinstance(attributes.get("map_pose"), dict):
            pose_payload = attributes["map_pose"]
            try:
                return Pose(
                    frame_id=str(pose_payload.get("frame_id") or map_frame),
                    position=Vector3(
                        x=float(pose_payload.get("x")),
                        y=float(pose_payload.get("y")),
                        z=float(pose_payload.get("z", 0.0)),
                    ),
                    orientation=Quaternion(w=1.0),
                )
            except (TypeError, ValueError):
                return None
        try:
            map_x = self._float_or_none(attributes.get("map_x"))
            map_y = self._float_or_none(attributes.get("map_y"))
            if map_x is None or map_y is None:
                return None
            map_z = self._float_or_none(attributes.get("map_z")) or 0.0
            return Pose(
                frame_id=map_frame,
                position=Vector3(x=map_x, y=map_y, z=map_z),
                orientation=Quaternion(w=1.0),
            )
        except (TypeError, ValueError):
            return None

    def _build_projected_observation(
        self,
        *,
        label: str,
        score: float,
        pose: Pose,
        camera_id: Optional[str],
        track_id: Optional[str],
        instance_type: str,
        source_observation_id: str,
        source_artifact_ids: Tuple[str, ...],
        map_context: SemanticMapBuildResult,
        attributes: Dict[str, object],
    ) -> ProjectedObservation:
        semantic_region_id, topo_node_id, semantic_labels = self._resolve_semantic_anchor(pose, map_context=map_context)
        return ProjectedObservation(
            label=label,
            score=float(score),
            pose=pose,
            camera_id=camera_id,
            track_id=track_id,
            instance_type=str(instance_type or "object"),
            source_observation_id=source_observation_id,
            source_artifact_ids=source_artifact_ids,
            semantic_region_id=semantic_region_id,
            topo_node_id=topo_node_id,
            semantic_labels=semantic_labels,
            visual_labels=(str(label).strip().lower(),),
            attributes=attributes,
        )

    def _resolve_semantic_anchor(
        self,
        pose: Pose,
        *,
        map_context: SemanticMapBuildResult,
    ) -> Tuple[Optional[str], Optional[str], Tuple[str, ...]]:
        best_region = None
        best_region_distance = None
        for region in map_context.semantic_regions:
            if region.centroid is None:
                continue
            distance_m = self._distance(pose, region.centroid)
            if best_region_distance is None or distance_m < best_region_distance:
                best_region_distance = distance_m
                best_region = region

        best_topology_node_id = None
        best_topology_distance = None
        for node_id, node in map_context.topology_nodes_by_id.items():
            try:
                node_x = float(node.get("x"))
                node_y = float(node.get("y"))
                node_z = float(node.get("z", 0.0))
            except (TypeError, ValueError):
                continue
            node_pose = Pose(
                frame_id=str(node.get("frame_id") or map_context.frame_id),
                position=Vector3(x=node_x, y=node_y, z=node_z),
                orientation=Quaternion(w=1.0),
            )
            distance_m = self._distance(pose, node_pose)
            if best_topology_distance is None or distance_m < best_topology_distance:
                best_topology_distance = distance_m
                best_topology_node_id = node_id

        semantic_labels: List[str] = []
        semantic_region_id = None
        if best_region is not None:
            semantic_region_id = best_region.region_id
            semantic_labels.extend([best_region.label, *best_region.aliases])
        if best_topology_node_id is not None:
            node = map_context.topology_nodes_by_id[best_topology_node_id]
            semantic_labels.extend(
                str(item).strip().lower()
                for item in [node.get("label"), node.get("name"), *(node.get("aliases") or [])]
                if str(item or "").strip()
            )
        normalized = []
        for item in semantic_labels:
            if item and item not in normalized:
                normalized.append(item)
        return semantic_region_id, best_topology_node_id, tuple(normalized)

    def _distance(self, left: Pose, right: Pose) -> float:
        delta_x = float(left.position.x) - float(right.position.x)
        delta_y = float(left.position.y) - float(right.position.y)
        delta_z = float(left.position.z) - float(right.position.z)
        return float((delta_x * delta_x + delta_y * delta_y + delta_z * delta_z) ** 0.5)

    def _float_or_none(self, value) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
