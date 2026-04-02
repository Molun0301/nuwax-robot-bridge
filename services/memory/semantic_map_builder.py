from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from typing import Dict, List, Optional, Tuple

from contracts.base import utc_now
from contracts.geometry import Pose, Quaternion, Vector3
from contracts.maps import SemanticMap
from contracts.runtime_views import MapSnapshot
from contracts.spatial_memory import SemanticRegion, SpatialAnchor, SpatialAnchorKind
from services.memory.inspection_pose_planner import InspectionPosePlanner


@dataclass(frozen=True)
class SemanticMapBuildResult:
    """语义地图构建结果。"""

    map_version_id: Optional[str]
    frame_id: str
    semantic_regions: Tuple[SemanticRegion, ...]
    anchors_by_id: Dict[str, SpatialAnchor]
    topology_nodes_by_id: Dict[str, Dict[str, object]]


class SemanticMapBuilder:
    """把地图快照整理为记忆侧可消费的语义区域与空间锚点。"""

    def __init__(self, *, inspection_pose_planner: Optional[InspectionPosePlanner] = None) -> None:
        self._inspection_pose_planner = inspection_pose_planner or InspectionPosePlanner()

    def build(self, snapshot: Optional[MapSnapshot]) -> SemanticMapBuildResult:
        if snapshot is None:
            return SemanticMapBuildResult(
                map_version_id=None,
                frame_id="map",
                semantic_regions=(),
                anchors_by_id={},
                topology_nodes_by_id={},
            )

        frame_id = self._resolve_frame_id(snapshot)
        topology_nodes_by_id = self._extract_topology_nodes(snapshot.metadata, frame_id=frame_id)
        anchors_by_id: Dict[str, SpatialAnchor] = {}
        semantic_regions: List[SemanticRegion] = []

        if snapshot.semantic_map is not None:
            semantic_regions.extend(
                self._build_from_semantic_map(
                    snapshot.semantic_map,
                    snapshot=snapshot,
                    topology_nodes_by_id=topology_nodes_by_id,
                    anchors_by_id=anchors_by_id,
                )
            )
        else:
            semantic_regions.extend(
                self._build_coarse_regions_from_topology(
                    snapshot=snapshot,
                    topology_nodes_by_id=topology_nodes_by_id,
                    anchors_by_id=anchors_by_id,
                    frame_id=frame_id,
                )
            )

        for node_id, node in topology_nodes_by_id.items():
            anchor_id = str(node.get("anchor_id") or "")
            if anchor_id in anchors_by_id:
                continue
            node_pose = self._build_pose_from_metadata(node, frame_id=frame_id)
            if node_pose is None:
                continue
            inspection_poses = self._inspection_pose_planner.plan_for_target(
                target_pose=node_pose,
                occupancy_grid=snapshot.occupancy_grid,
                cost_map=snapshot.cost_map,
            )
            label = str(node.get("label") or node.get("name") or node_id)
            anchors_by_id[anchor_id] = SpatialAnchor(
                anchor_id=anchor_id,
                anchor_kind=SpatialAnchorKind.TOPOLOGY_NODE,
                name=label,
                pose=node_pose,
                map_version_id=snapshot.version_id,
                topo_node_id=node_id,
                semantic_labels=self._normalize_labels(
                    [label, *(node.get("aliases") or []), *(node.get("labels") or [])]
                ),
                inspection_poses=inspection_poses,
                metadata={
                    "source": "topology_nodes",
                    "node_id": node_id,
                },
            )

        return SemanticMapBuildResult(
            map_version_id=snapshot.version_id,
            frame_id=frame_id,
            semantic_regions=tuple(semantic_regions),
            anchors_by_id=anchors_by_id,
            topology_nodes_by_id=topology_nodes_by_id,
        )

    def _build_from_semantic_map(
        self,
        semantic_map: SemanticMap,
        *,
        snapshot: MapSnapshot,
        topology_nodes_by_id: Dict[str, Dict[str, object]],
        anchors_by_id: Dict[str, SpatialAnchor],
    ) -> List[SemanticRegion]:
        regions: List[SemanticRegion] = []
        for region in semantic_map.regions:
            region_id = region.region_id
            anchor_id = self._stable_anchor_id("region", region_id)
            aliases = self._extract_aliases(region.attributes)
            topo_node_ids = self._resolve_region_topology_nodes(region, topology_nodes_by_id)
            centroid = region.centroid
            if centroid is None and region.polygon_points:
                centroid = self._centroid_from_polygon(
                    frame_id=semantic_map.frame_id,
                    polygon_points=region.polygon_points,
                )
            inspection_poses = (
                self._inspection_pose_planner.plan_for_target(
                    target_pose=centroid,
                    occupancy_grid=snapshot.occupancy_grid,
                    cost_map=snapshot.cost_map,
                )
                if centroid is not None
                else []
            )
            semantic_region = SemanticRegion(
                region_id=region_id,
                anchor_id=anchor_id,
                map_version_id=snapshot.version_id,
                frame_id=semantic_map.frame_id,
                label=region.label,
                aliases=aliases,
                centroid=centroid,
                polygon_points=list(region.polygon_points),
                topo_node_ids=topo_node_ids,
                inspection_poses=inspection_poses,
                attributes=dict(region.attributes),
            )
            regions.append(semantic_region)
            if centroid is None:
                continue
            anchors_by_id[anchor_id] = SpatialAnchor(
                anchor_id=anchor_id,
                anchor_kind=SpatialAnchorKind.SEMANTIC_REGION,
                name=region.label,
                pose=centroid,
                map_version_id=snapshot.version_id,
                topo_node_id=topo_node_ids[0] if topo_node_ids else None,
                semantic_region_id=region_id,
                semantic_labels=self._normalize_labels([region.label, *aliases]),
                inspection_poses=inspection_poses,
                metadata={"source": "semantic_map", "region_id": region_id},
            )
        return regions

    def _build_coarse_regions_from_topology(
        self,
        *,
        snapshot: MapSnapshot,
        topology_nodes_by_id: Dict[str, Dict[str, object]],
        anchors_by_id: Dict[str, SpatialAnchor],
        frame_id: str,
    ) -> List[SemanticRegion]:
        regions: List[SemanticRegion] = []
        for node_id, node in topology_nodes_by_id.items():
            label = str(node.get("label") or node.get("name") or "").strip()
            if not label:
                continue
            node_pose = self._build_pose_from_metadata(node, frame_id=frame_id)
            if node_pose is None:
                continue
            anchor_id = str(node.get("anchor_id"))
            aliases = self._normalize_labels([*(node.get("aliases") or []), *(node.get("labels") or [])])
            inspection_poses = self._inspection_pose_planner.plan_for_target(
                target_pose=node_pose,
                occupancy_grid=snapshot.occupancy_grid,
                cost_map=snapshot.cost_map,
            )
            region_id = "coarse_%s" % node_id
            regions.append(
                SemanticRegion(
                    region_id=region_id,
                    anchor_id=anchor_id,
                    map_version_id=snapshot.version_id,
                    frame_id=frame_id,
                    label=label,
                    aliases=aliases,
                    centroid=node_pose,
                    polygon_points=[],
                    topo_node_ids=[node_id],
                    inspection_poses=inspection_poses,
                    attributes={"source": "topology_metadata"},
                )
            )
            anchors_by_id[anchor_id] = SpatialAnchor(
                anchor_id=anchor_id,
                anchor_kind=SpatialAnchorKind.SEMANTIC_REGION,
                name=label,
                pose=node_pose,
                map_version_id=snapshot.version_id,
                topo_node_id=node_id,
                semantic_region_id=region_id,
                semantic_labels=self._normalize_labels([label, *aliases]),
                inspection_poses=inspection_poses,
                metadata={"source": "coarse_topology_region", "node_id": node_id},
            )
        return regions

    def _extract_topology_nodes(self, metadata: Dict[str, object], *, frame_id: str) -> Dict[str, Dict[str, object]]:
        raw_nodes = metadata.get("topology_nodes")
        if not isinstance(raw_nodes, list):
            return {}
        result: Dict[str, Dict[str, object]] = {}
        for item in raw_nodes:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("node_id") or item.get("id") or "").strip()
            if not node_id:
                continue
            result[node_id] = {
                **item,
                "anchor_id": str(item.get("anchor_id") or self._stable_anchor_id("topo", node_id)),
                "frame_id": str(item.get("frame_id") or frame_id),
            }
        return result

    def _resolve_region_topology_nodes(
        self,
        region,
        topology_nodes_by_id: Dict[str, Dict[str, object]],
    ) -> List[str]:
        topo_node_ids = region.attributes.get("topo_node_ids")
        if isinstance(topo_node_ids, list):
            return [str(item) for item in topo_node_ids if str(item).strip() in topology_nodes_by_id]
        if region.centroid is None:
            return []
        best_node_id = None
        best_distance = None
        for node_id, node in topology_nodes_by_id.items():
            node_pose = self._build_pose_from_metadata(node, frame_id=region.centroid.frame_id)
            if node_pose is None:
                continue
            distance_m = self._distance(region.centroid, node_pose)
            if best_distance is None or distance_m < best_distance:
                best_distance = distance_m
                best_node_id = node_id
        return [best_node_id] if best_node_id is not None else []

    def _build_pose_from_metadata(self, metadata: Dict[str, object], *, frame_id: str) -> Optional[Pose]:
        try:
            x = float(metadata.get("x"))
            y = float(metadata.get("y"))
            z = float(metadata.get("z", 0.0))
        except (TypeError, ValueError):
            return None
        return Pose(
            frame_id=str(metadata.get("frame_id") or frame_id),
            position=Vector3(x=x, y=y, z=z),
            orientation=Quaternion(w=1.0),
        )

    def _centroid_from_polygon(self, *, frame_id: str, polygon_points: List[Vector3]) -> Pose:
        count = max(1, len(polygon_points))
        return Pose(
            frame_id=frame_id,
            position=Vector3(
                x=sum(float(point.x) for point in polygon_points) / count,
                y=sum(float(point.y) for point in polygon_points) / count,
                z=sum(float(point.z) for point in polygon_points) / count,
            ),
            orientation=Quaternion(w=1.0),
        )

    def _resolve_frame_id(self, snapshot: MapSnapshot) -> str:
        if snapshot.semantic_map is not None:
            return snapshot.semantic_map.frame_id
        if snapshot.occupancy_grid is not None:
            return snapshot.occupancy_grid.frame_id
        if snapshot.cost_map is not None:
            return snapshot.cost_map.frame_id
        return "map"

    def _extract_aliases(self, attributes: Dict[str, object]) -> List[str]:
        aliases: List[str] = []
        alias_value = attributes.get("alias")
        if alias_value:
            aliases.append(str(alias_value))
        alias_list = attributes.get("aliases")
        if isinstance(alias_list, list):
            aliases.extend(str(item) for item in alias_list if str(item).strip())
        return self._normalize_labels(aliases)

    def _normalize_labels(self, labels: List[object]) -> List[str]:
        result: List[str] = []
        for item in labels:
            normalized = str(item or "").strip().lower()
            if normalized and normalized not in result:
                result.append(normalized)
        return result

    def _stable_anchor_id(self, kind: str, source_key: str) -> str:
        digest = hashlib.blake2b(str(source_key).encode("utf-8"), digest_size=4).hexdigest()
        return "anc_%s_19700101T000000Z_%s" % (kind, digest)

    def _distance(self, left: Pose, right: Pose) -> float:
        delta_x = float(left.position.x) - float(right.position.x)
        delta_y = float(left.position.y) - float(right.position.y)
        delta_z = float(left.position.z) - float(right.position.z)
        return float((delta_x * delta_x + delta_y * delta_y + delta_z * delta_z) ** 0.5)
