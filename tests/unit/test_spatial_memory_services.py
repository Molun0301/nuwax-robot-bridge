from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from contracts.base import utc_now
from contracts.geometry import Pose, Quaternion, Vector3
from contracts.maps import OccupancyGrid, SemanticMap, SemanticRegion
from contracts.perception import BoundingBox2D, Detection2D, Detection3D, Observation, Track, TrackState
from contracts.runtime_views import MapSnapshot, PerceptionContext, SceneObjectSummary, SceneSummary
from contracts.spatial_memory import InstanceLifecycleState, InstanceMovability
from services.memory.grounding_query_planner import GroundingQueryPlanner
from services.memory.inspection_pose_planner import InspectionPosePlanner
from services.memory.instance_association_service import InstanceAssociationService
from services.memory.semantic_map_builder import SemanticMapBuilder
from services.memory.vectorizer import HashingTextEmbedder
from services.memory.vision_to_map_projection_service import VisionToMapProjectionService


def _build_map_snapshot() -> MapSnapshot:
    return MapSnapshot(
        source_name="unit_test_map",
        version_id="mapv_test_000001",
        revision=1,
        occupancy_grid=OccupancyGrid(
            map_id="test_map",
            frame_id="map",
            width=20,
            height=20,
            resolution_m=0.2,
            origin=Pose(frame_id="map", position=Vector3(x=0.0, y=0.0, z=0.0)),
            data=[0] * 400,
        ),
        semantic_map=SemanticMap(
            map_id="semantic_map",
            frame_id="map",
            regions=[
                SemanticRegion(
                    region_id="dock_zone",
                    label="charging_dock",
                    centroid=Pose(frame_id="map", position=Vector3(x=1.2, y=0.4, z=0.0)),
                    attributes={"aliases": ["补给点"]},
                )
            ],
        ),
        metadata={
            "topology_nodes": [
                {"node_id": "dock_node", "x": 1.1, "y": 0.5, "z": 0.0, "label": "charging_dock", "aliases": ["dock"]}
            ]
        },
    )


def _build_perception_context() -> PerceptionContext:
    observation = Observation(
        observation_id="obs_camera_front_20260401T120000Z_deadbeef",
        frame_id="camera/front",
        summary="看到了充电器和人员。",
        detections_2d=[
            Detection2D(
                label="charger",
                score=0.91,
                bbox=BoundingBox2D(x_px=10, y_px=20, width_px=40, height_px=60),
                camera_id="front_camera",
                track_id="trk_charger",
                attributes={"depth_m": 0.8, "instance_type": "object"},
            )
        ],
        detections_3d=[
            Detection3D(
                label="person",
                score=0.96,
                pose=Pose(frame_id="map", position=Vector3(x=1.0, y=0.2, z=0.0)),
                attributes={"instance_type": "person"},
            )
        ],
        tracks=[
            Track(
                track_id="trk_charger",
                label="charger",
                state=TrackState.TRACKED,
                score=0.9,
                pose=Pose(frame_id="map", position=Vector3(x=1.3, y=0.45, z=0.0)),
            )
        ],
        artifact_ids=["art_image_1"],
    )
    scene_summary = SceneSummary(
        headline="人员站在补给架旁边。",
        details=["画面里有 charger 和 person。"],
        objects=[
            SceneObjectSummary(label="charger", count=1, max_score=0.91),
            SceneObjectSummary(label="person", count=1, max_score=0.96),
        ],
        metadata={"semantic_tags": ["补给点", "人员附近"], "semantic_relations": ["person_beside_charger"]},
    )
    return PerceptionContext(
        camera_id="front_camera",
        observation=observation,
        scene_summary=scene_summary,
        pipeline_name="test_pipeline",
        detector_backend="metadata",
        tracker_backend="basic",
    )


def test_semantic_map_builder_generates_regions_and_topology_anchors() -> None:
    snapshot = _build_map_snapshot()
    builder = SemanticMapBuilder()
    result = builder.build(snapshot)

    assert result.map_version_id == snapshot.version_id
    assert result.semantic_regions
    assert "dock_node" in result.topology_nodes_by_id
    anchor = next(iter(result.anchors_by_id.values()))
    assert anchor.pose.frame_id == "map"
    assert anchor.inspection_poses


def test_projection_service_projects_rgb_and_3d_targets_into_map_frame() -> None:
    projection_service = VisionToMapProjectionService()
    map_context = SemanticMapBuilder().build(_build_map_snapshot())
    projected = projection_service.project(
        _build_perception_context(),
        current_pose=Pose(frame_id="map", position=Vector3(x=0.5, y=0.1, z=0.0), orientation=Quaternion(w=1.0)),
        map_context=map_context,
    )

    labels = {item.label for item in projected}
    assert labels == {"charger", "person"}
    assert all(item.pose.frame_id == "map" for item in projected)
    assert any(item.semantic_region_id == "dock_zone" for item in projected)


def test_instance_association_merges_static_instances_and_marks_stale() -> None:
    map_snapshot = _build_map_snapshot()
    map_context = SemanticMapBuilder().build(map_snapshot)
    projected = VisionToMapProjectionService().project(
        _build_perception_context(),
        current_pose=Pose(frame_id="map", position=Vector3(x=0.5, y=0.1, z=0.0), orientation=Quaternion(w=1.0)),
        map_context=map_context,
    )
    association_service = InstanceAssociationService(
        text_embedder=HashingTextEmbedder(model_name="hashing-v1", dimension=64),
        inspection_pose_planner=InspectionPosePlanner(),
    )

    instances, outcomes = association_service.associate(
        projected,
        existing_instances={},
        map_version_id=map_snapshot.version_id,
        occupancy_grid=map_snapshot.occupancy_grid,
        cost_map=map_snapshot.cost_map,
        now=utc_now(),
    )
    assert len(instances) == 2
    assert outcomes

    charger_instance = next(item for item in instances.values() if item.label == "charger")
    updated_instances, second_outcomes = association_service.associate(
        [next(item for item in projected if item.label == "charger")],
        existing_instances=instances,
        map_version_id=map_snapshot.version_id,
        occupancy_grid=map_snapshot.occupancy_grid,
        cost_map=map_snapshot.cost_map,
        now=utc_now() + timedelta(minutes=1),
    )
    merged_charger = updated_instances[charger_instance.instance_id]
    assert merged_charger.observation_count >= 2
    assert second_outcomes[0].created is False

    uncertain = merged_charger.model_copy(
        update={
            "movability": InstanceMovability.MOVABLE,
            "last_seen_ts": utc_now() - timedelta(hours=3),
        },
        deep=True,
    )
    uncertain_instances, _ = association_service.associate(
        [],
        existing_instances={uncertain.instance_id: uncertain},
        map_version_id=map_snapshot.version_id,
        occupancy_grid=map_snapshot.occupancy_grid,
        cost_map=map_snapshot.cost_map,
        now=utc_now(),
    )
    assert uncertain_instances[uncertain.instance_id].lifecycle_state == InstanceLifecycleState.UNCERTAIN

    aged = merged_charger.model_copy(
        update={
            "movability": InstanceMovability.TRANSIENT,
            "last_seen_ts": utc_now() - timedelta(hours=1),
        },
        deep=True,
    )
    stale_instances, _ = association_service.associate(
        [],
        existing_instances={aged.instance_id: aged},
        map_version_id=map_snapshot.version_id,
        occupancy_grid=map_snapshot.occupancy_grid,
        cost_map=map_snapshot.cost_map,
        now=utc_now(),
    )
    assert stale_instances[aged.instance_id].lifecycle_state == InstanceLifecycleState.STALE


def test_grounding_query_planner_prioritizes_object_and_episode_collections() -> None:
    planner = GroundingQueryPlanner()
    plan = planner.plan("带我去刚才看到的红色补给架", known_labels=["charger", "person", "charging_dock"])

    assert plan.intent == "navigate"
    assert plan.temporal_hint == "recent"
    assert plan.preferred_collections[0] == "object_instances"
    assert "红色" in plan.attributes
