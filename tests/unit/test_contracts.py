from __future__ import annotations

from datetime import datetime, timezone

import pytest

from contracts.artifacts import ArtifactKind, ArtifactRef
from contracts.capabilities import CapabilityDescriptor, CapabilityExecutionMode, CapabilityRiskLevel
from contracts.events import RuntimeEvent, RuntimeEventCategory, RuntimeEventSeverity
from contracts.frame_semantics import frame_ids_semantically_equal, infer_frame_role
from contracts.geometry import Pose, Quaternion, Transform, Twist, Vector3
from contracts.image import CameraInfo, ImageEncoding, ImageFrame
from contracts.maps import OccupancyGrid
from contracts.memory import MemoryQueryMatch, MemoryQueryResult, MemoryRecordKind, SemanticMemoryEntry, TaggedLocation
from contracts.navigation import NavigationGoal
from contracts.naming import (
    build_anchor_id,
    build_artifact_id,
    build_event_id,
    build_instance_id,
    build_location_id,
    build_memory_id,
    build_navigation_candidate_id,
    build_observation_id,
    build_observation_event_id,
    build_task_id,
    validate_capability_name,
    validate_event_type,
    validate_frame_id,
)
from contracts.perception import BoundingBox2D, Detection2D, Observation, Track, TrackState
from contracts.robot_state import IMUState, JointState, RobotControlMode, RobotState, SafetyState
from contracts.skills import SkillCategory, SkillDescriptor, SkillRuntimeView
from contracts.spatial_memory import (
    GroundingQueryPlan,
    NavigationCandidate,
    ObservationEvent,
    SemanticInstance,
    SpatialAnchor,
    SpatialAnchorKind,
    VerificationResult,
)
from contracts.tasks import TaskEvent, TaskSpec, TaskState, TaskStatus


def test_pose_and_robot_state_can_roundtrip() -> None:
    """统一契约应能稳定序列化和反序列化。"""

    robot_state = RobotState(
        robot_id="go2_dev",
        frame_id="world/go2/base",
        mode=RobotControlMode.HIGH_LEVEL,
        pose=Pose(
            frame_id="world/go2/base",
            position=Vector3(x=1.0, y=2.0, z=0.0),
            orientation=Quaternion(),
        ),
        twist=Twist(frame_id="world/go2/base", linear=Vector3(x=0.2), angular=Vector3(z=0.1)),
        joints=[JointState(name="hip", position_rad=0.1, velocity_rad_s=0.0)],
        imu=IMUState(frame_id="world/go2/imu"),
        safety=SafetyState(is_estopped=False, motors_enabled=True, can_move=True),
    )

    payload = robot_state.model_dump_json()
    restored = RobotState.model_validate_json(payload)

    assert restored.robot_id == "go2_dev"
    assert restored.pose is not None
    assert restored.pose.position.x == pytest.approx(1.0)
    assert restored.joints[0].name == "hip"


def test_image_frame_requires_payload() -> None:
    """图像契约必须包含至少一种可消费载荷。"""

    with pytest.raises(ValueError):
        ImageFrame(
            camera_id="front",
            frame_id="world/go2/camera_front",
            width_px=640,
            height_px=480,
            encoding=ImageEncoding.JPEG,
        )


def test_frame_semantics_support_scoped_aliases_without_cross_scope_confusion() -> None:
    """坐标系语义比较应接受 scoped alias（带路径别名），同时避免跨作用域误判。"""

    assert infer_frame_role("world/go2/map") == "map"
    assert infer_frame_role("world/go2/body") == "base"
    assert frame_ids_semantically_equal("map", "world/go2/map") is True
    assert frame_ids_semantically_equal("body", "world/go2/body") is True
    assert frame_ids_semantically_equal("map", "odom") is False
    assert frame_ids_semantically_equal("world/go2/map", "world/other_robot/map") is False


def test_navigation_goal_requires_pose_or_name() -> None:
    """导航目标不能完全为空。"""

    with pytest.raises(ValueError):
        NavigationGoal(goal_id="goal_1")


def test_occupancy_grid_checks_size() -> None:
    """栅格地图尺寸必须自洽。"""

    with pytest.raises(ValueError):
        OccupancyGrid(
            map_id="map_1",
            frame_id="world/map",
            width=2,
            height=2,
            resolution_m=0.05,
            origin=Pose(frame_id="world/map"),
            data=[0, 1, 2],
        )


def test_artifact_and_task_models_validate_names() -> None:
    """命名规则要能落到实际契约模型。"""

    artifact_id = build_artifact_id("image", datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc), "deadbeef")
    artifact = ArtifactRef(artifact_id=artifact_id, kind=ArtifactKind.IMAGE, mime_type="image/jpeg")
    task = TaskSpec(task_id="task_capture_image_001", capability_name="capture_image")
    event = TaskEvent(event_id="evt_1", event_type="task.accepted", task_id=task.task_id, state=TaskState.ACCEPTED)

    assert artifact.artifact_id.endswith("deadbeef")
    assert task.capability_name == "capture_image"
    assert event.event_type == "task.accepted"


def test_capability_descriptor_and_observation_are_valid() -> None:
    """能力和观察契约应能表达首批平台需求。"""

    capability = CapabilityDescriptor(
        name="capture_image",
        display_name="抓取图像",
        description="从指定相机抓取单帧图像。",
        execution_mode=CapabilityExecutionMode.SYNC,
        risk_level=CapabilityRiskLevel.LOW,
        input_contract="ImageFrame",
        output_contract="ArtifactRef",
        exposed_to_agent=True,
    )

    observation = Observation(
        observation_id=build_observation_id(
            "camera_front",
            datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc),
            "deadbeef",
        ),
        frame_id="world/go2/camera_front",
        detections_2d=[
            Detection2D(
                label="person",
                score=0.99,
                bbox=BoundingBox2D(x_px=10, y_px=20, width_px=100, height_px=200),
            )
        ],
        tracks=[Track(track_id="track_1", label="person", state=TrackState.TRACKED, score=0.9)],
    )

    assert capability.name == "capture_image"
    assert observation.detections_2d[0].label == "person"


def test_naming_helpers_cover_examples() -> None:
    """命名校验应覆盖首版规范示例。"""

    assert validate_frame_id("world/go2/base") == "world/go2/base"
    assert validate_capability_name("navigate_to_pose") == "navigate_to_pose"
    assert validate_event_type("task.succeeded") == "task.succeeded"
    assert build_task_id("capture_image", datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc), "deadbeef").startswith(
        "task_capture_image_20260330t120000z_"
    )
    assert build_observation_id(
        "camera_front",
        datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc),
        "deadbeef",
    ).startswith("obs_camera_front_20260330T120000Z_")
    assert build_event_id("task.accepted", datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc), "deadbeef").startswith(
        "evt_task_accepted_20260330T120000Z_"
    )


def test_supporting_contracts_can_be_constructed() -> None:
    """补充契约构造能力应正常。"""

    camera_info = CameraInfo(
        camera_id="front",
        frame_id="world/go2/camera_front",
        width_px=640,
        height_px=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
    )
    transform = Transform(parent_frame_id="world/go2/base", child_frame_id="world/go2/camera_front")
    status = TaskStatus(task_id="task_capture_image_001", state=TaskState.RUNNING, progress=0.5, stage="capturing")

    assert camera_info.width_px == 640
    assert transform.child_frame_id == "world/go2/camera_front"
    assert status.state == TaskState.RUNNING


def test_runtime_event_contract_can_be_constructed() -> None:
    """运行时事件契约应支持首版事件总线。"""

    event = RuntimeEvent(
        event_id="evt_task_progress_20260330T120000Z_deadbeef",
        event_type="task.progress",
        category=RuntimeEventCategory.TASK,
        source="task_manager",
        task_id="task_capture_image_001",
        severity=RuntimeEventSeverity.INFO,
        message="任务正在执行。",
        payload={"progress": 0.5},
    )

    assert event.category == RuntimeEventCategory.TASK
    assert event.payload["progress"] == 0.5


def test_memory_and_skill_contracts_can_be_constructed() -> None:
    """记忆和技能契约应满足首版工具层需求。"""

    location = TaggedLocation(
        location_id=build_location_id("tag", datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc), "deadbeef"),
        name="补给点",
        normalized_name="补给点",
        aliases=["dock"],
        pose=Pose(frame_id="map", position=Vector3(x=1.0, y=2.0, z=0.0)),
    )
    memory = SemanticMemoryEntry(
        memory_id=build_memory_id("scene", datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc), "deadbeef"),
        kind=MemoryRecordKind.SCENE,
        title="当前巡检画面",
        summary="画面中存在 person 和 dock。",
        tags=["person", "dock"],
        linked_location_id=location.location_id,
    )
    result = MemoryQueryResult(
        query="dock",
        similarity_threshold=0.25,
        matches=[
            MemoryQueryMatch(
                record_kind=MemoryRecordKind.SCENE,
                record_id=memory.memory_id,
                score=0.8,
                semantic_memory=memory,
            )
        ],
    )
    descriptor = SkillDescriptor(
        name="tag_location",
        display_name="标记当前位置",
        description="把当前位置写入命名地点记忆。",
        category=SkillCategory.MEMORY,
        capability_name="tag_location",
    )
    runtime_view = SkillRuntimeView(
        descriptor=descriptor,
        supported=True,
        runnable=True,
        capability_name="tag_location",
    )

    assert location.location_id.startswith("loc_tag_")
    assert memory.memory_id.startswith("mem_scene_")
    assert result.matches[0].semantic_memory is not None
    assert runtime_view.descriptor.category == SkillCategory.MEMORY


def test_spatial_memory_contracts_can_be_constructed() -> None:
    """空间语义记忆契约应满足语义空间闭环需求。"""

    anchor = SpatialAnchor(
        anchor_id=build_anchor_id("region", datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc), "deadbeef"),
        anchor_kind=SpatialAnchorKind.SEMANTIC_REGION,
        name="补给区",
        pose=Pose(frame_id="map", position=Vector3(x=1.0, y=2.0, z=0.0)),
    )
    instance = SemanticInstance(
        instance_id=build_instance_id("charger", datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc), "deadbeef"),
        anchor_id=anchor.anchor_id,
        label="charger",
        pose=Pose(frame_id="map", position=Vector3(x=1.0, y=2.1, z=0.0)),
        first_seen_ts=datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc),
        last_seen_ts=datetime(2026, 3, 30, 12, 1, 0, tzinfo=timezone.utc),
    )
    event = ObservationEvent(
        event_id=build_observation_event_id("perception", datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc), "deadbeef"),
        title="补给点观察",
        summary="看到了 charger。",
        anchor_id=anchor.anchor_id,
    )
    candidate = NavigationCandidate(
        candidate_id=build_navigation_candidate_id("memory", datetime(2026, 3, 30, 12, 0, 0, tzinfo=timezone.utc), "deadbeef"),
        anchor_id=anchor.anchor_id,
        source_collection="object_instances",
        record_id=instance.instance_id,
        target_name="补给架",
        inspection_pose=Pose(frame_id="map", position=Vector3(x=0.5, y=2.0, z=0.0)),
    )
    verification = VerificationResult(anchor_id=anchor.anchor_id, verified=True, score=0.9)
    plan = GroundingQueryPlan(
        raw_query="带我去红色补给架",
        normalized_query="带我去红色补给架",
        intent="navigate",
    )

    assert anchor.anchor_kind == SpatialAnchorKind.SEMANTIC_REGION
    assert instance.anchor_id == anchor.anchor_id
    assert event.anchor_id == anchor.anchor_id
    assert candidate.source_collection == "object_instances"
    assert verification.verified is True
    assert plan.intent == "navigate"
