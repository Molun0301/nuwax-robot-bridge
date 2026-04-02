from __future__ import annotations

import json
from pathlib import Path
import time

from fastapi.testclient import TestClient
import httpx

from contracts.capabilities import CapabilityMatrix
from contracts.geometry import FrameTree, Pose, Quaternion, Transform, Twist, Vector3
from contracts.image import CameraInfo, ImageEncoding, ImageFrame
from contracts.maps import CostMap, OccupancyGrid, SemanticMap, SemanticRegion
from contracts.navigation import ExplorationState, ExplorationStatus, ExploreAreaRequest, NavigationGoal, NavigationState, NavigationStatus
from contracts.robot_state import IMUState, JointState, RobotControlMode, RobotState, SafetyState
from drivers.robots.common.assembly import RobotAssemblyBase, RobotAssemblyStatus
from drivers.robots.common.manifest import RobotDefaults, RobotManifest
from gateways.app import create_gateway_app
from gateways.relay.server import create_relay_app
from gateways.runtime import create_default_gateway_runtime
from providers import ExplorationProvider, ImageProvider, LocalizationProvider, MapProvider, NavigationProvider
from providers.motion import MotionControl
from providers.safety import SafetyProvider
from providers.state import StateProvider
from settings import load_config
from typing import Dict, List, Optional, Tuple


def _agent_headers() -> Dict[str, str]:
    return {"Authorization": "Bearer agent-token"}


def _admin_headers() -> Dict[str, str]:
    return {"Authorization": "Bearer admin-token"}


def _relay_headers() -> Dict[str, str]:
    return {"Authorization": "Bearer relay-token"}


class FakeProviderBundle(
    StateProvider,
    ImageProvider,
    MotionControl,
    SafetyProvider,
    LocalizationProvider,
    MapProvider,
    NavigationProvider,
    ExplorationProvider,
):
    """用于集成测试的假提供器集合。"""

    provider_name = "fake_bundle"
    provider_version = "0.1.0"

    def __init__(self, robot: "FakeRobotAssembly") -> None:
        self.robot = robot

    def is_available(self) -> bool:
        return self.robot.started

    def get_robot_state(self) -> RobotState:
        return RobotState(
            robot_id="fake_go2",
            frame_id="world/fake_go2/base",
            mode=RobotControlMode.HIGH_LEVEL if self.robot.current_mode == "high" else RobotControlMode.LOW_LEVEL,
            imu=self.get_imu_state(),
            safety=self.get_safety_state(),
            metadata={
                "move_command_count": len(self.robot.move_commands),
                "navigation_goal_count": self.robot.navigation_goal_count,
            },
        )

    def get_joint_states(self) -> list:
        return [
            JointState(name="front_left_hip", position_rad=0.12, velocity_rad_s=0.01),
            JointState(name="front_left_knee", position_rad=-0.25, velocity_rad_s=0.0),
        ]

    def get_imu_state(self) -> IMUState:
        return IMUState(
            frame_id="world/fake_go2/imu",
            orientation=Quaternion(w=1.0),
        )

    def capture_image(self, camera_id: Optional[str] = None) -> ImageFrame:
        camera = self.get_camera_info(camera_id)
        assert camera is not None
        return ImageFrame(
            camera_id=camera.camera_id,
            frame_id=camera.frame_id,
            width_px=camera.width_px,
            height_px=camera.height_px,
            encoding=ImageEncoding.JPEG,
            data=b"fake-jpeg-data",
            metadata={
                "detections_2d": [
                    {
                        "label": "person",
                        "score": 0.96,
                        "bbox": {"x_px": 40, "y_px": 60, "width_px": 120, "height_px": 220},
                    },
                    {
                        "label": "box",
                        "score": 0.88,
                        "bbox": {"x_px": 260, "y_px": 180, "width_px": 100, "height_px": 120},
                    },
                ]
            },
        )

    def get_camera_info(self, camera_id: Optional[str] = None) -> CameraInfo:
        return CameraInfo(
            camera_id=camera_id or "front_camera",
            frame_id="world/fake_go2/front_camera",
            width_px=640,
            height_px=480,
            fx=320.0,
            fy=320.0,
            cx=320.0,
            cy=240.0,
        )

    def send_twist(self, twist: Twist) -> None:
        self.robot.move_commands.append(twist)

    def stop_motion(self) -> None:
        self.robot.stop_count += 1

    def get_safety_state(self) -> SafetyState:
        return SafetyState(is_estopped=False, motors_enabled=self.robot.started, can_move=self.robot.started)

    def request_safe_stop(self, reason: Optional[str] = None) -> None:
        del reason
        self.stop_motion()

    def get_current_pose(self) -> Optional[Pose]:
        return self.robot.current_pose

    def get_frame_tree(self) -> Optional[FrameTree]:
        return FrameTree(
            root_frame_id="world",
            transforms=[
                Transform(
                    parent_frame_id="world",
                    child_frame_id="world/fake_go2/base",
                    translation=self.robot.current_pose.position,
                    rotation=self.robot.current_pose.orientation,
                    authority="fake_localization",
                )
            ],
        )

    def get_occupancy_grid(self) -> Optional[OccupancyGrid]:
        return OccupancyGrid(
            map_id="fake_slam_map",
            frame_id="map",
            width=2,
            height=2,
            resolution_m=0.1,
            origin=Pose(frame_id="map", position=Vector3(x=0.0, y=0.0, z=0.0)),
            data=[0, 0, 20, 100],
        )

    def get_cost_map(self) -> Optional[CostMap]:
        return CostMap(
            map_id="fake_cost_map",
            frame_id="map",
            width=2,
            height=2,
            resolution_m=0.1,
            origin=Pose(frame_id="map", position=Vector3(x=0.0, y=0.0, z=0.0)),
            data=[1.0, 2.0, 10.0, 100.0],
        )

    def get_semantic_map(self) -> Optional[SemanticMap]:
        return SemanticMap(
            map_id="fake_semantic_map",
            frame_id="map",
            regions=[
                SemanticRegion(
                    region_id="charging_station",
                    label="charging_station",
                    centroid=Pose(frame_id="map", position=Vector3(x=1.0, y=1.0, z=0.0)),
                    attributes={"alias": "充电桩", "aliases": ["dock", "charging dock"]},
                ),
                SemanticRegion(
                    region_id="meeting_point",
                    label="meeting_point",
                    centroid=Pose(frame_id="map", position=Vector3(x=2.0, y=0.5, z=0.0)),
                    attributes={"aliases": ["集合点"]},
                ),
            ],
        )

    def set_goal(self, goal: NavigationGoal) -> bool:
        self.robot.navigation_goal_count += 1
        self.robot.active_goal = goal
        self.robot.last_goal_id = goal.goal_id
        self.robot.nav_poll_count = 0
        self.robot.nav_cancelled = False
        return True

    def cancel_goal(self) -> bool:
        self.robot.nav_cancelled = True
        self.robot.active_goal = None
        return True

    def get_navigation_state(self) -> NavigationState:
        goal = self.robot.active_goal
        if self.robot.nav_cancelled:
            return NavigationState(
                current_goal_id=self.robot.last_goal_id,
                status=NavigationStatus.CANCELLED,
                current_pose=self.robot.current_pose,
                message="测试后端已取消导航。",
            )
        if goal is None:
            return NavigationState(
                current_goal_id=self.robot.last_goal_id,
                status=NavigationStatus.IDLE,
                current_pose=self.robot.current_pose,
            )

        self.robot.nav_poll_count += 1
        target_pose = goal.target_pose or self.robot.current_pose
        target_x = target_pose.position.x
        if target_x >= 90.0:
            return NavigationState(
                current_goal_id=goal.goal_id,
                status=NavigationStatus.RUNNING,
                current_pose=self.robot.current_pose,
                remaining_distance_m=5.0,
                message="测试后端模拟长时间导航中。",
            )
        if target_x <= -90.0 and self.robot.nav_poll_count >= 2:
            self.robot.active_goal = None
            return NavigationState(
                current_goal_id=goal.goal_id,
                status=NavigationStatus.FAILED,
                current_pose=self.robot.current_pose,
                remaining_distance_m=3.0,
                message="测试后端模拟导航失败。",
            )
        if self.robot.nav_poll_count >= 3:
            self.robot.current_pose = target_pose
            self.robot.active_goal = None
            return NavigationState(
                current_goal_id=goal.goal_id,
                status=NavigationStatus.SUCCEEDED,
                current_pose=self.robot.current_pose,
                remaining_distance_m=0.0,
                goal_reached=True,
                message="测试后端已到达目标。",
            )
        return NavigationState(
            current_goal_id=goal.goal_id,
            status=NavigationStatus.RUNNING,
            current_pose=self.robot.current_pose,
            remaining_distance_m=max(0.0, 3.0 - float(self.robot.nav_poll_count)),
            message="测试后端导航执行中。",
        )

    def is_goal_reached(self) -> bool:
        return self.robot.active_goal is None and not self.robot.nav_cancelled and self.robot.nav_poll_count >= 3

    def start_exploration(self, request: ExploreAreaRequest) -> bool:
        self.robot.active_exploration_request = request
        self.robot.exploration_poll_count = 0
        self.robot.exploration_cancelled = False
        return True

    def stop_exploration(self) -> bool:
        self.robot.exploration_cancelled = True
        self.robot.active_exploration_request = None
        return True

    def get_exploration_state(self) -> ExplorationState:
        request = self.robot.active_exploration_request
        if self.robot.exploration_cancelled:
            return ExplorationState(
                current_request_id=self.robot.last_exploration_request_id,
                status=ExplorationStatus.CANCELLED,
                strategy="frontier",
                message="测试后端已取消探索。",
            )
        if request is None:
            return ExplorationState(
                current_request_id=self.robot.last_exploration_request_id,
                status=ExplorationStatus.IDLE,
            )

        self.robot.last_exploration_request_id = request.request_id
        self.robot.exploration_poll_count += 1
        if request.strategy == "hang":
            return ExplorationState(
                current_request_id=request.request_id,
                status=ExplorationStatus.RUNNING,
                strategy=request.strategy,
                covered_ratio=0.4,
                frontier_count=3,
                message="测试后端模拟长时间探索中。",
            )
        if self.robot.exploration_poll_count >= 3:
            self.robot.active_exploration_request = None
            return ExplorationState(
                current_request_id=request.request_id,
                status=ExplorationStatus.SUCCEEDED,
                strategy=request.strategy,
                covered_ratio=1.0,
                frontier_count=0,
                message="测试后端已完成探索。",
            )
        return ExplorationState(
            current_request_id=request.request_id,
            status=ExplorationStatus.RUNNING,
            strategy=request.strategy,
            covered_ratio=min(0.9, self.robot.exploration_poll_count * 0.35),
            frontier_count=max(0, 4 - self.robot.exploration_poll_count),
            message="测试后端探索执行中。",
        )


class FakeRobotAssembly(RobotAssemblyBase):
    """用于集成测试的假机器人入口。"""

    def __init__(self) -> None:
        self.manifest = RobotManifest(
            robot_name="fake_go2",
            robot_model="fake_go2",
            entrypoint="tests/integration/test_gateway_integration.py:FakeRobotAssembly",
            description="测试用假机器人入口。",
            capability_matrix=CapabilityMatrix(robot_model="fake_go2", capabilities=[]),
        )
        self.defaults = RobotDefaults(frame_ids={"base": "world/fake_go2/base"}, topics={})
        self._initialize_adapter_runtime()
        self.current_pose = Pose(frame_id="map", position=Vector3(x=0.0, y=0.0, z=0.0))
        self.started = False
        self.current_mode = "high"
        self.move_commands: List[Twist] = []
        self.stop_count = 0
        self.active_goal: Optional[NavigationGoal] = None
        self.last_goal_id: Optional[str] = None
        self.nav_poll_count = 0
        self.nav_cancelled = False
        self.navigation_goal_count = 0
        self.active_exploration_request: Optional[ExploreAreaRequest] = None
        self.last_exploration_request_id: Optional[str] = None
        self.exploration_poll_count = 0
        self.exploration_cancelled = False
        self.volume_ratio = 0.5
        self.volume_switch_enabled = True
        self.speech_requests: List[Dict[str, object]] = []
        self.action_history: List[Tuple[str, Dict[str, object]]] = []
        self.providers = FakeProviderBundle(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def get_status(self) -> RobotAssemblyStatus:
        return RobotAssemblyStatus(
            robot_name=self.manifest.robot_name,
            initialized=self.started,
            control_mode=self.current_mode,
            low_level_ready=True,
            low_level_running=self.current_mode == "low",
        )

    def switch_mode(self, mode: str):
        self.current_mode = mode
        return 0, {"mode": mode}

    def get_vui_volume_info(self) -> Dict[str, object]:
        return {
            "volume": self.volume_ratio,
            "backend": "fake_vui",
            "switch_enabled": self.volume_switch_enabled,
        }

    def set_vui_volume_ratio(self, volume: float, auto_enable_switch: bool) -> Dict[str, object]:
        self.volume_ratio = max(0.0, min(1.0, float(volume)))
        if auto_enable_switch:
            self.volume_switch_enabled = True
        return self.get_vui_volume_info()

    def execute_action(self, action: str, params: Optional[Dict[str, object]] = None) -> Tuple[Optional[int], object]:
        payload = dict(params or {})
        self.action_history.append((action, payload))
        return 0, {"action": action, "params": payload}


def _build_test_config(tmp_path: Path):
    config = load_config()
    config.gateway.agent_tokens = ("agent-token",)
    config.gateway.admin_tokens = ("admin-token",)
    config.gateway.allow_unknown_client_hosts = True
    config.gateway.allowed_source_cidrs = ("127.0.0.1/32",)
    config.gateway.public_base_url = "http://testserver"
    config.gateway.artifact_dir = str(tmp_path / "artifacts")
    config.gateway.sse_keepalive_sec = 0.05
    config.gateway.ws_ping_interval_sec = 0.05
    config.runtime_data.memory_db_path = str(tmp_path / "runtime_data" / "memory" / "vector_memory.db")
    config.runtime_data.memory_embedding_model = "hashing-v1"
    config.runtime_data.memory_embedding_dimension = 128
    config.runtime_data.memory_image_embedding_model = "disabled"
    config.relay.enabled = True
    config.relay.incoming_tokens = ("relay-token",)
    config.relay.upstream_base_url = "http://upstream.test"
    config.relay.upstream_token = "agent-token"
    return config


def _build_host_app(tmp_path: Path, *, start_runtime_on_lifespan: bool = True):
    config = _build_test_config(tmp_path)
    robot = FakeRobotAssembly()
    runtime = create_default_gateway_runtime(config, robot)
    app = create_gateway_app(runtime, config, start_runtime_on_lifespan=start_runtime_on_lifespan)
    return app, runtime, robot, config


def _wait_for_task(client: TestClient, task_id: str) -> dict:
    deadline = time.time() + 2.0
    while time.time() < deadline:
        response = client.get(f"/api/tasks/{task_id}", headers=_agent_headers())
        payload = response.json()
        if payload["status"]["state"] in {"succeeded", "failed", "cancelled", "timeout"}:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"任务 {task_id} 未在超时时间内结束。")


def _wait_for_condition(predicate, *, timeout_sec: float = 1.0) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("条件在超时时间内未满足。")


def test_gateway_mcp_http_and_auth_behaviour(tmp_path: Path) -> None:
    """宿主机网关应同时支持 MCP、HTTP 和角色鉴权。"""

    app, _, robot, _ = _build_host_app(tmp_path)
    with TestClient(app) as client:
        health = client.get("/api/health", headers=_agent_headers())
        assert health.status_code == 200
        assert health.json()["robot_name"] == "fake_go2"

        initialize = client.post(
            "/mcp",
            headers=_agent_headers(),
            json={"jsonrpc": "2.0", "id": "init", "method": "initialize"},
        )
        assert initialize.status_code == 200
        assert initialize.json()["result"]["serverInfo"]["name"] == "nuwax_robot_bridge"

        tools = client.post(
            "/mcp",
            headers=_agent_headers(),
            json={"jsonrpc": "2.0", "id": "tools", "method": "tools/list"},
        )
        tool_names = {item["name"] for item in tools.json()["result"]["tools"]}
        assert "capture_image" in tool_names
        assert "describe_current_scene" in tool_names
        assert "get_joint_state" in tool_names
        assert "get_imu_state" in tool_names
        assert "get_localization_snapshot" in tool_names
        assert "get_map_snapshot" in tool_names
        assert "get_navigation_snapshot" in tool_names
        assert "navigate_to_pose" in tool_names
        assert "navigate_to_named_location" in tool_names
        assert "explore_area" in tool_names
        assert "relative_move" in tool_names
        assert "switch_control_mode" not in tool_names

        forbidden = client.post(
            "/api/capabilities/switch_control_mode/invoke",
            headers=_agent_headers(),
            json={"arguments": {"mode": "low"}},
        )
        assert forbidden.status_code == 403

        admin_ok = client.post(
            "/api/capabilities/switch_control_mode/invoke",
            headers=_admin_headers(),
            json={"arguments": {"mode": "low"}},
        )
        assert admin_ok.status_code == 200
        assert robot.current_mode == "low"

        move_response = client.post(
            "/api/capabilities/relative_move/invoke",
            headers=_agent_headers(),
            json={"arguments": {"vx": 0.2, "duration_sec": 0.08, "interval_sec": 0.01}},
        )
        assert move_response.status_code == 200
        task_id = move_response.json()["task"]["task_id"]
        task_payload = _wait_for_task(client, task_id)

        assert task_payload["status"]["state"] == "succeeded"
        assert robot.move_commands
        assert robot.stop_count >= 1


def test_gateway_websocket_and_sse_event_stream(tmp_path: Path) -> None:
    """网关应支持 WebSocket 实时事件和 SSE 回放。"""

    app, _, _, _ = _build_host_app(tmp_path)
    with TestClient(app) as client:
        trigger = client.post(
            "/api/capabilities/relative_move/invoke",
            headers=_agent_headers(),
            json={"arguments": {"vx": 0.1, "duration_sec": 0.05, "interval_sec": 0.01}},
        )
        assert trigger.status_code == 200

        response = client.get("/events/stream?after_cursor=0&max_events=10", headers=_agent_headers())
        assert response.status_code == 200
        found_sse = None
        for line in response.text.splitlines():
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            if payload.get("event_type") == "task.accepted":
                found_sse = payload
                break
        assert found_sse is not None

        with client.websocket_connect("/events/ws?access_token=agent-token") as websocket:
            client.post(
                "/api/capabilities/relative_move/invoke",
                headers=_agent_headers(),
                json={"arguments": {"vx": 0.1, "duration_sec": 0.05, "interval_sec": 0.01}},
            )
            found_ws = None
            for _ in range(20):
                message = websocket.receive_json()
                if message.get("type") != "event":
                    continue
                event = message.get("event", {})
                if event.get("event_type") in {"task.accepted", "task.started"}:
                    found_ws = event
                    break
            assert found_ws is not None


def test_gateway_artifact_fetch(tmp_path: Path) -> None:
    """图像能力应落制品并允许元数据与原始内容拉取。"""

    app, _, _, _ = _build_host_app(tmp_path)
    with TestClient(app) as client:
        capture = client.post(
            "/api/capabilities/capture_image/invoke",
            headers=_agent_headers(),
            json={"arguments": {}},
        )
        assert capture.status_code == 200
        artifact = capture.json()["result"]["artifact"]
        artifact_id = artifact["artifact_id"]

        meta = client.get(f"/artifacts/{artifact_id}/meta", headers=_agent_headers())
        raw = client.get(f"/artifacts/{artifact_id}", headers=_agent_headers())

        assert meta.status_code == 200
        assert meta.json()["artifact_id"] == artifact_id
        assert raw.status_code == 200
        assert raw.content == b"fake-jpeg-data"


def test_gateway_state_and_observation_endpoints(tmp_path: Path) -> None:
    """状态面和观察面接口应返回稳定快照。"""

    app, _, _, _ = _build_host_app(tmp_path)
    with TestClient(app) as client:
        state_latest = client.get("/api/state/latest", headers=_agent_headers())
        assert state_latest.status_code == 200
        assert state_latest.json()["robot_state"]["robot_id"] == "fake_go2"

        capture = client.post(
            "/api/capabilities/capture_image/invoke",
            headers=_agent_headers(),
            json={"arguments": {"camera_id": "front_camera"}},
        )
        assert capture.status_code == 200
        artifact_id = capture.json()["result"]["artifact"]["artifact_id"]

        latest_observation = client.get(
            "/api/observations/latest?camera_id=front_camera",
            headers=_agent_headers(),
        )
        history = client.get("/api/observations/history?limit=10", headers=_agent_headers())
        artifact_summary = client.get("/api/artifacts/summary", headers=_agent_headers())

        assert latest_observation.status_code == 200
        assert latest_observation.json()["observation_context"]["image_artifact"]["artifact_id"] == artifact_id
        assert history.status_code == 200
        assert history.json()["history"]
        assert artifact_summary.status_code == 200
        assert artifact_summary.json()["artifact_count"] >= 1


def test_gateway_ops_security_and_layout_endpoints(tmp_path: Path) -> None:
    """运维页、安全策略和目录布局接口应可直接用于排障。"""

    app, _, _, _ = _build_host_app(tmp_path)
    with TestClient(app) as client:
        security = client.get("/api/security/policy", headers=_agent_headers())
        layout = client.get("/api/deployment/layout", headers=_agent_headers())
        ops_page = client.get("/ops", headers=_agent_headers())

        assert security.status_code == 200
        assert "low" in security.json()["risk_policy"]
        assert layout.status_code == 200
        assert layout.json()["state_store_mode"] == "in_memory"
        assert ops_page.status_code == 200
        assert "nuwax_robot_bridge 运维页" in ops_page.text


def test_gateway_perception_capabilities_and_history(tmp_path: Path) -> None:
    """宿主机网关应暴露感知、摘要与历史查询接口。"""

    app, _, _, _ = _build_host_app(tmp_path)
    with TestClient(app) as client:
        describe = client.post(
            "/api/capabilities/describe_current_scene/invoke",
            headers=_agent_headers(),
            json={"arguments": {"camera_id": "front_camera"}},
        )
        assert describe.status_code == 200
        payload = describe.json()
        assert "person" in payload["result"]["summary"]
        assert payload["result"]["perception_context"]["scene_summary"]["detection_count"] == 2

        latest = client.get("/api/perception/latest", headers=_agent_headers())
        assert latest.status_code == 200
        latest_payload = latest.json()
        assert latest_payload["perception_context"]["scene_summary"]["active_track_count"] == 2

        history = client.get("/api/perception/history", headers=_agent_headers())
        assert history.status_code == 200
        history_payload = history.json()
        assert len(history_payload["history"]) >= 1
        assert history_payload["latest_contexts"][0]["detector_backend"] == "metadata_detector"


def test_gateway_localization_mapping_navigation_and_exploration(tmp_path: Path) -> None:
    """宿主机网关应暴露定位、地图、导航和探索主链路。"""

    app, _, robot, _ = _build_host_app(tmp_path)
    with TestClient(app) as client:
        localization = client.get("/api/localization/latest", headers=_agent_headers())
        assert localization.status_code == 200
        assert localization.json()["localization_snapshot"]["current_pose"]["position"]["x"] == 0.0

        localization_history = client.get("/api/localization/history?limit=10", headers=_agent_headers())
        assert localization_history.status_code == 200
        assert len(localization_history.json()["history"]) >= 1

        map_latest = client.get("/api/maps/latest", headers=_agent_headers())
        assert map_latest.status_code == 200
        assert map_latest.json()["map_snapshot"]["semantic_map"]["regions"][0]["region_id"] == "charging_station"

        navigate_pose = client.post(
            "/api/capabilities/navigate_to_pose/invoke",
            headers=_agent_headers(),
            json={"arguments": {"frame_id": "map", "x": 1.0, "y": 1.0, "yaw_rad": 0.0, "timeout_sec": 1.0}},
        )
        assert navigate_pose.status_code == 200
        navigate_pose_task = _wait_for_task(client, navigate_pose.json()["task"]["task_id"])
        assert navigate_pose_task["status"]["state"] == "succeeded"
        assert robot.current_pose.position.x == 1.0

        navigate_named = client.post(
            "/api/capabilities/navigate_to_named_location/invoke",
            headers=_agent_headers(),
            json={"arguments": {"target_name": "充电桩", "timeout_sec": 1.0}},
        )
        assert navigate_named.status_code == 200
        navigate_named_task = _wait_for_task(client, navigate_named.json()["task"]["task_id"])
        assert navigate_named_task["status"]["state"] == "succeeded"

        explore = client.post(
            "/api/capabilities/explore_area/invoke",
            headers=_agent_headers(),
            json={"arguments": {"target_name": "meeting_point", "strategy": "frontier", "radius_m": 1.5, "timeout_sec": 1.0}},
        )
        assert explore.status_code == 200
        explore_task = _wait_for_task(client, explore.json()["task"]["task_id"])
        assert explore_task["status"]["state"] == "succeeded"

        navigation_latest = client.get("/api/navigation/latest", headers=_agent_headers())
        navigation_history = client.get("/api/navigation/history?limit=10", headers=_agent_headers())

        assert navigation_latest.status_code == 200
        assert navigation_latest.json()["navigation_context"]["goal_reached"] is True
        assert navigation_latest.json()["exploration_context"]["exploration_state"]["status"] == "succeeded"
        assert navigation_history.status_code == 200
        assert len(navigation_history.json()["navigation_history"]) >= 2
        assert len(navigation_history.json()["exploration_history"]) >= 1


def test_gateway_memory_tools_and_named_navigation(tmp_path: Path) -> None:
    """地点记忆、语义记忆和命名导航应形成可复用链路。"""

    app, _, robot, _ = _build_host_app(tmp_path)
    with TestClient(app) as client:
        tools = client.get("/api/tools", headers=_agent_headers())
        admin_tools = client.get("/api/tools", headers=_admin_headers())

        assert tools.status_code == 200
        assert admin_tools.status_code == 200
        tool_names = {item["descriptor"]["name"] for item in tools.json()["tools"]}
        admin_tool_names = {item["descriptor"]["name"] for item in admin_tools.json()["tools"]}
        assert "tag_location" in tool_names
        assert "remember_current_scene" in tool_names
        assert "switch_control_mode" not in tool_names
        assert "switch_control_mode" in admin_tool_names

        nav_pose = client.post(
            "/api/capabilities/navigate_to_pose/invoke",
            headers=_agent_headers(),
            json={"arguments": {"frame_id": "map", "x": 1.2, "y": 0.4, "yaw_rad": 0.0, "timeout_sec": 1.0}},
        )
        assert nav_pose.status_code == 200
        nav_pose_payload = _wait_for_task(client, nav_pose.json()["task"]["task_id"])
        assert nav_pose_payload["status"]["state"] == "succeeded"

        describe = client.post(
            "/api/capabilities/describe_current_scene/invoke",
            headers=_agent_headers(),
            json={"arguments": {"camera_id": "front_camera"}},
        )
        assert describe.status_code == 200

        tag_location = client.post(
            "/api/capabilities/tag_location/invoke",
            headers=_agent_headers(),
            json={
                "arguments": {
                    "name": "补给点",
                    "aliases": ["充电位", "记忆补给位"],
                    "description": "前方补给位置。",
                    "camera_id": "front_camera",
                }
            },
        )
        assert tag_location.status_code == 200
        location_id = tag_location.json()["result"]["tagged_location"]["location_id"]

        remember_scene = client.post(
            "/api/capabilities/remember_current_scene/invoke",
            headers=_agent_headers(),
            json={"arguments": {"title": "人工巡检记录", "camera_id": "front_camera"}},
        )
        assert remember_scene.status_code == 200

        query_location = client.post(
            "/api/capabilities/query_location/invoke",
            headers=_agent_headers(),
            json={"arguments": {"query": "充电位", "similarity_threshold": 0.2}},
        )
        assert query_location.status_code == 200
        assert query_location.json()["result"]["query_result"]["matches"][0]["tagged_location"]["location_id"] == location_id

        query_memory = client.post(
            "/api/capabilities/query_semantic_memory/invoke",
            headers=_agent_headers(),
            json={"arguments": {"query": "person", "similarity_threshold": 0.1}},
        )
        assert query_memory.status_code == 200
        assert query_memory.json()["result"]["query_result"]["matches"]

        query_memory_with_visual_filter = client.post(
            "/api/capabilities/query_semantic_memory/invoke",
            headers=_agent_headers(),
            json={"arguments": {"query": "box", "similarity_threshold": 0.05, "visual_labels": ["box"]}},
        )
        assert query_memory_with_visual_filter.status_code == 200
        assert query_memory_with_visual_filter.json()["result"]["query_result"]["matches"]

        summary = client.get("/api/memory/summary", headers=_agent_headers())
        locations = client.get("/api/memory/locations?limit=10", headers=_agent_headers())
        semantic = client.get("/api/memory/semantic?limit=10", headers=_agent_headers())

        assert summary.status_code == 200
        assert summary.json()["memory_summary"]["tagged_location_count"] == 1
        assert locations.status_code == 200
        assert locations.json()["locations"][0]["name"] == "补给点"
        assert semantic.status_code == 200
        assert len(semantic.json()["entries"]) >= 2

        robot.current_pose = Pose(frame_id="map", position=Vector3(x=0.0, y=0.0, z=0.0))
        nav_named = client.post(
            "/api/capabilities/navigate_to_named_location/invoke",
            headers=_agent_headers(),
            json={"arguments": {"target_name": "记忆补给位", "timeout_sec": 1.0}},
        )
        assert nav_named.status_code == 200
        nav_named_payload = _wait_for_task(client, nav_named.json()["task"]["task_id"])
        assert nav_named_payload["status"]["state"] == "succeeded"
        assert nav_named_payload["result"]["arrival_verification"]["verified"] is True

        list_capabilities = client.post(
            "/mcp",
            headers=_agent_headers(),
            json={
                "jsonrpc": "2.0",
                "id": "list_capabilities",
                "method": "tools/call",
                "params": {"name": "list_capabilities", "arguments": {}},
            },
        )
        assert list_capabilities.status_code == 200
        structured = list_capabilities.json()["result"]["structuredContent"]["result"]
        skill_names = {item["descriptor"]["name"] for item in structured["skills"]}
        assert "tag_location" in skill_names
        assert "navigate_to_named_location" in skill_names


def test_gateway_audio_motion_and_task_tools(tmp_path: Path) -> None:
    """音频、动作、系统工具和任务式技能应能通过网关联动。"""

    app, _, robot, _ = _build_host_app(tmp_path)
    with TestClient(app) as client:
        tools = client.post(
            "/mcp",
            headers=_agent_headers(),
            json={"jsonrpc": "2.0", "id": "tools", "method": "tools/list"},
        )
        assert tools.status_code == 200
        tool_names = {item["name"] for item in tools.json()["result"]["tools"]}
        assert "speak_text" in tool_names
        assert "execute_sport_command" in tool_names
        assert "follow_target" in tool_names
        assert "inspect_target" in tool_names

        robot_status = client.post(
            "/mcp",
            headers=_agent_headers(),
            json={
                "jsonrpc": "2.0",
                "id": "status",
                "method": "tools/call",
                "params": {"name": "get_robot_status", "arguments": {}},
            },
        )
        assert robot_status.status_code == 200
        assert robot_status.json()["result"]["structuredContent"]["result"]["robot_state"]["robot_id"] == "fake_go2"

        joint_state = client.post(
            "/mcp",
            headers=_agent_headers(),
            json={
                "jsonrpc": "2.0",
                "id": "joints",
                "method": "tools/call",
                "params": {"name": "get_joint_state", "arguments": {"joint_name": "front_left_hip"}},
            },
        )
        assert joint_state.status_code == 200
        assert joint_state.json()["result"]["structuredContent"]["result"]["joint"]["name"] == "front_left_hip"

        imu_state = client.post(
            "/mcp",
            headers=_agent_headers(),
            json={
                "jsonrpc": "2.0",
                "id": "imu",
                "method": "tools/call",
                "params": {"name": "get_imu_state", "arguments": {}},
            },
        )
        assert imu_state.status_code == 200
        assert (
            imu_state.json()["result"]["structuredContent"]["result"]["imu_state"]["frame_id"]
            == "world/fake_go2/imu"
        )

        set_volume = client.post(
            "/mcp",
            headers=_agent_headers(),
            json={
                "jsonrpc": "2.0",
                "id": "set_volume",
                "method": "tools/call",
                "params": {"name": "set_volume", "arguments": {"volume": 0.8}},
            },
        )
        get_volume = client.post(
            "/mcp",
            headers=_agent_headers(),
            json={
                "jsonrpc": "2.0",
                "id": "get_volume",
                "method": "tools/call",
                "params": {"name": "get_volume", "arguments": {}},
            },
        )
        speak_text = client.post(
            "/mcp",
            headers=_agent_headers(),
            json={
                "jsonrpc": "2.0",
                "id": "speak",
                "method": "tools/call",
                "params": {"name": "speak_text", "arguments": {"text": "你好，开始巡检。"}},
            },
        )

        assert set_volume.status_code == 200
        assert get_volume.status_code == 200
        assert speak_text.status_code == 200
        assert robot.volume_ratio == 0.8
        assert (
            get_volume.json()["result"]["structuredContent"]["result"]["volume_state"]["robot_volume"]["volume"]
            == 0.8
        )
        assert speak_text.json()["result"]["structuredContent"]["result"]["speech"]["accepted"] is True

        execute_action = client.post(
            "/mcp",
            headers=_agent_headers(),
            json={
                "jsonrpc": "2.0",
                "id": "sport",
                "method": "tools/call",
                "params": {"name": "execute_sport_command", "arguments": {"action": "stand_up", "params": {"speed": 1}}},
            },
        )
        set_body_pose = client.post(
            "/mcp",
            headers=_agent_headers(),
            json={
                "jsonrpc": "2.0",
                "id": "pose",
                "method": "tools/call",
                "params": {"name": "set_body_pose", "arguments": {"roll": 0.1, "pitch": 0.0, "yaw": 0.2}},
            },
        )
        set_speed = client.post(
            "/mcp",
            headers=_agent_headers(),
            json={
                "jsonrpc": "2.0",
                "id": "speed",
                "method": "tools/call",
                "params": {"name": "set_speed_level", "arguments": {"level": 2}},
            },
        )

        assert execute_action.status_code == 200
        assert set_body_pose.status_code == 200
        assert set_speed.status_code == 200
        assert [item[0] for item in robot.action_history][-3:] == ["stand_up", "euler", "speed_level"]

        navigation_task = client.post(
            "/api/capabilities/navigate_to_pose/invoke",
            headers=_agent_headers(),
            json={
                "arguments": {
                    "frame_id": "map",
                    "x": 99.0,
                    "y": 0.0,
                    "poll_interval_sec": 0.02,
                    "timeout_sec": 1.0,
                }
            },
        )
        assert navigation_task.status_code == 200
        navigation_task_id = navigation_task.json()["task"]["task_id"]
        time.sleep(0.08)

        active_tasks = client.post(
            "/mcp",
            headers=_agent_headers(),
            json={
                "jsonrpc": "2.0",
                "id": "active",
                "method": "tools/call",
                "params": {"name": "get_active_tasks", "arguments": {}},
            },
        )
        cancel_task = client.post(
            "/mcp",
            headers=_agent_headers(),
            json={
                "jsonrpc": "2.0",
                "id": "cancel",
                "method": "tools/call",
                "params": {"name": "cancel_task", "arguments": {"task_id": navigation_task_id}},
            },
        )
        assert active_tasks.status_code == 200
        assert any(
            item["task_id"] == navigation_task_id
            for item in active_tasks.json()["result"]["structuredContent"]["result"]["active_tasks"]
        )
        assert cancel_task.status_code == 200
        cancelled_payload = _wait_for_task(client, navigation_task_id)
        assert cancelled_payload["status"]["state"] == "cancelled"
        _wait_for_condition(lambda: client.get("/api/tasks", headers=_agent_headers()).json()["active_tasks"] == [])

        stop_all = client.post(
            "/mcp",
            headers=_agent_headers(),
            json={
                "jsonrpc": "2.0",
                "id": "stop",
                "method": "tools/call",
                "params": {"name": "stop_all_motion", "arguments": {"reason": "单元测试触发停止。"}},
            },
        )
        assert stop_all.status_code == 200

        follow_ok = client.post(
            "/api/capabilities/follow_target/invoke",
            headers=_agent_headers(),
            json={"arguments": {"query": "person", "duration_sec": 0.18, "interval_sec": 0.03, "timeout_sec": 1.0}},
        )
        assert follow_ok.status_code == 200
        follow_ok_payload = _wait_for_task(client, follow_ok.json()["task"]["task_id"])
        assert follow_ok_payload["status"]["state"] == "succeeded"
        assert robot.move_commands
        assert robot.stop_count >= 1

        inspect = client.post(
            "/api/capabilities/inspect_target/invoke",
            headers=_agent_headers(),
            json={"arguments": {"query": "person", "camera_id": "front_camera", "timeout_sec": 1.0}},
        )
        assert inspect.status_code == 200
        inspect_payload = _wait_for_task(client, inspect.json()["task"]["task_id"])
        assert inspect_payload["status"]["state"] == "succeeded"

        memory_query = client.post(
            "/api/capabilities/query_semantic_memory/invoke",
            headers=_agent_headers(),
            json={"arguments": {"query": "person", "similarity_threshold": 0.1}},
        )
        assert memory_query.status_code == 200
        assert memory_query.json()["result"]["query_result"]["matches"]


def test_gateway_navigation_task_cancel_and_timeout(tmp_path: Path) -> None:
    """导航任务应支持取消与超时终止。"""

    app, _, robot, _ = _build_host_app(tmp_path)
    with TestClient(app) as client:
        cancel_response = client.post(
            "/api/capabilities/navigate_to_pose/invoke",
            headers=_agent_headers(),
            json={"arguments": {"frame_id": "map", "x": 99.0, "y": 0.0, "poll_interval_sec": 0.02, "timeout_sec": 1.0}},
        )
        assert cancel_response.status_code == 200
        cancel_task_id = cancel_response.json()["task"]["task_id"]

        time.sleep(0.05)
        cancel_request = client.post(f"/api/tasks/{cancel_task_id}/cancel", headers=_agent_headers())
        assert cancel_request.status_code == 200
        cancelled_payload = _wait_for_task(client, cancel_task_id)
        assert cancelled_payload["status"]["state"] == "cancelled"
        _wait_for_condition(lambda: robot.nav_cancelled is True)

        timeout_response = client.post(
            "/api/capabilities/navigate_to_pose/invoke",
            headers=_agent_headers(),
            json={"arguments": {"frame_id": "map", "x": 99.0, "y": 0.0, "poll_interval_sec": 0.02, "timeout_sec": 0.2}},
        )
        assert timeout_response.status_code == 200
        timeout_task_id = timeout_response.json()["task"]["task_id"]
        timeout_payload = _wait_for_task(client, timeout_task_id)
        assert timeout_payload["status"]["state"] == "timeout"
        _wait_for_condition(lambda: robot.nav_cancelled is True)


def test_relay_forwards_mcp_and_artifacts(tmp_path: Path) -> None:
    """边车转发器应能把 MCP 和制品请求转发到宿主机网关。"""

    host_app, runtime, _, config = _build_host_app(tmp_path, start_runtime_on_lifespan=False)
    runtime.start()
    try:
        relay_app = create_relay_app(
            config.relay,
            transport=httpx.ASGITransport(app=host_app),
        )
        with TestClient(relay_app) as client:
            initialize = client.post(
                "/mcp",
                headers=_relay_headers(),
                json={"jsonrpc": "2.0", "id": "init", "method": "initialize"},
            )
            assert initialize.status_code == 200

            capture = client.post(
                "/mcp",
                headers=_relay_headers(),
                json={
                    "jsonrpc": "2.0",
                    "id": "capture",
                    "method": "tools/call",
                    "params": {"name": "capture_image", "arguments": {}},
                },
            )
            structured = capture.json()["result"]["structuredContent"]
            artifact_id = structured["result"]["artifact"]["artifact_id"]

            raw = client.get(f"/artifacts/{artifact_id}", headers=_relay_headers())
            assert raw.status_code == 200
            assert raw.content == b"fake-jpeg-data"
    finally:
        runtime.stop()
