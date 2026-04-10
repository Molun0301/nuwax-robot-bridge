from __future__ import annotations

from contracts.geometry import FrameTree, Pose, Quaternion, Transform, Vector3
from contracts.maps import CostMap, OccupancyGrid, SemanticMap, SemanticRegion
from contracts.navigation import ExplorationState, ExplorationStatus, ExploreAreaRequest, NavigationGoal, NavigationState, NavigationStatus
from contracts.runtime_views import LocalizationSessionStatus
from core import EventBus, StateNamespace, StateStore
from providers import ExplorationProvider, LocalizationProvider, MapProvider, NavigationProvider
from services import LocalizationService, MappingService, NavigationService
from typing import Optional


class _ProviderOwner:
    """测试用提供器宿主。"""

    def __init__(self, providers) -> None:
        self.providers = providers


class FakeNavigationBundle(LocalizationProvider, MapProvider, NavigationProvider, ExplorationProvider):
    """导航服务测试用组合后端。"""

    provider_name = "fake_navigation_bundle"
    provider_version = "0.1.0"

    def __init__(self) -> None:
        self.current_pose = Pose(frame_id="map", position=Vector3(x=0.0, y=0.0, z=0.0))
        self.semantic_map = self._build_semantic_map(1.0)
        self.active_goal: Optional[NavigationGoal] = None
        self.last_goal_id: Optional[str] = None
        self.nav_poll_count = 0
        self.nav_cancelled = False
        self.active_request: Optional[ExploreAreaRequest] = None
        self.last_request_id: Optional[str] = None
        self.explore_poll_count = 0
        self.explore_cancelled = False
        self.localization_available = True
        self.map_available = True
        self.navigation_available = True
        self.exploration_available = True

    def is_available(self) -> bool:
        return True

    def is_localization_available(self) -> bool:
        return self.localization_available

    def is_map_available(self) -> bool:
        return self.map_available

    def is_navigation_available(self) -> bool:
        return self.navigation_available

    def is_exploration_available(self) -> bool:
        return self.exploration_available

    def get_current_pose(self) -> Optional[Pose]:
        return self.current_pose

    def get_frame_tree(self) -> Optional[FrameTree]:
        return FrameTree(
            root_frame_id="world",
            transforms=[
                Transform(
                    parent_frame_id="world",
                    child_frame_id="map",
                    translation=Vector3(),
                    rotation=Quaternion(w=1.0),
                    authority="test_localization",
                ),
                Transform(
                    parent_frame_id="map",
                    child_frame_id="base_link",
                    translation=self.current_pose.position,
                    rotation=self.current_pose.orientation,
                    authority="test_localization",
                ),
            ],
        )

    def get_occupancy_grid(self) -> Optional[OccupancyGrid]:
        return OccupancyGrid(
            map_id="test_map",
            frame_id="map",
            width=2,
            height=2,
            resolution_m=0.1,
            origin=Pose(frame_id="map", position=Vector3()),
            data=[0, 0, 10, 100],
        )

    def get_cost_map(self) -> Optional[CostMap]:
        return CostMap(
            map_id="test_cost_map",
            frame_id="map",
            width=2,
            height=2,
            resolution_m=0.1,
            origin=Pose(frame_id="map", position=Vector3()),
            data=[1.0, 5.0, 15.0, 100.0],
        )

    def get_semantic_map(self) -> Optional[SemanticMap]:
        return self.semantic_map

    def set_goal(self, goal: NavigationGoal) -> bool:
        self.active_goal = goal
        self.last_goal_id = goal.goal_id
        self.nav_poll_count = 0
        self.nav_cancelled = False
        return True

    def cancel_goal(self) -> bool:
        self.nav_cancelled = True
        self.active_goal = None
        return True

    def get_navigation_state(self) -> NavigationState:
        if self.nav_cancelled:
            return NavigationState(
                current_goal_id=self.last_goal_id,
                status=NavigationStatus.CANCELLED,
                current_pose=self.current_pose,
                message="测试后端已取消导航。",
            )
        if self.active_goal is None:
            return NavigationState(
                current_goal_id=self.last_goal_id,
                status=NavigationStatus.IDLE,
                current_pose=self.current_pose,
            )

        self.nav_poll_count += 1
        target_pose = self.active_goal.target_pose or self.current_pose
        if self.nav_poll_count >= 2:
            self.current_pose = target_pose
            self.active_goal = None
            return NavigationState(
                current_goal_id=self.last_goal_id,
                status=NavigationStatus.SUCCEEDED,
                current_pose=self.current_pose,
                remaining_distance_m=0.0,
                goal_reached=True,
                message="测试后端已到达。",
            )
        return NavigationState(
            current_goal_id=self.last_goal_id,
            status=NavigationStatus.RUNNING,
            current_pose=self.current_pose,
            remaining_distance_m=1.0,
            message="测试后端导航中。",
        )

    def is_goal_reached(self) -> bool:
        return self.active_goal is None and not self.nav_cancelled and self.nav_poll_count >= 2

    def start_exploration(self, request: ExploreAreaRequest) -> bool:
        self.active_request = request
        self.last_request_id = request.request_id
        self.explore_poll_count = 0
        self.explore_cancelled = False
        return True

    def stop_exploration(self) -> bool:
        self.explore_cancelled = True
        self.active_request = None
        return True

    def get_exploration_state(self) -> ExplorationState:
        if self.explore_cancelled:
            return ExplorationState(
                current_request_id=self.last_request_id,
                status=ExplorationStatus.CANCELLED,
                strategy="frontier",
                message="测试后端已取消探索。",
            )
        if self.active_request is None:
            return ExplorationState(
                current_request_id=self.last_request_id,
                status=ExplorationStatus.IDLE,
            )

        self.explore_poll_count += 1
        if self.explore_poll_count >= 2:
            self.active_request = None
            return ExplorationState(
                current_request_id=self.last_request_id,
                status=ExplorationStatus.SUCCEEDED,
                strategy="frontier",
                covered_ratio=1.0,
                frontier_count=0,
                message="测试后端已完成探索。",
            )
        return ExplorationState(
            current_request_id=self.last_request_id,
            status=ExplorationStatus.RUNNING,
            strategy="frontier",
            covered_ratio=0.5,
            frontier_count=2,
            message="测试后端探索中。",
        )

    def _build_semantic_map(self, dock_x: float) -> SemanticMap:
        return SemanticMap(
            map_id="semantic_map",
            frame_id="map",
            regions=[
                SemanticRegion(
                    region_id="dock",
                    label="dock",
                    centroid=Pose(frame_id="map", position=Vector3(x=dock_x, y=0.5, z=0.0)),
                    attributes={"alias": "充电桩", "aliases": ["charging dock"]},
                )
            ],
        )


def _build_navigation_services():
    providers = FakeNavigationBundle()
    provider_owner = _ProviderOwner(providers)
    state_store = StateStore()
    event_bus = EventBus()
    localization_service = LocalizationService(
        provider_owner=provider_owner,
        state_store=state_store,
        event_bus=event_bus,
        history_limit=10,
    )
    mapping_service = MappingService(
        provider_owner=provider_owner,
        state_store=state_store,
        event_bus=event_bus,
        history_limit=10,
    )
    navigation_service = NavigationService(
        provider_owner=provider_owner,
        localization_service=localization_service,
        mapping_service=mapping_service,
        state_store=state_store,
        event_bus=event_bus,
        history_limit=10,
    )
    return providers, state_store, localization_service, mapping_service, navigation_service


def test_localization_and_mapping_services_write_latest_state_and_versions() -> None:
    """定位与地图服务应写入状态缓存并维护版本号。"""

    _, state_store, localization_service, mapping_service, _ = _build_navigation_services()

    localization_snapshot = localization_service.refresh()
    map_snapshot = mapping_service.refresh()
    external_map_snapshot = mapping_service.ingest_external_snapshot(
        source_name="cartographer",
        occupancy_grid=OccupancyGrid(
            map_id="external_map",
            frame_id="map",
            width=2,
            height=2,
            resolution_m=0.1,
            origin=Pose(frame_id="map", position=Vector3()),
            data=[0, 0, 0, 0],
        ),
        semantic_map=SemanticMap(
            map_id="external_semantic_map",
            frame_id="map",
            regions=[
                SemanticRegion(
                    region_id="dock",
                    label="dock",
                    centroid=Pose(frame_id="map", position=Vector3(x=2.0, y=0.5, z=0.0)),
                    attributes={"alias": "充电桩"},
                )
            ],
        ),
    )

    latest_localization = state_store.read_latest(StateNamespace.LOCALIZATION)
    latest_map = state_store.read_latest(StateNamespace.MAP)

    assert localization_snapshot.current_pose is not None
    assert latest_localization is not None
    assert latest_localization.value.source_name == "fake_navigation_bundle"
    assert map_snapshot.version_id.startswith("mapv_fake_navigation_bundle_")
    assert external_map_snapshot.version_id.startswith("mapv_cartographer_")
    assert latest_map is not None
    assert latest_map.value.source_name == "cartographer"


def test_localization_service_accepts_scoped_map_frame_alias() -> None:
    """定位会话应接受 scoped map frame（带路径地图坐标系）作为语义等价坐标系。"""

    providers, _, localization_service, _, _ = _build_navigation_services()
    providers.current_pose = Pose(
        frame_id="world/fake_navigation/map",
        position=Vector3(x=1.0, y=2.0, z=0.0),
        orientation=Quaternion(w=1.0),
    )

    session = localization_service.refresh_active_session(
        map_name="测试地图",
        map_version_id="mapver_000001",
        frame_id="map",
    )

    assert session.status == LocalizationSessionStatus.READY
    assert session.pose_available is True
    assert session.last_error is None


def test_navigation_service_supports_named_goal_resolution_after_map_source_switch() -> None:
    """切换地图来源后，导航服务仍应复用同一套命名导航逻辑。"""

    providers, _, localization_service, mapping_service, navigation_service = _build_navigation_services()
    localization_service.refresh()
    mapping_service.refresh()

    first_goal = navigation_service.resolve_named_goal("充电桩")
    first_context = navigation_service.navigate_until_complete(first_goal, poll_interval_sec=0.01)

    providers.semantic_map = providers._build_semantic_map(2.5)
    mapping_service.ingest_external_snapshot(
        source_name="external_slam_b",
        occupancy_grid=providers.get_occupancy_grid(),
        cost_map=providers.get_cost_map(),
        semantic_map=providers.get_semantic_map(),
    )
    second_goal = navigation_service.resolve_named_goal("charging dock")
    second_context = navigation_service.navigate_until_complete(second_goal, poll_interval_sec=0.01)

    assert first_context.goal_reached is True
    assert second_goal.target_pose is not None
    assert second_goal.target_pose.position.x == 2.5
    assert second_context.navigation_state.status == NavigationStatus.SUCCEEDED
    assert len(navigation_service.list_navigation_history()) >= 2


def test_navigation_service_exploration_flow_reuses_same_runtime_boundary() -> None:
    """探索服务应复用统一状态缓存与终态语义。"""

    _, state_store, localization_service, mapping_service, navigation_service = _build_navigation_services()
    localization_service.refresh()
    mapping_service.refresh()

    context = navigation_service.explore_until_complete(
        ExploreAreaRequest(request_id="explore_001", target_name="dock", strategy="frontier"),
        poll_interval_sec=0.01,
    )
    latest_navigation_state = state_store.read(StateNamespace.NAVIGATION, "exploration")

    assert context.exploration_state.status == ExplorationStatus.SUCCEEDED
    assert latest_navigation_state is not None
    assert latest_navigation_state.value.exploration_state.covered_ratio == 1.0
    assert len(navigation_service.list_exploration_history()) >= 1


def test_services_consult_provider_specific_availability_flags() -> None:
    """服务可用性应优先反映提供器的真实运行状态。"""

    providers, _, localization_service, mapping_service, navigation_service = _build_navigation_services()
    providers.localization_available = False
    providers.map_available = False
    providers.navigation_available = False
    providers.exploration_available = False

    assert localization_service.is_available() is False
    assert mapping_service.is_available() is False
    assert navigation_service.is_navigation_available() is False
    assert navigation_service.is_exploration_available() is False
