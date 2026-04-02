from __future__ import annotations

from collections import deque

from contracts.events import RuntimeEventCategory
from contracts.runtime_views import LocalizationSnapshot
from core import EventBus, StateNamespace, StateStore
from gateways.errors import GatewayError
from providers import LocalizationProvider
from typing import Deque, Dict, Optional, Tuple


class LocalizationService:
    """统一定位与 TF 快照服务。"""

    def __init__(
        self,
        *,
        provider_owner,
        state_store: StateStore,
        event_bus: Optional[EventBus] = None,
        history_limit: int = 100,
    ) -> None:
        self._provider_owner = provider_owner
        self._state_store = state_store
        self._event_bus = event_bus
        self._history: Deque[LocalizationSnapshot] = deque(maxlen=max(1, history_limit))
        self._latest_snapshot: Optional[LocalizationSnapshot] = None

    def refresh(self) -> LocalizationSnapshot:
        """从本地定位提供器刷新定位快照。"""

        provider = self._get_localization_provider()
        return self.ingest_external_snapshot(
            source_name=provider.provider_name,
            current_pose=provider.get_current_pose(),
            frame_tree=provider.get_frame_tree(),
            metadata={"source_kind": "provider", "provider_version": provider.provider_version},
        )

    def ingest_external_snapshot(
        self,
        *,
        source_name: str,
        current_pose=None,
        frame_tree=None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> LocalizationSnapshot:
        """写入来自适配器或外部系统的定位快照。"""

        snapshot = LocalizationSnapshot(
            source_name=source_name,
            current_pose=current_pose,
            frame_tree=frame_tree,
            metadata=dict(metadata or {}),
        )
        self._latest_snapshot = snapshot
        self._history.append(snapshot)
        self._state_store.write(
            StateNamespace.LOCALIZATION,
            source_name,
            snapshot,
            source=source_name,
            metadata={"kind": "latest_localization"},
        )

        if self._event_bus is not None:
            self._event_bus.publish(
                self._event_bus.build_event(
                    "localization.snapshot_ready",
                    category=RuntimeEventCategory.ROBOT,
                    source="localization_service",
                    message="定位快照已更新。",
                    payload={"source_name": source_name},
                )
            )
        return snapshot

    def get_latest_snapshot(self) -> Optional[LocalizationSnapshot]:
        """返回最新定位快照。"""

        return self._latest_snapshot

    def list_history(self, *, limit: Optional[int] = None) -> Tuple[LocalizationSnapshot, ...]:
        """返回定位历史。"""

        items = list(self._history)
        if limit is not None:
            items = items[-max(0, limit) :]
        return tuple(items)

    def is_available(self) -> bool:
        """判断定位服务当前是否可用。"""

        return self._latest_snapshot is not None or self._find_localization_provider() is not None

    def _get_localization_provider(self) -> LocalizationProvider:
        provider = self._find_localization_provider()
        if provider is None:
            raise GatewayError("当前机器人入口未提供定位提供器。")
        if not provider.is_available():
            raise GatewayError("当前定位提供器暂不可用。")
        return provider

    def _find_localization_provider(self) -> Optional[LocalizationProvider]:
        providers = getattr(self._provider_owner, "providers", None)
        if providers is None or not isinstance(providers, LocalizationProvider):
            return None
        return providers
