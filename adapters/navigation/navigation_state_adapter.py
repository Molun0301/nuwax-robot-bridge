from __future__ import annotations

from typing import Dict, Optional, Any

from adapters.base import AdapterBase, AdapterCategory, AdapterConfig
from adapters.helpers import resolve_timestamp
from contracts.navigation import NavigationState, NavigationStatus


class NavigationStateAdapter(AdapterBase[Dict[str, Any], NavigationState]):
    """外部导航状态适配器模板。"""

    @classmethod
    def build_default_config(
        cls,
        name: str = "navigation_state",
        *,
        source_ref: Optional[str] = "/navigation/state",
    ) -> AdapterConfig:
        return AdapterConfig(
            name=name,
            category=AdapterCategory.NAVIGATION,
            source_kind="navigation",
            contract_type="NavigationState",
            source_ref=source_ref,
            settings={"default_status": "idle"},
        )

    def convert_payload(self, payload: Dict[str, Any]) -> NavigationState:
        status = self._normalize_status(
            str(payload.get("status") or self.config.settings.get("default_status") or "idle")
        )
        return NavigationState(
            timestamp=resolve_timestamp(payload),
            current_goal_id=payload.get("current_goal_id"),
            status=status,
            remaining_distance_m=(
                float(payload["remaining_distance_m"]) if payload.get("remaining_distance_m") is not None else None
            ),
            remaining_yaw_rad=(
                float(payload["remaining_yaw_rad"]) if payload.get("remaining_yaw_rad") is not None else None
            ),
            goal_reached=bool(payload.get("goal_reached", False)),
            message=payload.get("message"),
            metadata=dict(payload.get("metadata") or {}),
        )

    def _normalize_status(self, raw_status: str) -> NavigationStatus:
        aliases = {
            "idle": NavigationStatus.IDLE,
            "accepted": NavigationStatus.ACCEPTED,
            "planning": NavigationStatus.PLANNING,
            "running": NavigationStatus.RUNNING,
            "succeeded": NavigationStatus.SUCCEEDED,
            "failed": NavigationStatus.FAILED,
            "cancelled": NavigationStatus.CANCELLED,
            "canceled": NavigationStatus.CANCELLED,
        }
        normalized = raw_status.strip().lower()
        if normalized not in aliases:
            raise ValueError(f"不支持的导航状态: {raw_status}")
        return aliases[normalized]
