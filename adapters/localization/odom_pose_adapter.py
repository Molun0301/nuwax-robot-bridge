from __future__ import annotations

from typing import Dict, Optional, Any

from adapters.base import AdapterBase, AdapterCategory, AdapterConfig
from adapters.helpers import build_pose, build_transform, resolve_timestamp
from contracts.geometry import Pose, Transform
from contracts.navigation import NavigationState, NavigationStatus


class OdomPoseAdapter(AdapterBase[Dict[str, Any], Pose]):
    """里程计位姿适配器模板。"""

    @classmethod
    def build_default_config(
        cls,
        name: str = "odom_pose",
        *,
        source_ref: Optional[str] = "/odom",
    ) -> AdapterConfig:
        """构造默认配置。"""

        return AdapterConfig(
            name=name,
            category=AdapterCategory.LOCALIZATION,
            source_kind="odometry",
            contract_type="Pose",
            source_ref=source_ref,
            settings={
                "default_frame_id": "odom",
                "default_child_frame_id": "base",
                "default_navigation_status": "running",
            },
        )

    def convert_payload(self, payload: Dict[str, Any]) -> Pose:
        """把外部定位数据转换成位姿契约。"""

        timestamp = resolve_timestamp(payload)
        pose_payload = payload.get("pose") or payload
        frame_id = str(payload.get("frame_id") or self.config.settings.get("default_frame_id") or "odom")
        return build_pose(pose_payload, frame_id=frame_id, timestamp=timestamp)

    def build_transform_contract(self, payload: Dict[str, Any]) -> Transform:
        """额外导出对应的坐标变换。"""

        timestamp = resolve_timestamp(payload)
        pose_payload = payload.get("pose") or payload
        parent_frame_id = str(payload.get("frame_id") or self.config.settings.get("default_frame_id") or "odom")
        child_frame_id = str(
            payload.get("child_frame_id") or self.config.settings.get("default_child_frame_id") or "base"
        )
        return build_transform(
            pose_payload,
            parent_frame_id=parent_frame_id,
            child_frame_id=child_frame_id,
            timestamp=timestamp,
        )

    def build_navigation_state(self, payload: Dict[str, Any]) -> NavigationState:
        """额外导出导航状态快照。"""

        status = self._normalize_status(
            str(payload.get("status") or self.config.settings.get("default_navigation_status") or "running")
        )
        return NavigationState(
            timestamp=resolve_timestamp(payload),
            current_goal_id=payload.get("current_goal_id"),
            status=status,
            current_pose=self.convert_payload(payload),
            remaining_distance_m=(
                float(payload["remaining_distance_m"]) if payload.get("remaining_distance_m") is not None else None
            ),
            remaining_yaw_rad=(
                float(payload["remaining_yaw_rad"]) if payload.get("remaining_yaw_rad") is not None else None
            ),
            message=payload.get("message"),
        )

    def _normalize_status(self, raw_status: str) -> NavigationStatus:
        normalized = raw_status.strip().lower()
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
        if normalized not in aliases:
            raise ValueError(f"不支持的导航状态: {raw_status}")
        return aliases[normalized]
