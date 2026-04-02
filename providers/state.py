from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from contracts.robot_state import IMUState, JointState, RobotState
from providers.base import BaseProvider


@runtime_checkable
class StateProvider(BaseProvider, Protocol):
    """机器人状态读取接口。"""

    def get_robot_state(self) -> RobotState:
        """返回当前机器人状态快照。"""

    def get_joint_states(self) -> List[JointState]:
        """返回全部关节状态。"""

    def get_imu_state(self) -> Optional[IMUState]:
        """返回 IMU 状态。"""

