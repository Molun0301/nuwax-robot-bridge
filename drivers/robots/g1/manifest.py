from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from drivers.robots.common.manifest import ComponentBinding, RobotManifest
from drivers.robots.g1.capabilities import G1_CAPABILITY_MATRIX

if TYPE_CHECKING:
    from settings import NuwaxRobotBridgeConfig


def build_g1_manifest(config: Optional["NuwaxRobotBridgeConfig"] = None) -> RobotManifest:
    """构造 G1 入口骨架清单。"""

    del config
    return RobotManifest(
        robot_name="g1",
        robot_model="unitree_g1",
        entrypoint="drivers/robots/g1/assembly.py:create_g1_assembly",
        description="G1 机器人入口骨架，当前仅预留接入格式，不提供具体功能。",
        capability_matrix=G1_CAPABILITY_MATRIX,
        required_components=(
            ComponentBinding(
                name="body_driver",
                path="drivers/robots/g1",
                description="G1 本体驱动入口占位。",
            ),
        ),
    )


G1_MANIFEST = build_g1_manifest()
