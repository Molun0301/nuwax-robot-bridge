"""机器人入口公共模型导出。"""

from drivers.robots.common.assembly import RobotAssemblyBase, RobotAssemblyStatus
from drivers.robots.common.manifest import ComponentBinding, RobotDefaults, RobotManifest

__all__ = [
    "ComponentBinding",
    "RobotAssemblyBase",
    "RobotAssemblyStatus",
    "RobotDefaults",
    "RobotManifest",
]
