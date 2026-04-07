from launch import LaunchDescription
from launch.actions import LogInfo


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            LogInfo(
                msg="nuwax_go2_bringup 已移除旧的串口雷达与本地 Nav2 启动链，请直接使用 drivers/robots/go2/data_plane_entry。"
            )
        ]
    )
