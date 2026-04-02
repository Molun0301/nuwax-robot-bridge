from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    lidar_mapping_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("nuwax_go2_bringup"), "launch", "go2_lidar_mapping.launch.py"])
        )
    )
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("nuwax_go2_bringup"), "launch", "go2_nav2.launch.py"])
        )
    )
    return LaunchDescription([lidar_mapping_launch, nav2_launch])
