from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=PathJoinSubstitution(
            [FindPackageShare("nuwax_go2_bringup"), "config", "go2_nav2.yaml"]
        ),
    )
    use_sim_time_arg = DeclareLaunchArgument("use_sim_time", default_value="false")
    autostart_arg = DeclareLaunchArgument("autostart", default_value="true")
    dds_iface_arg = DeclareLaunchArgument(
        "dds_iface",
        default_value=EnvironmentVariable("GO2_DATA_PLANE_DDS_IFACE", default_value=""),
    )
    cmd_vel_topic_arg = DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel")

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("nav2_bringup"), "launch", "navigation_launch.py"])
        ),
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "params_file": LaunchConfiguration("params_file"),
            "autostart": LaunchConfiguration("autostart"),
        }.items(),
    )

    cmd_vel_bridge_node = Node(
        package="nuwax_go2_bringup",
        executable="go2_cmd_vel_bridge",
        name="go2_cmd_vel_bridge",
        output="screen",
        parameters=[
            {
                "dds_iface": LaunchConfiguration("dds_iface"),
                "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
            }
        ],
    )

    return LaunchDescription(
        [
            params_file_arg,
            use_sim_time_arg,
            autostart_arg,
            dds_iface_arg,
            cmd_vel_topic_arg,
            cmd_vel_bridge_node,
            nav2_launch,
        ]
    )
