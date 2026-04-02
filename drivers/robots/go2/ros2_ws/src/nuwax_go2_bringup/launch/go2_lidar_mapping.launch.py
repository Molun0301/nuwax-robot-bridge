from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    lidar_port_arg = DeclareLaunchArgument("lidar_port", default_value="/dev/ttyUSB0")
    lidar_yaw_bias_arg = DeclareLaunchArgument("lidar_yaw_bias", default_value="0.0")
    point_lio_config_arg = DeclareLaunchArgument(
        "point_lio_config",
        default_value=PathJoinSubstitution(
            [FindPackageShare("point_lio"), "config", "unilidar_l2.yaml"]
        ),
    )
    elevation_config_arg = DeclareLaunchArgument(
        "elevation_config",
        default_value=PathJoinSubstitution(
            [FindPackageShare("nuwax_go2_bringup"), "config", "go2_elevation_mapping.yaml"]
        ),
    )
    core_param_arg = DeclareLaunchArgument(
        "elevation_core_config",
        default_value=PathJoinSubstitution(
            [FindPackageShare("elevation_mapping_cupy"), "config", "core", "core_param.yaml"]
        ),
    )

    unitree_lidar_node = Node(
        package="unitree_lidar_ros2",
        executable="unitree_lidar_ros2_node",
        name="unitree_lidar_ros2_node",
        output="screen",
        parameters=[
            {
                "port": LaunchConfiguration("lidar_port"),
                "rotate_yaw_bias": LaunchConfiguration("lidar_yaw_bias"),
                "range_scale": 0.001,
                "range_bias": 0.0,
                "range_max": 50.0,
                "range_min": 0.0,
                "cloud_frame": "unilidar_lidar",
                "cloud_topic": "unilidar/cloud",
                "cloud_scan_num": 18,
                "imu_frame": "unilidar_imu",
                "imu_topic": "unilidar/imu",
            }
        ],
    )

    point_lio_node = Node(
        package="point_lio",
        executable="pointlio_mapping",
        name="go2_point_lio",
        output="screen",
        parameters=[
            LaunchConfiguration("point_lio_config"),
            {
                "odom_only": False,
                "odom_header_frame_id": "odom",
                "odom_child_frame_id": "body",
                "publish.scan_publish_en": True,
                "publish.scan_bodyframe_pub_en": True,
            },
        ],
        remappings=[
            ("/aft_mapped_to_init", "/odom_lio"),
        ],
    )

    elevation_mapping_node = Node(
        package="elevation_mapping_cupy",
        executable="elevation_mapping_node.py",
        name="elevation_mapping_node",
        output="screen",
        parameters=[
            LaunchConfiguration("elevation_core_config"),
            LaunchConfiguration("elevation_config"),
            {"use_sim_time": False},
        ],
    )

    return LaunchDescription(
        [
            lidar_port_arg,
            lidar_yaw_bias_arg,
            point_lio_config_arg,
            elevation_config_arg,
            core_param_arg,
            unitree_lidar_node,
            point_lio_node,
            elevation_mapping_node,
        ]
    )
