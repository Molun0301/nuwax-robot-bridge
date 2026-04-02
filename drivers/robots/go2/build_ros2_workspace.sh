#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WS_DIR="${REPO_ROOT}/drivers/robots/go2/ros2_ws"

detect_ros_setup() {
  local candidates=(
    "/opt/ros/humble/setup.bash"
    "/opt/ros/jazzy/setup.bash"
    "/opt/ros/iron/setup.bash"
    "/opt/ros/foxy/setup.bash"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

ROS_SETUP="$(detect_ros_setup || true)"
if [[ -z "${ROS_SETUP}" ]]; then
  echo "未找到 ROS2 安装，请先安装 ROS2 Humble 或兼容发行版。" >&2
  exit 1
fi

source "${ROS_SETUP}"
cd "${WS_DIR}"

echo "使用 ROS2 环境: ${ROS_SETUP}"
echo "开始构建 Go2 端侧工作空间: ${WS_DIR}"

colcon build \
  --symlink-install \
  --base-paths src \
  --packages-up-to \
    grid_map_cmake_helpers \
    grid_map_msgs \
    elevation_map_msgs \
    elevation_mapping_cupy \
    unitree_lidar_ros2 \
    point_lio \
    nuwax_go2_bringup

echo "构建完成。请执行: source ${WS_DIR}/install/setup.bash"
