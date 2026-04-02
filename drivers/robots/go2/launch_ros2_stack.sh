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

if [[ ! -f "${WS_DIR}/install/setup.bash" ]]; then
  echo "未找到 ${WS_DIR}/install/setup.bash，请先运行 build_ros2_workspace.sh。" >&2
  exit 1
fi

if [[ -f "${REPO_ROOT}/drivers/robots/go2/data_plane.env" ]]; then
  set -a
  source "${REPO_ROOT}/drivers/robots/go2/data_plane.env"
  set +a
fi

source "${ROS_SETUP}"
source "${WS_DIR}/install/setup.bash"

if [[ -n "${GO2_DATA_PLANE_DDS_IFACE:-${GO2_DDS_IFACE:-}}" ]]; then
  exec ros2 launch nuwax_go2_bringup go2_full_stack.launch.py \
    dds_iface:="${GO2_DATA_PLANE_DDS_IFACE:-${GO2_DDS_IFACE:-}}" \
    "$@"
fi

exec ros2 launch nuwax_go2_bringup go2_full_stack.launch.py "$@"
