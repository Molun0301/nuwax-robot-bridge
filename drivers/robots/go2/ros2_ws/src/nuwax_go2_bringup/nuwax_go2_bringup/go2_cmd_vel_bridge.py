from __future__ import annotations

import os
from threading import Lock
from typing import Optional

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node


class Go2CmdVelBridge(Node):
    """将 Nav2 输出的速度命令桥接到 Go2 高层运动接口。"""

    def __init__(self) -> None:
        super().__init__("nuwax_go2_cmd_vel_bridge")
        self.declare_parameter("dds_iface", "")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("sport_timeout_sec", 3.0)
        self.declare_parameter("send_rate_hz", 20.0)
        self.declare_parameter("command_timeout_sec", 0.5)
        self.declare_parameter("max_linear_x", 1.0)
        self.declare_parameter("max_linear_y", 0.6)
        self.declare_parameter("max_angular_z", 1.5)

        self._dds_iface = str(self.get_parameter("dds_iface").value or os.environ.get("GO2_DDS_IFACE", "")).strip()
        self._cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self._sport_timeout_sec = float(self.get_parameter("sport_timeout_sec").value)
        self._send_rate_hz = max(1.0, float(self.get_parameter("send_rate_hz").value))
        self._command_timeout_sec = max(0.1, float(self.get_parameter("command_timeout_sec").value))
        self._max_linear_x = max(0.1, float(self.get_parameter("max_linear_x").value))
        self._max_linear_y = max(0.1, float(self.get_parameter("max_linear_y").value))
        self._max_angular_z = max(0.1, float(self.get_parameter("max_angular_z").value))

        self._sport_client = self._create_sport_client()
        self._lock = Lock()
        self._latest_twist: Optional[Twist] = None
        self._latest_stamp_sec: float = 0.0
        self._last_sent_signature: Optional[tuple] = None

        self.create_subscription(Twist, self._cmd_vel_topic, self._on_twist, 20)
        self.create_timer(1.0 / self._send_rate_hz, self._flush_motion_command)
        self.get_logger().info(
            f"Go2 cmd_vel 桥已启动，topic={self._cmd_vel_topic} dds_iface={self._dds_iface or '<default>'}"
        )

    def _create_sport_client(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.sport.sport_client import SportClient

        if self._dds_iface:
            ChannelFactoryInitialize(0, self._dds_iface)
        else:
            ChannelFactoryInitialize(0)

        client = SportClient()
        client.SetTimeout(self._sport_timeout_sec)
        client.Init()
        return client

    def _on_twist(self, message: Twist) -> None:
        with self._lock:
            self._latest_twist = message
            self._latest_stamp_sec = self.get_clock().now().nanoseconds / 1e9

    def _flush_motion_command(self) -> None:
        now_sec = self.get_clock().now().nanoseconds / 1e9
        with self._lock:
            latest_twist = self._latest_twist
            latest_stamp_sec = self._latest_stamp_sec

        if latest_twist is None or (now_sec - latest_stamp_sec) > self._command_timeout_sec:
            self._send_stop_if_needed()
            return

        vx = self._clamp(latest_twist.linear.x, self._max_linear_x)
        vy = self._clamp(latest_twist.linear.y, self._max_linear_y)
        vyaw = self._clamp(latest_twist.angular.z, self._max_angular_z)
        signature = (round(vx, 4), round(vy, 4), round(vyaw, 4))
        if signature == self._last_sent_signature:
            return
        code = self._sport_client.Move(vx, vy, vyaw)
        if code != 0:
            self.get_logger().warning(f"发送 Go2 Move 失败，code={code}")
            return
        self._last_sent_signature = signature

    def _send_stop_if_needed(self) -> None:
        if self._last_sent_signature in (None, (0.0, 0.0, 0.0)):
            return
        code = self._sport_client.StopMove()
        if code != 0:
            self.get_logger().warning(f"发送 Go2 StopMove 失败，code={code}")
            return
        self._last_sent_signature = (0.0, 0.0, 0.0)

    def destroy_node(self) -> bool:
        try:
            self._sport_client.StopMove()
        except Exception:
            self.get_logger().exception("停止 Go2 运动失败。")
        return super().destroy_node()

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        return max(-limit, min(limit, float(value)))


def main() -> None:
    rclpy.init(args=None)
    node = Go2CmdVelBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
