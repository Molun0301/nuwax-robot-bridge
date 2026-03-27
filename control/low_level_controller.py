#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Go2 低级别控制器
使用 LowCmd/LowState 进行单关节控制
"""

import os
import sys
import time
import threading
import logging

from settings import APP_CONFIG

logger = logging.getLogger(__name__)

# 添加SDK路径
SDK_PATH = APP_CONFIG.dds.sdk_path
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)

# 导入SDK模块
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

# 导入关节配置
from .joint_config import (
    LegID, JOINT_NAMES, SAFETY_LIMITS,
    DEFAULT_KP, DEFAULT_KD, DEFAULT_MAX_VELOCITY,
    check_safety_limit, clamp_position
)


class LowLevelController:
    """低级别控制器"""

    def __init__(self, iface=None):
        self.iface = iface

        # 控制状态
        self.is_initialized = False
        self.is_running = False
        self.mode = "high"  # "high" or "low"

        # LowCmd 和 LowState
        self.low_cmd = None
        self.low_state = None

        # 线程
        self.cmd_thread = None

        # CRC计算器
        self.crc = CRC()

        # 控制参数
        self.default_kp = DEFAULT_KP
        self.default_kd = DEFAULT_KD

    def init(self):
        """初始化低级别控制"""
        if self.is_initialized:
            logger.warning("低级别控制器已初始化")
            return True

        logger.info("初始化DDS通道 (低级别)...")
        if self.iface:
            ChannelFactoryInitialize(0, self.iface)
        else:
            ChannelFactoryInitialize(0)

        # 初始化LowCmd
        self._init_low_cmd()

        # 创建 publisher
        self.cmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.cmd_publisher.Init()

        # 创建 subscriber
        self.state_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.state_subscriber.Init(self._on_low_state, 10)

        # 等待状态数据
        logger.info("等待LowState数据...")
        timeout = 5.0
        start = time.time()
        while self.low_state is None and (time.time() - start) < timeout:
            time.sleep(0.1)

        if self.low_state is None:
            logger.error("未收到LowState数据")
            return False

        self.is_initialized = True
        logger.info("低级别控制器初始化成功")
        return True

    def _init_low_cmd(self):
        """初始化LowCmd结构"""
        self.low_cmd = unitree_go_msg_dds__LowCmd_()

        # 设置头部
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0

        # 初始化所有电机命令
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # PMSM模式
            self.low_cmd.motor_cmd[i].q = 0  # 位置
            self.low_cmd.motor_cmd[i].kp = 0  # 位置增益
            self.low_cmd.motor_cmd[i].dq = 0  # 速度
            self.low_cmd.motor_cmd[i].kd = 0  # 速度增益
            self.low_cmd.motor_cmd[i].tau = 0  # 力矩

    def _on_low_state(self, msg: LowState_):
        """LowState回调"""
        self.low_state = msg

    def start(self):
        """启动控制循环"""
        if not self.is_initialized:
            logger.error("控制器未初始化")
            return False

        if self.is_running:
            logger.warning("控制循环已在运行")
            return True

        self.is_running = True

        # 启动控制线程
        self.cmd_thread = RecurrentThread(
            interval=APP_CONFIG.low_level.control_interval_sec,
            target=self._control_loop,
            name="lowcmd_writer"
        )
        self.cmd_thread.Start()

        logger.info("低级别控制循环已启动")
        return True

    def stop(self):
        """停止控制循环"""
        self.is_running = False

        if self.cmd_thread:
            self.cmd_thread.Stop()

        logger.info("低级别控制循环已停止")

    def _control_loop(self):
        """控制循环 (在独立线程中运行)"""
        # 这里可以添加自定义控制逻辑
        # 目前只是发送命令

        # 计算CRC并发送
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.cmd_publisher.Write(self.low_cmd)

    def set_joint_position(self, joint_name: str, position: float,
                          kp: float = None, kd: float = None) -> bool:
        """
        设置单关节位置

        Args:
            joint_name: 关节名称 (如 "FR_1")
            position: 目标位置 (弧度)
            kp: 位置增益 (可选)
            kd: 速度增益 (可选)

        Returns:
            是否成功
        """
        if not self.is_initialized:
            logger.error("控制器未初始化")
            return False

        try:
            joint_id = LegID[joint_name]
        except KeyError:
            logger.error(f"未知关节: {joint_name}")
            return False

        # 检查安全限制
        if not check_safety_limit(joint_name, position):
            position = clamp_position(joint_name, position)
            logger.warning(f"位置超出限制，已限制到: {position}")

        # 设置电机命令
        self.low_cmd.motor_cmd[joint_id].q = position
        self.low_cmd.motor_cmd[joint_id].dq = 0
        self.low_cmd.motor_cmd[joint_id].kp = kp if kp is not None else self.default_kp
        self.low_cmd.motor_cmd[joint_id].kd = kd if kd is not None else self.default_kd
        self.low_cmd.motor_cmd[joint_id].tau = 0

        return True

    def set_joints_position(self, joint_positions: dict,
                          kp: float = None, kd: float = None) -> bool:
        """
        设置多关节位置

        Args:
            joint_positions: 字典 {关节名称: 位置}
            kp: 位置增益 (可选)
            kd: 速度增益 (可选)

        Returns:
            是否成功
        """
        for joint_name, position in joint_positions.items():
            if not self.set_joint_position(joint_name, position, kp, kd):
                return False
        return True

    def set_joint_velocity(self, joint_name: str, velocity: float,
                          kd: float = None) -> bool:
        """
        设置单关节速度

        Args:
            joint_name: 关节名称
            velocity: 目标速度 (弧度/秒)
            kd: 速度增益 (可选)

        Returns:
            是否成功
        """
        if not self.is_initialized:
            logger.error("控制器未初始化")
            return False

        try:
            joint_id = LegID[joint_name]
        except KeyError:
            logger.error(f"未知关节: {joint_name}")
            return False

        # 限制速度
        if abs(velocity) > DEFAULT_MAX_VELOCITY:
            velocity = DEFAULT_MAX_VELOCITY if velocity > 0 else -DEFAULT_MAX_VELOCITY
            logger.warning(f"速度超出限制，已限制到: {velocity}")

        # 设置电机命令
        self.low_cmd.motor_cmd[joint_id].q = 0  # 保持当前位置
        self.low_cmd.motor_cmd[joint_id].dq = velocity
        self.low_cmd.motor_cmd[joint_id].kp = 0
        self.low_cmd.motor_cmd[joint_id].kd = kd if kd is not None else self.default_kd
        self.low_cmd.motor_cmd[joint_id].tau = 0

        return True

    def set_joint_torque(self, joint_name: str, torque: float) -> bool:
        """
        设置单关节力矩

        Args:
            joint_name: 关节名称
            torque: 目标力矩 (Nm)

        Returns:
            是否成功
        """
        if not self.is_initialized:
            logger.error("控制器未初始化")
            return False

        try:
            joint_id = LegID[joint_name]
        except KeyError:
            logger.error(f"未知关节: {joint_name}")
            return False

        # 设置电机命令
        self.low_cmd.motor_cmd[joint_id].q = 0
        self.low_cmd.motor_cmd[joint_id].dq = 0
        self.low_cmd.motor_cmd[joint_id].kp = 0
        self.low_cmd.motor_cmd[joint_id].kd = 0
        self.low_cmd.motor_cmd[joint_id].tau = torque

        return True

    def get_joint_state(self, joint_name: str = None) -> dict:
        """
        获取关节状态

        Args:
            joint_name: 关节名称 (可选，不传则返回所有)

        Returns:
            关节状态字典
        """
        if self.low_state is None:
            return {}

        if joint_name:
            try:
                joint_id = LegID[joint_name]
            except KeyError:
                return {}

            motor = self.low_state.motor_state[joint_id]
            return {
                "name": joint_name,
                "position": motor.q,
                "velocity": motor.dq,
                "torque": motor.tau,
            }
        else:
            # 返回所有关节状态
            states = []
            for i, name in enumerate(JOINT_NAMES):
                motor = self.low_state.motor_state[i]
                states.append({
                    "name": name,
                    "position": motor.q,
                    "velocity": motor.dq,
                    "torque": motor.tau,
                })
            return {"joints": states}

    def get_imu_state(self) -> dict:
        """获取IMU状态"""
        if self.low_state is None:
            return {}

        imu = self.low_state.imu_state
        return {
            "quaternion": list(imu.quaternion),
            "gyroscope": list(imu.gyroscope),
            "accelerometer": list(imu.accelerometer),
            "temperature": imu.temperature,
        }

    def apply_pose_preset(self, preset_name: str) -> bool:
        """
        应用预设姿态

        Args:
            preset_name: 预设名称 (stand/sit/zero)

        Returns:
            是否成功
        """
        presets = {
            "stand": {
                "FR_0": 0.0, "FR_1": 1.36, "FR_2": -2.65,
                "FL_0": 0.0, "FL_1": 1.36, "FL_2": -2.65,
                "BR_0": 0.0, "BR_1": 1.36, "BR_2": -2.65,
                "BL_0": 0.0, "BL_1": 1.36, "BL_2": -2.65,
            },
            "sit": {
                "FR_0": 0.0, "FR_1": 0.67, "FR_2": -1.3,
                "FL_0": 0.0, "FL_1": 0.67, "FL_2": -1.3,
                "BR_0": 0.0, "BR_1": 0.67, "BR_2": -1.3,
                "BL_0": 0.0, "BL_1": 0.67, "BL_2": -1.3,
            },
            "zero": {
                "FR_0": 0.0, "FR_1": 0.0, "FR_2": 0.0,
                "FL_0": 0.0, "FL_1": 0.0, "FL_2": 0.0,
                "BR_0": 0.0, "BR_1": 0.0, "BR_2": 0.0,
                "BL_0": 0.0, "BL_1": 0.0, "BL_2": 0.0,
            },
        }

        if preset_name not in presets:
            logger.error(f"未知预设: {preset_name}")
            return False

        return self.set_joints_position(presets[preset_name])
