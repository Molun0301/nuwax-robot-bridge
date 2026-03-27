#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Go2 关节配置
定义关节索引和安全限制
"""

from settings import APP_CONFIG

# 关节ID映射 (来自 unitree_legged_const.py)
LegID = {
    "FR_0": 0,  # Front right hip (髋关节)
    "FR_1": 1,  # Front right thigh (大腿)
    "FR_2": 2,  # Front right calf (小腿)

    "FL_0": 3,  # Front left hip
    "FL_1": 4,  # Front left thigh
    "FL_2": 5,  # Front left calf

    "BR_0": 6,  # Back right hip
    "BR_1": 7,  # Back right thigh
    "BR_2": 8,  # Back right calf

    "BL_0": 9,  # Back left hip
    "BL_1": 10, # Back left thigh
    "BL_2": 11, # Back left calf

    # 别名
    "FR": 0,
    "FL": 3,
    "BR": 6,
    "BL": 9,
}

# 逆映射: ID -> 名称
LegID_reverse = {v: k for k, v in LegID.items()}

# 关节名称列表 (按索引顺序)
JOINT_NAMES = ["FR_0", "FR_1", "FR_2", "FL_0", "FL_1", "FL_2",
               "BR_0", "BR_1", "BR_2", "BL_0", "BL_1", "BL_2"]

# 默认安全限制 (单位: 弧度)
# 格式: [min, max]
SAFETY_LIMITS = {
    "FR_0": [-1.5, 1.5],   # 髋关节左右摆动
    "FR_1": [-0.2, 2.5],   # 大腿弯曲
    "FR_2": [-2.8, -0.5],  # 小腿弯曲

    "FL_0": [-1.5, 1.5],
    "FL_1": [-0.2, 2.5],
    "FL_2": [-2.8, -0.5],

    "BR_0": [-1.5, 1.5],
    "BR_1": [-0.2, 2.5],
    "BR_2": [-2.8, -0.5],

    "BL_0": [-1.5, 1.5],
    "BL_1": [-0.2, 2.5],
    "BL_2": [-2.8, -0.5],
}

# 默认控制参数
# 这些运行期默认值统一来自 `.env`，便于集中调整。
DEFAULT_KP = APP_CONFIG.low_level.default_kp
DEFAULT_KD = APP_CONFIG.low_level.default_kd
DEFAULT_MAX_VELOCITY = APP_CONFIG.low_level.max_velocity  # rad/s
DEFAULT_MAX_TORQUE = APP_CONFIG.low_level.max_torque      # Nm


def get_joint_id(joint_name: str) -> int:
    """获取关节索引"""
    if joint_name in LegID:
        return LegID[joint_name]
    raise ValueError(f"未知关节: {joint_name}")


def get_joint_name(joint_id: int) -> str:
    """获取关节名称"""
    if joint_id in LegID_reverse:
        return LegID_reverse[joint_id]
    raise ValueError(f"未知关节ID: {joint_id}")


def check_safety_limit(joint_name: str, position: float) -> bool:
    """检查位置是否在安全范围内"""
    if joint_name not in SAFETY_LIMITS:
        return True  # 未知关节不限制
    limits = SAFETY_LIMITS[joint_name]
    return limits[0] <= position <= limits[1]


def clamp_position(joint_name: str, position: float) -> float:
    """限制位置在安全范围内"""
    if joint_name not in SAFETY_LIMITS:
        return position
    limits = SAFETY_LIMITS[joint_name]
    return max(limits[0], min(limits[1], position))
