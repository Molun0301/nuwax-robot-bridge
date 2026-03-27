#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Go2 代理服务端 - TCP版本
高层控制：直接使用 SportClient / VideoClient
监听TCP 8765端口，接收JSON格式命令
"""

import os
import sys
import json
import re
import socket
import subprocess
import threading
import time
import traceback
import base64
import logging
import shutil
from dataclasses import replace

from settings import APP_CONFIG, configure_logging

# 配置日志
configure_logging(APP_CONFIG.logging)
logger = logging.getLogger(__name__)
logger.info("日志文件路径: %s", APP_CONFIG.logging.log_file)


def _parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ('true', '1', 'yes', 'on')


# 导入实时TTS播放器模块
try:
    from tts.realtime_tts_player import (
        RealtimeTTSPlayer,
        build_player_from_config,
    )
    REALTIME_TTS_AVAILABLE = True
except ImportError as e:
    REALTIME_TTS_AVAILABLE = False
    RealtimeTTSPlayer = None
    build_player_from_config = None
    logger.warning(f"实时TTS播放器模块不可用: {e}")

# 导入日志监听转TTS桥接模块
try:
    from tts.log_tts_bridge import (
        LogTTSBridgeService,
        build_log_tts_bridge_from_config,
    )
    LOG_TTS_BRIDGE_AVAILABLE = True
except ImportError as e:
    LOG_TTS_BRIDGE_AVAILABLE = False
    LogTTSBridgeService = None
    build_log_tts_bridge_from_config = None
    logger.warning(f"日志转TTS桥接模块不可用: {e}")

try:
    from tts.doubao_realtime_client import (
        DoubaoRealtimeClient,
        build_doubao_realtime_client,
    )
    DOUBAO_TTS_AVAILABLE = True
except ImportError as e:
    DOUBAO_TTS_AVAILABLE = False
    DoubaoRealtimeClient = None
    build_doubao_realtime_client = None
    logger.warning(f"本地 Doubao 流式客户端模块不可用: {e}")

# 导入控制模块
LOW_LEVEL_AVAILABLE = False
try:
    from control.low_level_controller import LowLevelController
    from control.joint_config import LegID
    LOW_LEVEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"控制模块不可用: {e}")
except Exception as e:
    logger.warning(f"控制模块加载错误: {e}")

# 全局低级别控制器
low_level_controller = None
current_mode = "high"  # "high" or "low"
dds_iface = APP_CONFIG.dds.iface
vui_client = None

# 设置DDS环境
os.environ['CYCLONEDDS_URI'] = APP_CONFIG.dds.cyclone_dds_uri

# TCP配置
TCP_HOST = APP_CONFIG.tcp.host
TCP_PORT = APP_CONFIG.tcp.port
TTS_CONFIG = APP_CONFIG.tts
DOUBAO_CONFIG = APP_CONFIG.doubao
LOG_TTS_CONFIG = APP_CONFIG.log_tts
VOLUME_CONFIG = APP_CONFIG.volume
AUDIO_COMMAND_TIMEOUT = VOLUME_CONFIG.audio_command_timeout
INIT_VOLUME_ON_START = VOLUME_CONFIG.init_volume_on_start
VOLUME_MIXER_DEVICE = VOLUME_CONFIG.mixer_device
VOLUME_MIXER_CONTROL = VOLUME_CONFIG.mixer_control

# 音量控制 (0.0 - 1.0)
current_volume = max(0.0, min(1.0, TTS_CONFIG.initial_volume))

# 全局实时TTS播放器
realtime_tts_player = None
doubao_tts_client = None
log_tts_bridge = None


def _build_realtime_tts_player(params=None):
    params = params or {}
    player_config = replace(TTS_CONFIG)
    player_config.local = replace(
        player_config.local,
        preferred_backend=params.get('preferred_backend', player_config.local.preferred_backend),
        audio_device=params.get('audio_device', player_config.local.audio_device),
        enable_spk_ctl=_parse_bool(
            params.get('enable_spk_ctl', player_config.local.enable_spk_ctl),
            player_config.local.enable_spk_ctl,
        ),
        enable_ape_route=_parse_bool(
            params.get('enable_ape_route', player_config.local.enable_ape_route),
            player_config.local.enable_ape_route,
        ),
        ape_card=params.get('ape_card', player_config.local.ape_card),
        ape_admaif=int(params.get('ape_admaif', player_config.local.ape_admaif)),
        ape_speaker=params.get('ape_speaker', player_config.local.ape_speaker),
    )
    player_config.robot = replace(
        player_config.robot,
        robot_ip=params.get('robot_ip', player_config.robot.robot_ip),
        connection_mode=params.get('robot_mode', player_config.robot.connection_mode),
        use_megaphone=_parse_bool(
            params.get('use_megaphone', player_config.robot.use_megaphone),
            player_config.robot.use_megaphone,
        ),
        streaming_enabled=_parse_bool(
            params.get('streaming_enabled', player_config.robot.streaming_enabled),
            player_config.robot.streaming_enabled,
        ),
        streaming_chunk_ms=int(
            params.get('streaming_chunk_ms', player_config.robot.streaming_chunk_ms)
        ),
        connect_timeout=float(
            params.get('robot_connect_timeout', player_config.robot.connect_timeout)
        ),
        request_timeout=float(
            params.get('robot_request_timeout', player_config.robot.request_timeout)
        ),
        upload_chunk_size=int(
            params.get('upload_chunk_size', player_config.robot.upload_chunk_size)
        ),
        megaphone_chunk_size=int(
            params.get('megaphone_chunk_size', player_config.robot.megaphone_chunk_size)
        ),
        auto_fallback_to_local=_parse_bool(
            params.get('auto_fallback_to_local', player_config.robot.auto_fallback_to_local),
            player_config.robot.auto_fallback_to_local,
        ),
    )
    player_config = replace(
        player_config,
        auto_start=_parse_bool(params.get('auto_start', player_config.auto_start), player_config.auto_start),
        print_subtitle=_parse_bool(
            params.get('print_subtitle', player_config.print_subtitle),
            player_config.print_subtitle,
        ),
        initial_volume=float(params.get('volume', current_volume)),
        backend_mode=params.get('backend_mode', player_config.backend_mode),
        fallback_backend_mode=params.get(
            'fallback_backend_mode',
            player_config.fallback_backend_mode,
        ),
    )
    return build_player_from_config(player_config)


def _build_doubao_tts_client(player):
    return build_doubao_realtime_client(player, DOUBAO_CONFIG)


def _run_audio_command(command):
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=AUDIO_COMMAND_TIMEOUT,
    )


def _clamp_volume(volume):
    return max(0.0, min(1.0, float(volume)))


def _parse_percent_to_ratio(text):
    matches = re.findall(r'(\d+)%', text)
    if not matches:
        return None
    return max(0.0, min(1.0, int(matches[0]) / 100.0))


def _parse_wpctl_volume(text):
    match = re.search(r'Volume:\s*([0-9]*\.?[0-9]+)', text)
    if not match:
        return None
    return max(0.0, min(1.0, float(match.group(1))))


def _parse_pactl_sink_names(text):
    names = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            names.append(parts[1])
    return names


def _parse_amixer_controls(text):
    controls = []
    for match in re.finditer(r"Simple mixer control '([^']+)'", text):
        control = match.group(1).strip()
        if control and control not in controls:
            controls.append(control)
    return controls


def _parse_amixer_limits(text):
    match = re.search(r'Limits:\s*(-?\d+)\s*-\s*(-?\d+)', text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _control_priority(control):
    normalized = control.strip().lower()
    priorities = {
        'master': 0,
        'speaker': 1,
        'headphone': 2,
        'pcm': 3,
        'playback': 4,
    }
    return priorities.get(normalized, 100)


def _is_suspicious_amixer_control(control, lower_limit, upper_limit):
    normalized = control.strip().lower()
    bad_keywords = (
        'channel',
        'channels',
        'byte',
        'map',
        'path',
        'mux',
        'route',
        'enum',
        'mode',
        'rate',
        'clock',
        'switch',
    )
    if any(keyword in normalized for keyword in bad_keywords):
        return True
    if lower_limit is None or upper_limit is None:
        return False
    return (upper_limit - lower_limit) <= 4 and _control_priority(control) >= 100


def _amixer_control_candidates():
    candidates = []
    if VOLUME_MIXER_CONTROL:
        candidates.append(VOLUME_MIXER_CONTROL)

    candidates.extend(["Master", "Speaker", "PCM", "Playback", "Headphone"])

    try:
        result = _run_audio_command(['amixer', '-D', VOLUME_MIXER_DEVICE, 'scontrols'])
        for control in _parse_amixer_controls(result.stdout):
            if control not in candidates:
                candidates.append(control)
    except Exception:
        pass

    if VOLUME_MIXER_CONTROL:
        return candidates

    ranked = []
    for control in candidates:
        ranked.append((_control_priority(control), control))
    ranked.sort(key=lambda item: (item[0], item[1].lower()))
    return [control for _, control in ranked]


def _get_amixer_control_info(control, allow_suspicious=False):
    result = _run_audio_command(['amixer', '-D', VOLUME_MIXER_DEVICE, 'get', control])
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    volume = _parse_percent_to_ratio(combined)
    if volume is None:
        raise RuntimeError("无法解析音量输出: %s" % combined.strip())

    lower_limit, upper_limit = _parse_amixer_limits(combined)
    if not allow_suspicious and _is_suspicious_amixer_control(control, lower_limit, upper_limit):
        raise RuntimeError(
            "控件 `%s` 看起来不是实际音量控件: limits=%s-%s"
            % (control, lower_limit, upper_limit)
        )

    return {
        'backend': 'amixer',
        'device': VOLUME_MIXER_DEVICE,
        'control': control,
        'volume': volume,
        'lower_limit': lower_limit,
        'upper_limit': upper_limit,
        'raw': combined.strip(),
    }


def _get_amixer_volume_info():
    errors = []
    for control in _amixer_control_candidates():
        try:
            return _get_amixer_control_info(
                control,
                allow_suspicious=bool(VOLUME_MIXER_CONTROL),
            )
        except Exception as exc:
            errors.append("%s: %s" % (control, exc))

    raise RuntimeError(
        "amixer 设备 `%s` 上没有可用音量控件: %s"
        % (VOLUME_MIXER_DEVICE, "; ".join(errors or ['未找到控件']))
    )


def _set_amixer_volume(volume):
    percent = int(round(volume * 100))
    errors = []
    for control in _amixer_control_candidates():
        try:
            if not VOLUME_MIXER_CONTROL:
                _get_amixer_control_info(control, allow_suspicious=False)
            _run_audio_command(
                ['amixer', '-D', VOLUME_MIXER_DEVICE, '-q', 'sset', control, '%d%%' % percent, 'unmute']
            )
            info = _get_amixer_control_info(
                control,
                allow_suspicious=bool(VOLUME_MIXER_CONTROL),
            )
            info['set_control'] = control
            return info
        except Exception as exc:
            errors.append("%s: %s" % (control, exc))

    raise RuntimeError(
        "amixer 无法设置设备 `%s` 的音量: %s"
        % (VOLUME_MIXER_DEVICE, "; ".join(errors or ['未找到控件']))
    )


def _get_pactl_sink_candidates():
    candidates = ['@DEFAULT_SINK@']
    if not shutil.which('pactl'):
        return candidates
    try:
        result = _run_audio_command(['pactl', 'list', 'sinks', 'short'])
        for sink in _parse_pactl_sink_names(result.stdout):
            if sink not in candidates:
                candidates.append(sink)
    except Exception:
        pass
    return candidates


def _get_pactl_volume_info():
    errors = []
    for sink in _get_pactl_sink_candidates():
        try:
            volume_result = _run_audio_command(['pactl', 'get-sink-volume', sink])
            mute_result = _run_audio_command(['pactl', 'get-sink-mute', sink])
            volume_combined = "\n".join(
                part for part in (volume_result.stdout, volume_result.stderr) if part
            )
            mute_combined = "\n".join(
                part for part in (mute_result.stdout, mute_result.stderr) if part
            )
            volume = _parse_percent_to_ratio(volume_combined)
            if volume is None:
                raise RuntimeError("无法解析音量输出: %s" % volume_combined.strip())
            muted = 'yes' in mute_combined.lower()
            return {
                'backend': 'pactl',
                'sink': sink,
                'volume': volume,
                'muted': muted,
                'raw': (volume_combined + "\n" + mute_combined).strip(),
            }
        except Exception as exc:
            errors.append('%s: %s' % (sink, exc))
    raise RuntimeError("无法获取Pulse sink音量: %s" % "; ".join(errors or ['未找到sink']))


def _set_pactl_volume(volume):
    percent = int(round(_clamp_volume(volume) * 100))
    errors = []
    for sink in _get_pactl_sink_candidates():
        try:
            try:
                _run_audio_command(['pactl', 'set-default-sink', sink])
            except Exception:
                pass
            _run_audio_command(['pactl', 'set-sink-volume', sink, '%d%%' % percent])
            _run_audio_command(['pactl', 'set-sink-mute', sink, '0'])
            info = _get_pactl_volume_info()
            info['set_sink'] = sink
            return info
        except Exception as exc:
            errors.append('%s: %s' % (sink, exc))
    raise RuntimeError("pactl 无法设置sink音量: %s" % "; ".join(errors or ['未找到sink']))


def _get_system_volume_info():
    probes = []

    if shutil.which('wpctl'):
        probes.append(('wpctl', ['wpctl', 'get-volume', '@DEFAULT_AUDIO_SINK@'], _parse_wpctl_volume))

    if shutil.which('pactl'):
        probes.append(('pactl', ['pactl', 'get-sink-volume', '@DEFAULT_SINK@'], _parse_percent_to_ratio))

    if shutil.which('amixer'):
        probes.append(('amixer', None, None))

    errors = []
    for backend, command, parser in probes:
        try:
            if backend == 'amixer':
                return _get_amixer_volume_info()
            if backend == 'pactl':
                return _get_pactl_volume_info()
            result = _run_audio_command(command)
            combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
            volume = parser(combined)
            if volume is None:
                raise RuntimeError("无法解析音量输出: %s" % combined.strip())
            return {
                'backend': backend,
                'volume': volume,
                'raw': combined.strip(),
            }
        except Exception as exc:
            errors.append('%s: %s' % (backend, exc))

    raise RuntimeError("无法获取系统音量: %s" % "; ".join(errors or ['未找到可用音量后端']))


def _set_system_volume(volume):
    volume = _clamp_volume(volume)
    percent = int(round(volume * 100))
    backends = []

    if shutil.which('wpctl'):
        backends.append((
            'wpctl',
            [
                ['wpctl', 'set-volume', '@DEFAULT_AUDIO_SINK@', '%.2f' % volume],
                ['wpctl', 'set-mute', '@DEFAULT_AUDIO_SINK@', '0'],
            ],
        ))

    if shutil.which('pactl'):
        backends.append((
            'pactl',
            None,
        ))

    if shutil.which('amixer'):
        backends.append((
            'amixer',
            None,
        ))

    errors = []
    for backend, commands in backends:
        try:
            if backend == 'amixer':
                return _set_amixer_volume(volume)
            if backend == 'pactl':
                return _set_pactl_volume(volume)
            for command in commands:
                _run_audio_command(command)
            info = _get_system_volume_info()
            if info.get('backend') != backend:
                info['set_via_backend'] = backend
            return info
        except Exception as exc:
            errors.append('%s: %s' % (backend, exc))

    raise RuntimeError("无法设置系统音量: %s" % "; ".join(errors or ['未找到可用音量后端']))


def _vui_level_to_ratio(level):
    span = max(1, VOLUME_CONFIG.vui_max_level - VOLUME_CONFIG.vui_min_level)
    normalized = (int(level) - VOLUME_CONFIG.vui_min_level) / float(span)
    return _clamp_volume(normalized)


def _ratio_to_vui_level(volume):
    volume = _clamp_volume(volume)
    span = max(1, VOLUME_CONFIG.vui_max_level - VOLUME_CONFIG.vui_min_level)
    return int(round(VOLUME_CONFIG.vui_min_level + volume * span))


def _get_vui_volume_info():
    if vui_client is None:
        raise RuntimeError("VuiClient 尚未初始化")

    code, level = vui_client.GetVolume()
    if code != 0 or level is None:
        raise RuntimeError("读取 Go2 VUI 音量失败: code=%s" % code)

    switch_code, switch_enabled = vui_client.GetSwitch()
    result = {
        'backend': 'vui',
        'level': int(level),
        'volume': _vui_level_to_ratio(level),
    }
    if switch_code == 0 and switch_enabled is not None:
        result['switch_enabled'] = bool(switch_enabled)
    return result


def _set_vui_volume(volume):
    if vui_client is None:
        raise RuntimeError("VuiClient 尚未初始化")

    target_level = _ratio_to_vui_level(volume)
    if VOLUME_CONFIG.auto_enable_vui_switch:
        switch_code = vui_client.SetSwitch(1)
        if switch_code != 0:
            raise RuntimeError("启用 Go2 语音开关失败: code=%s" % switch_code)

    code = vui_client.SetVolume(target_level)
    if code != 0:
        raise RuntimeError("设置 Go2 VUI 音量失败: code=%s" % code)

    info = _get_vui_volume_info()
    info['set_level'] = target_level
    return info


def _player_backend_mode():
    if realtime_tts_player is None:
        return TTS_CONFIG.backend_mode
    try:
        status = realtime_tts_player.status()
    except Exception:
        return TTS_CONFIG.backend_mode
    return (
        status.get('active_backend_mode')
        or status.get('configured_backend_mode')
        or TTS_CONFIG.backend_mode
    )


def _sync_player_volume(volume):
    global realtime_tts_player

    if realtime_tts_player is None:
        return None
    return realtime_tts_player.set_volume(volume)


def _stop_tts_runtime():
    global realtime_tts_player, doubao_tts_client, log_tts_bridge

    if log_tts_bridge is not None:
        try:
            log_tts_bridge.stop()
        except Exception as exc:
            logger.warning("停止日志转TTS桥接失败: %s", exc)
        log_tts_bridge = None

    if doubao_tts_client is not None:
        try:
            doubao_tts_client.stop()
        except Exception as exc:
            logger.warning("停止本地Doubao流式客户端失败: %s", exc)
        doubao_tts_client = None

    if realtime_tts_player is not None:
        try:
            realtime_tts_player.stop()
        except Exception as exc:
            logger.warning("停止本地TTS播放管理器失败: %s", exc)
        realtime_tts_player = None


def _start_tts_runtime(params=None):
    global realtime_tts_player, doubao_tts_client, log_tts_bridge, current_volume

    if not REALTIME_TTS_AVAILABLE:
        raise RuntimeError("本地TTS播放管理器模块不可用")
    if not DOUBAO_TTS_AVAILABLE:
        raise RuntimeError("本地 Doubao 流式客户端模块不可用")
    if not LOG_TTS_BRIDGE_AVAILABLE and LOG_TTS_CONFIG.enabled:
        raise RuntimeError("日志转TTS桥接模块不可用")

    params = params or {}
    should_recreate = (
        realtime_tts_player is None
        or doubao_tts_client is None
        or any(
            key in params
            for key in (
                'preferred_backend',
                'audio_device',
                'enable_spk_ctl',
                'enable_ape_route',
                'ape_card',
                'ape_admaif',
                'ape_speaker',
                'robot_ip',
                'robot_mode',
                'use_megaphone',
                'streaming_enabled',
                'streaming_chunk_ms',
                'robot_connect_timeout',
                'robot_request_timeout',
                'upload_chunk_size',
                'megaphone_chunk_size',
                'auto_fallback_to_local',
                'backend_mode',
                'fallback_backend_mode',
                'volume',
                'print_subtitle',
                'auto_start',
            )
        )
    )

    if 'volume' in params:
        current_volume = _clamp_volume(params.get('volume', current_volume))

    if should_recreate:
        _stop_tts_runtime()
        realtime_tts_player = _build_realtime_tts_player(params)
        doubao_tts_client = _build_doubao_tts_client(realtime_tts_player)

    started_player = realtime_tts_player.start()
    started_doubao = doubao_tts_client.start()
    _sync_player_volume(current_volume)

    started_log_bridge = False
    log_bridge_error = ""
    if LOG_TTS_CONFIG.enabled:
        try:
            if log_tts_bridge is None or should_recreate:
                log_tts_bridge = build_log_tts_bridge_from_config(LOG_TTS_CONFIG, doubao_tts_client, DOUBAO_CONFIG)
            started_log_bridge = log_tts_bridge.start()
        except Exception as exc:
            log_bridge_error = str(exc)
            logger.error("日志转TTS桥接启动失败: %s", exc)

    status = _tts_player_status_action()[1]
    status['started_player_now'] = started_player
    status['started_doubao_now'] = started_doubao
    status['started_log_bridge_now'] = started_log_bridge
    if log_bridge_error:
        status['log_tts_bridge_error'] = log_bridge_error
    return status


def _initialize_volume_state():
    global current_volume

    try:
        if VOLUME_CONFIG.prefer_vui:
            info = _get_vui_volume_info()
        else:
            info = _get_system_volume_info()
        current_volume = info['volume']
        logger.info("检测到系统音量: backend=%s volume=%.2f", info['backend'], current_volume)
    except Exception as exc:
        logger.info("未检测到可用系统音量后端，继续使用软件音量 %.2f: %s", current_volume, exc)


def _tts_player_start_action(params):
    try:
        status = _start_tts_runtime(params)
    except Exception as exc:
        return None, str(exc)
    logger.info("本地TTS运行时启动状态: %s", status)
    return 0, status


def _tts_player_stop_action():
    _stop_tts_runtime()
    status = _tts_player_status_action()[1]
    logger.info("本地TTS运行时已停止: %s", status)
    return 0, status


def _tts_player_status_action():
    global log_tts_bridge, doubao_tts_client
    robot_volume = None
    compat_volume = None
    try:
        robot_volume = _get_vui_volume_info()
    except Exception as exc:
        robot_volume = {'error': str(exc)}
    try:
        compat_volume = _get_system_volume_info()
    except Exception as exc:
        compat_volume = {'error': str(exc)}

    if realtime_tts_player is None:
        return 0, {
            'running': False,
            'configured_from_config': REALTIME_TTS_AVAILABLE,
            'config_file': os.path.join(APP_CONFIG.base_dir, '.env'),
            'volume': current_volume,
            'robot_volume': robot_volume,
            'compat_volume': compat_volume,
            'doubao_tts': (
                doubao_tts_client.status()
                if doubao_tts_client is not None
                else {
                    'running': False,
                    'has_credentials': bool(DOUBAO_CONFIG.app_id and DOUBAO_CONFIG.access_key),
                }
            ),
            'log_tts_bridge': (
                log_tts_bridge.status()
                if log_tts_bridge is not None
                else {
                    'enabled': LOG_TTS_CONFIG.enabled,
                    'available': LOG_TTS_BRIDGE_AVAILABLE,
                    'running': False,
                }
            ),
            'message': '本地TTS运行时尚未创建',
        }
    status = realtime_tts_player.status()
    status['robot_volume'] = robot_volume
    status['compat_volume'] = compat_volume
    status['doubao_tts'] = (
        doubao_tts_client.status()
        if doubao_tts_client is not None
        else {
            'running': False,
            'has_credentials': bool(DOUBAO_CONFIG.app_id and DOUBAO_CONFIG.access_key),
        }
    )
    status['log_tts_bridge'] = (
        log_tts_bridge.status()
        if log_tts_bridge is not None
        else {
            'enabled': LOG_TTS_CONFIG.enabled,
            'available': LOG_TTS_BRIDGE_AVAILABLE,
            'running': False,
        }
    )
    return 0, status


def init_go2_client(iface=None):
    """
    初始化Go2客户端 - 高层控制
    """
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    from unitree_sdk2py.go2.video.video_client import VideoClient
    from unitree_sdk2py.go2.vui.vui_client import VuiClient

    logger.info("初始化DDS通道...")
    if iface:
        ChannelFactoryInitialize(0, iface)
    else:
        ChannelFactoryInitialize(0)

    logger.info("创建SportClient...")
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    logger.info("创建VideoClient...")
    video_client = VideoClient()
    video_client.SetTimeout(3.0)
    video_client.Init()

    logger.info("创建VuiClient...")
    vui = VuiClient()
    vui.SetTimeout(3.0)
    vui.Init()

    logger.info("等待连接稳定...")
    time.sleep(1.0)

    return sport_client, video_client, vui


def _move_for(sport_client, params):
    """
    持续移动指定时间
    params: vx, vy, vyaw, duration(秒)
    """
    vx = float(params.get('vx', 0))
    vy = float(params.get('vy', 0))
    vyaw = float(params.get('vyaw', 0))
    duration = float(params.get('duration', 1.0))
    interval = 0.5  # 每0.5秒发送一次move命令

    logger.info(f"move_for: vx={vx}, vy={vy}, vyaw={vyaw}, duration={duration}")

    start_time = time.time()
    while time.time() - start_time < duration:
        code = sport_client.Move(vx, vy, vyaw)
        if code != 0:
            logger.warning(f"move_for: Move返回错误码 {code}")
        time.sleep(interval)

    # 移动结束后停止
    sport_client.StopMove()
    logger.info("move_for: 移动完成，已执行StopMove")
    return 0


def _volume_action(params):
    """音量控制"""
    global current_volume

    action = params.get('action', 'get')  # get, set

    if action == 'get':
        result = {'volume': current_volume}
        try:
            robot_volume = _get_vui_volume_info()
            current_volume = robot_volume['volume']
            result['volume'] = current_volume
            result['robot_volume'] = robot_volume
        except Exception as exc:
            result['robot_volume_error'] = str(exc)
        try:
            result['compat_volume'] = _get_system_volume_info()
        except Exception as exc:
            result['compat_volume_error'] = str(exc)
        if realtime_tts_player is not None:
            result['player_status'] = realtime_tts_player.status().get('audio_sink')
        return 0, result
    elif action == 'set':
        volume = params.get('volume')
        if volume is None:
            return None, "缺少volume参数"
        try:
            volume = _clamp_volume(volume)
            current_volume = volume
            result = {'volume': current_volume}
            try:
                result['robot_volume'] = _set_vui_volume(current_volume)
            except Exception as exc:
                result['robot_volume_error'] = str(exc)

            if _player_backend_mode() == 'alsa_local':
                player_volume = _sync_player_volume(current_volume)
                if player_volume is not None:
                    result['player_volume'] = player_volume
                try:
                    result['compat_volume'] = _set_system_volume(current_volume)
                except Exception as exc:
                    result['compat_volume_error'] = str(exc)
                    result['software_only'] = True
                    logger.info("本地兼容链路系统音量不可用，已回退到播放器软件音量: %s", exc)
            else:
                result['player_volume'] = None
            logger.info("音量设置完成: %s", result)
            return 0, result
        except ValueError:
            return None, "volume必须是数字"
    else:
        return None, f"未知action: {action}"


def _legacy_speak_removed_action():
    return None, (
        "当前 `speak` 动作尚未接入新的本地直连TTS运行时。"
        "请优先使用日志驱动自动播报，或先启动 `tts_player_start` 让本地TTS运行时就绪。"
    )


def execute_action(sport_client, video_client, action, params):
    """执行动作"""
    params = params or {}

    # 记录所有操作
    logger.info(f"执行动作: {action} | 参数: {params}")

    # ========== 摄像头操作 ==========
    if action == 'get_image':
        """获取单帧图像，返回base64编码"""
        logger.info("获取摄像头图像...")
        code, data = video_client.GetImageSample()
        if code == 0:
            # 返回base64编码的图像数据
            img_base64 = base64.b64encode(bytes(data)).decode('utf-8')
            return code, {'image_base64': img_base64, 'size': len(data)}
        else:
            return code, None

    if action == 'save_image':
        """保存图像到文件"""
        filename = params.get('filename', '/tmp/go2_image.jpg')
        logger.info(f"保存图像到: {filename}")
        code, data = video_client.GetImageSample()
        if code == 0:
            with open(filename, 'wb') as f:
                f.write(bytes(data))
            return code, {'filename': filename, 'size': len(data)}
        else:
            return code, None

    # ========== 运动控制 ==========
    no_param_actions = {
        'damp': sport_client.Damp,
        'balance_stand': sport_client.BalanceStand,
        'stop_move': sport_client.StopMove,
        'stand_up': sport_client.StandUp,
        'stand_down': sport_client.StandDown,
        'recovery_stand': sport_client.RecoveryStand,
        'sit': sport_client.Sit,
        'rise_sit': sport_client.RiseSit,
        'hello': sport_client.Hello,
        'stretch': sport_client.Stretch,
        'dance1': sport_client.Dance1,
        'dance2': sport_client.Dance2,
        'front_flip': sport_client.FrontFlip,
        'front_jump': sport_client.FrontJump,
        'front_pounce': sport_client.FrontPounce,
        'left_flip': sport_client.LeftFlip,
        'back_flip': sport_client.BackFlip,
        'free_walk': sport_client.FreeWalk,
    }

    bool_param_actions = {
        'hand_stand': (sport_client.HandStand, 'flag'),
        'free_jump': (sport_client.FreeJump, 'flag'),
        'free_bound': (sport_client.FreeBound, 'flag'),
        'free_avoid': (sport_client.FreeAvoid, 'flag'),
        'walk_upright': (sport_client.WalkUpright, 'flag'),
        'cross_step': (sport_client.CrossStep, 'flag'),
        'switch_joystick': (sport_client.SwitchJoystick, 'on'),
        'pose': (sport_client.Pose, 'flag'),
        'classic_walk': (sport_client.ClassicWalk, 'flag'),
    }

    special_actions = {
        'move': lambda: sport_client.Move(
            float(params.get('vx', 0)),
            float(params.get('vy', 0)),
            float(params.get('vyaw', 0))
        ),
        'move_for': lambda: _move_for(sport_client, params),
        'volume': lambda: _volume_action(params),
        'speak': lambda: _legacy_speak_removed_action(),
        'tts_player_start': lambda: _tts_player_start_action(params),
        'tts_player_stop': lambda: _tts_player_stop_action(),
        'tts_player_status': lambda: _tts_player_status_action(),
        'euler': lambda: sport_client.Euler(
            float(params.get('roll', 0)),
            float(params.get('pitch', 0)),
            float(params.get('yaw', 0))
        ),
        'speed_level': lambda: sport_client.SpeedLevel(int(params.get('level', 1))),
    }

    if action in no_param_actions:
        logger.info(f"执行: {action}")
        code = no_param_actions[action]()
        return code, None

    elif action in bool_param_actions:
        method, param_name = bool_param_actions[action]
        flag_value = params.get(param_name, True)
        if isinstance(flag_value, str):
            flag_value = flag_value.lower() in ('true', '1', 'yes')
        logger.info(f"执行: {action}, {param_name}={flag_value}")
        code = method(bool(flag_value))
        return code, None

    elif action in special_actions:
        logger.info(f"执行: {action}, params={params}")
        result = special_actions[action]()
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, None

    # ========== 低级别关节控制 ==========
    elif action == 'switch_mode':
        """切换控制模式"""
        global low_level_controller, current_mode
        if not LOW_LEVEL_AVAILABLE:
            return None, "控制模块不可用"
        mode = params.get('mode', 'high')
        if mode == 'low':
            # 切换到低级别模式
            if low_level_controller is None:
                low_level_controller = LowLevelController(dds_iface)
                if not low_level_controller.init():
                    return None, "低级别控制器初始化失败"
                low_level_controller.start()
            current_mode = "low"
            return 0, {"mode": "low"}
        else:
            # 切换回高级别模式
            if low_level_controller:
                low_level_controller.stop()
                low_level_controller = None
            current_mode = "high"
            return 0, {"mode": "high"}

    elif action == 'joint_position':
        """单关节位置控制"""
        if current_mode != "low" or not low_level_controller:
            return None, "请先切换到低级别模式: switch_mode"
        joint = params.get('joint')
        position = params.get('position')
        if not joint or position is None:
            return None, "缺少必要参数: joint, position"
        kp = params.get('kp', 60.0)
        kd = params.get('kd', 5.0)
        success = low_level_controller.set_joint_position(joint, float(position), kp, kd)
        return 0 if success else -1

    elif action == 'joints_position':
        """多关节位置控制"""
        if current_mode != "low" or not low_level_controller:
            return None, "请先切换到低级别模式: switch_mode"
        joints = params.get('joints', [])
        positions = params.get('positions', [])
        if not joints or not positions or len(joints) != len(positions):
            return None, "参数错误: joints和positions长度需一致"
        kp = params.get('kp', 60.0)
        kd = params.get('kd', 5.0)
        joint_positions = {j: float(p) for j, p in zip(joints, positions)}
        success = low_level_controller.set_joints_position(joint_positions, kp, kd)
        return 0 if success else -1

    elif action == 'joint_velocity':
        """单关节速度控制"""
        if current_mode != "low" or not low_level_controller:
            return None, "请先切换到低级别模式: switch_mode"
        joint = params.get('joint')
        velocity = params.get('velocity')
        if not joint or velocity is None:
            return None, "缺少必要参数: joint, velocity"
        kd = params.get('kd', 5.0)
        success = low_level_controller.set_joint_velocity(joint, float(velocity), kd)
        return 0 if success else -1

    elif action == 'joint_torque':
        """单关节力矩控制"""
        if current_mode != "low" or not low_level_controller:
            return None, "请先切换到低级别模式: switch_mode"
        joint = params.get('joint')
        torque = params.get('torque')
        if not joint or torque is None:
            return None, "缺少必要参数: joint, torque"
        success = low_level_controller.set_joint_torque(joint, float(torque))
        return 0 if success else -1

    elif action == 'get_joint_state':
        """获取关节状态"""
        if current_mode != "low" or not low_level_controller:
            return None, "请先切换到低级别模式: switch_mode"
        joint = params.get('joint')
        state = low_level_controller.get_joint_state(joint)
        return 0, state

    elif action == 'get_imu_state':
        """获取IMU状态"""
        if current_mode != "low" or not low_level_controller:
            return None, "请先切换到低级别模式: switch_mode"
        state = low_level_controller.get_imu_state()
        return 0, state

    elif action == 'pose_preset':
        """预设姿态"""
        if current_mode != "low" or not low_level_controller:
            return None, "请先切换到低级别模式: switch_mode"
        name = params.get('name', 'stand')
        success = low_level_controller.apply_pose_preset(name)
        return 0 if success else -1

    else:
        return None, f'未知动作: {action}'


def handle_client(conn, addr, sport_client, video_client):
    """处理TCP客户端请求"""
    logger.info(f"客户端连接: {addr}")
    try:
        while True:
            data = conn.recv(65536)  # 增大缓冲区以支持图像数据
            if not data:
                break

            request_str = data.decode().strip()
            if not request_str:
                continue

            logger.info(f"收到请求: {request_str[:100]}...")

            try:
                request = json.loads(request_str)
            except json.JSONDecodeError as e:
                result = {'success': False, 'error': f'JSON解析错误: {e}'}
                conn.send((json.dumps(result, ensure_ascii=False) + '\n').encode())
                continue

            action = request.get('action')
            params = request.get('params', {})

            if not action:
                result = {'success': False, 'error': '缺少action字段'}
                conn.send((json.dumps(result, ensure_ascii=False) + '\n').encode())
                continue

            try:
                code, data_result = execute_action(sport_client, video_client, action, params)

                if isinstance(code, str) and code.startswith('error'):
                    result = {'success': False, 'error': data_result}
                elif code is None:
                    result = {'success': False, 'error': data_result}
                else:
                    result = {
                        'success': code == 0,
                        'code': code,
                        'action': action
                    }
                    if data_result:
                        result['data'] = data_result
                    logger.info(f"结果: code={code}")

            except Exception as e:
                traceback.print_exc()
                result = {'success': False, 'error': str(e)}

            response = json.dumps(result, ensure_ascii=False) + '\n'
            conn.send(response.encode())

    except ConnectionResetError:
        logger.info(f"客户端断开: {addr}")
    except Exception as e:
        logger.error(f"处理异常: {e}")
        traceback.print_exc()
    finally:
        conn.close()
        logger.info(f"连接关闭: {addr}")


def main():
    global dds_iface, vui_client, log_tts_bridge, doubao_tts_client

    iface = sys.argv[1] if len(sys.argv) > 1 else None
    dds_iface = iface or APP_CONFIG.dds.iface

    if INIT_VOLUME_ON_START:
        _initialize_volume_state()

    logger.info("=" * 60)
    logger.info("  Go2 代理服务端 - TCP版本")
    logger.info(f"  监听端口: {TCP_PORT}")
    logger.info("  支持功能: 运动控制 + 摄像头 + 本地Doubao流式TTS + 日志转TTS桥接")
    logger.info(
        "  本地TTS默认播放: backend_mode=%s fallback=%s local_backend=%s audio_device=%s robot_ip=%s robot_mode=%s megaphone=%s streaming=%s streaming_chunk_ms=%s volume=%.2f auto_start=%s",
        TTS_CONFIG.backend_mode,
        TTS_CONFIG.fallback_backend_mode,
        TTS_CONFIG.local.preferred_backend,
        TTS_CONFIG.local.audio_device or "default",
        TTS_CONFIG.robot.robot_ip or "-",
        TTS_CONFIG.robot.connection_mode,
        TTS_CONFIG.robot.use_megaphone,
        TTS_CONFIG.robot.streaming_enabled,
        TTS_CONFIG.robot.streaming_chunk_ms,
        current_volume,
        TTS_CONFIG.auto_start,
    )
    logger.info(
        "  豆包默认配置: app_id=%s resource_id=%s speaker=%s audio_format=%s sample_rate=%s send_app_key=%s send_app_id=%s",
        DOUBAO_CONFIG.app_id or "-",
        DOUBAO_CONFIG.resource_id,
        DOUBAO_CONFIG.default_speaker or "-",
        DOUBAO_CONFIG.default_audio_format,
        DOUBAO_CONFIG.default_sample_rate,
        DOUBAO_CONFIG.send_app_key_header,
        DOUBAO_CONFIG.send_app_id_header,
    )
    logger.info(
        "  日志转TTS桥接: enabled=%s available=%s log_path=%s require_root=%s speaker=%s resource_id=%s flush=%.2fs quiet=%.2fs",
        LOG_TTS_CONFIG.enabled,
        LOG_TTS_BRIDGE_AVAILABLE,
        LOG_TTS_CONFIG.log_path,
        LOG_TTS_CONFIG.require_root,
        LOG_TTS_CONFIG.speaker or "-",
        LOG_TTS_CONFIG.resource_id,
        LOG_TTS_CONFIG.flush_interval_sec,
        LOG_TTS_CONFIG.quiet_period_sec,
    )
    logger.info(
        "  音量控制: mixer_device=%s mixer_control=%s init_volume_on_start=%s audio_cmd_timeout=%.1fs",
        VOLUME_MIXER_DEVICE,
        VOLUME_MIXER_CONTROL or "auto",
        INIT_VOLUME_ON_START,
        AUDIO_COMMAND_TIMEOUT,
    )
    logger.info("=" * 60)

    # 初始化Go2客户端
    try:
        sport_client, video_client, vui_client = init_go2_client(dds_iface)
        logger.info("Go2客户端初始化成功")
    except Exception as e:
        logger.error(f"Go2客户端初始化失败: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 测试连接
    logger.info("测试连接...")
    code = sport_client.Damp()
    if code == 0:
        logger.info("Go2响应正常")
    else:
        logger.warning(f"Go2返回错误码: {code}")

    if REALTIME_TTS_AVAILABLE and TTS_CONFIG.auto_start:
        try:
            code, status = _tts_player_start_action({})
            if code == 0:
                logger.info("本地TTS运行时已自动启动: %s", status)
        except Exception as e:
            logger.error(f"本地TTS运行时自动启动失败: {e}")

    # 创建TCP Socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((TCP_HOST, TCP_PORT))
    server.listen(5)

    logger.info(f"TCP服务已启动: {TCP_HOST}:{TCP_PORT}")
    logger.info("等待客户端连接...")

    try:
        while True:
            try:
                conn, addr = server.accept()
                thread = threading.Thread(
                    target=handle_client,
                    args=(conn, addr, sport_client, video_client)
                )
                thread.daemon = True
                thread.start()
            except KeyboardInterrupt:
                logger.info("收到退出信号")
                break
            except Exception as e:
                logger.error(f"错误: {e}")
    finally:
        _stop_tts_runtime()
        try:
            sport_client.StandDown()
            logger.info("已执行StandDown")
        except:
            pass
        server.close()
        logger.info("服务已退出")


if __name__ == "__main__":
    main()
