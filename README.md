# nuwax_robot_bridge

`nuwax_robot_bridge` 是一个运行在 Go2 侧的 TCP 代理服务，入口是 `go2_proxy_server.py`。它把 Unitree Go2 的高层运动控制、摄像头抓图、可选低级别关节控制、可选的 Doubao 实时 TTS / 日志播报，以及音量控制整合到了同一个进程里。

这个目录现在补了 `requirements.txt` 用来记录 Python 依赖，但仍然没有 `Dockerfile` 或现成的部署脚本；CycloneDDS、本地音频命令和 Go2 运行环境仍然需要宿主机自行准备。

如果你当前只需要控制、摄像头或 low-level 控制，不需要 TTS，那么 Doubao、播放链路和日志桥接相关内容都可以跳过。

## 功能概览

- 高层控制：直接调用 `SportClient`，支持站立、坐下、翻转、舞蹈、移动、姿态调整等动作。
- 摄像头：支持单帧抓图并直接返回 base64，或保存为本地文件。
- 低级控制：支持切换到 low-level 模式后按关节发送位置、速度、力矩命令，并读取 IMU / 关节状态。
- 可选 TTS：本进程内直连 Doubao WebSocket，不再经过本地 HTTP 或额外 playback websocket。
- 可选播放链路：优先走 `robot_webrtc` 播放到 Go2 机身扬声器，失败时可回退到本地 `alsa_local`。
- 可选日志播报：可监听 Nuwax 客户端日志 `main.log`，把模型输出实时转成语音播报。
- 音量控制：优先走 Go2 的 `VuiClient`，本地 `wpctl` / `pactl` / `amixer` 作为兼容链路。

## 目录结构

- `go2_proxy_server.py`：TCP 服务入口、动作分发、音量控制、TTS 运行时托管。
- `settings.py`：统一配置解析，读取 `.env` / `.config` 并构建结构化配置。
- `requirements.txt`：当前项目的 Python 依赖清单。
- `tts/`：Doubao 协议、实时客户端、播放管理器、日志转 TTS 桥接。
- `control/`：低级别关节控制器、关节映射与安全限位。
- `.env`：当前部署使用的配置文件。
- `go2_proxy.log`：服务日志。

## 运行链路

```text
TCP Client
    |
    v
go2_proxy_server.py
    |-- SportClient / VideoClient / VuiClient
    |-- control.low_level_controller
    `-- optional TTS runtime
         |-- tts.doubao_realtime_client
         |-- tts.realtime_tts_player
         |    |-- robot_webrtc -> Go2 机身扬声器
         |    `-- alsa_local -> Jetson 本地音频
         `-- tts.log_tts_bridge
              `-- 监听 Nuwax main.log 自动播报
```

## 使用模式

### 基础模式

只使用运动控制、摄像头、low-level 控制和音量控制，不启用 TTS。

建议配置：

```text
TTS_PLAYER_AUTO_START=false
TTS_LOG_BRIDGE_ENABLED=false
```

这种模式下，可以跳过：

- Doubao 鉴权
- `robot_webrtc` / 本地音频播放链路
- Nuwax 宿主机日志路径确认
- 日志桥接的 root 权限要求

### TTS 增强模式

在基础模式之外，再启用 Doubao 实时 TTS，或者进一步启用 Nuwax 日志自动播报。

常见组合：

- 只启用 TTS 运行时：`TTS_PLAYER_AUTO_START=true`，`TTS_LOG_BRIDGE_ENABLED=false`
- 启用日志自动播报：`TTS_PLAYER_AUTO_START=true`，`TTS_LOG_BRIDGE_ENABLED=true`

## 运行前提

- 先安装 Python 依赖：

```bash
cd /home/ml/Deeprob/TTS_SKILL/nuwax_robot_bridge
# 首次部署或 requirements.txt 变更后执行
python3 -m pip install -r requirements.txt
```

- Python 建议使用 `3.10+`。
  当前源码使用了 `str | None` 这类新语法，且 TTS 代码依赖 `asyncio.to_thread`。项目现有日志里已经出现过 Python 3.8 下 `asyncio.to_thread` 不存在的问题。
- 当前解释器里必须能导入 `unitree_sdk2py`。
  `go2_proxy_server.py` 的高层控制导入并不会自动把 `GO2_SDK_PATH` 加到 `sys.path`；`GO2_SDK_PATH` 只在 low-level 控制器里使用。
- `requirements.txt` 只解决 Python 包，不会替你安装 CycloneDDS 原生动态库。
  如果 `unitree_sdk2py` / `cyclonedds` 导入时报 `libddsc.so` 相关错误，说明还缺系统侧的 CycloneDDS 运行时。
- 基础模式下，机器只需要能访问 Go2 所在局域网。
- 仅启用 TTS 时，还需要：
  - `unitree_webrtc_connect_leshy==2.0.7`，或通过 `TTS_PLAYER_ROBOT_PYTHONPATH` 指到对应依赖目录
  - 如果启用本地音频回退链路，宿主机至少要有 `aplay`
  - `ffplay`、`wpctl`、`pactl`、`amixer` 会被按可用性自动使用
  - 宿主机可访问 Doubao 上游 WebSocket
- 仅启用日志播报时，还需要能读取 Nuwax 客户端产生的宿主机日志文件。

## 配置方式

服务启动时会按下面优先级读取配置：

1. 当前进程环境变量
2. 同目录 `.env`
3. 同目录 `.config`（兼容旧部署）

基础模式必配：

| 变量 | 推荐值 | 说明 |
| --- | --- | --- |
| `GO2_PROXY_TCP_HOST` | `0.0.0.0` | TCP 监听地址 |
| `GO2_PROXY_TCP_PORT` | `8765` | TCP 监听端口 |
| `GO2_DDS_IFACE` | 你的 DDS 网卡 | 例如 `eth0` |
| `GO2_SDK_PATH` | SDK 路径 | low-level 控制器补充 SDK 路径 |
| `TTS_PLAYER_AUTO_START` | `false` | 基础模式下建议关闭 |
| `TTS_LOG_BRIDGE_ENABLED` | `false` | 基础模式下建议关闭 |
| `GO2_VOLUME_PREFER_VUI` | `true` | 音量优先走 VUI |

仅启用 TTS 时需要：

| 变量 | 推荐值 | 说明 |
| --- | --- | --- |
| `DOUBAO_APP_ID` | 必填 | Doubao 鉴权 |
| `DOUBAO_ACCESS_KEY` | 必填 | Doubao 鉴权 |
| `DOUBAO_DEFAULT_SPEAKER` | 按需配置 | 默认发音人 |
| `TTS_PLAYER_AUTO_START` | `true` | 启用 TTS 运行时 |
| `TTS_PLAYER_BACKEND_MODE` | `robot_webrtc` | 主播放链路 |
| `TTS_PLAYER_FALLBACK_BACKEND_MODE` | `alsa_local` | 回退播放链路 |
| `TTS_PLAYER_ROBOT_IP` | Go2 IP | 机身扬声器目标 IP |

仅启用日志播报时需要：

| 变量 | 推荐值 | 说明 |
| --- | --- | --- |
| `TTS_LOG_BRIDGE_ENABLED` | `true` | 启用日志桥接 |
| `TTS_LOG_BRIDGE_LOG_PATH` | 宿主机真实路径 | Nuwax 日志文件 |
| `TTS_LOG_BRIDGE_REQUIRE_ROOT` | 按需配置 | 是否强制 root |

补充说明：

- `TTS_LOG_BRIDGE_SPEAKER` 为空时，会自动回退到 `DOUBAO_DEFAULT_SPEAKER`。
- 如果 `TTS_PLAYER_BACKEND_MODE=robot_webrtc` 且失败，而 `TTS_PLAYER_AUTO_FALLBACK_TO_LOCAL=true`，会尝试回退到 `alsa_local`。
- `TTS_PLAYER_STREAMING_ENABLED=true` 时会尝试实时 megaphone 流式发送，但代码里已经明确标注该路径仍属实验链路。

## 启动方式

进入目录后直接启动：

```bash
cd /home/ml/Deeprob/TTS_SKILL/nuwax_robot_bridge
# 基础模式建议先在 .env 中确认：
# TTS_PLAYER_AUTO_START=false
# TTS_LOG_BRIDGE_ENABLED=false
# 首次部署或 requirements.txt 变更后执行
# python3 -m pip install -r requirements.txt
python3 go2_proxy_server.py
```

也可以显式传入 DDS 网卡，优先级高于 `.env`：

```bash
cd /home/ml/Deeprob/TTS_SKILL/nuwax_robot_bridge
# 如果启用 TTS，再在 .env 中打开对应配置
# 首次部署或 requirements.txt 变更后执行
# python3 -m pip install -r requirements.txt
python3 go2_proxy_server.py eth0
```

只有在启用日志桥接，且保持 `TTS_LOG_BRIDGE_REQUIRE_ROOT=true` 时，进程才必须以 root 权限运行。

启动后日志写入：

```text
/home/unitree/nuwax_robot_bridge/go2_proxy.log
```

## TCP 协议

服务监听 `GO2_PROXY_TCP_PORT`，请求体是 JSON，响应体也是 JSON，并以换行结尾。

请求格式：

```json
{
  "action": "move",
  "params": {
    "vx": 0.2,
    "vy": 0,
    "vyaw": 0
  }
}
```

响应格式：

```json
{
  "success": true,
  "code": 0,
  "action": "move"
}
```

建议一次只发送一个完整 JSON 请求，再读取一行响应。

## 可用动作

高层动作：

- 无参：`damp`、`balance_stand`、`stop_move`、`stand_up`、`stand_down`、`recovery_stand`、`sit`、`rise_sit`、`hello`、`stretch`、`dance1`、`dance2`、`front_flip`、`front_jump`、`front_pounce`、`left_flip`、`back_flip`、`free_walk`
- 布尔参数：`hand_stand`、`free_jump`、`free_bound`、`free_avoid`、`walk_upright`、`cross_step`、`switch_joystick`、`pose`、`classic_walk`
- 特殊参数：`move`、`move_for`、`euler`、`speed_level`

摄像头：

- `get_image`
- `save_image`

TTS / 音量：

- `volume`
- `tts_player_start`
- `tts_player_stop`
- `tts_player_status`

低级控制：

- `switch_mode`
- `joint_position`
- `joints_position`
- `joint_velocity`
- `joint_torque`
- `get_joint_state`
- `get_imu_state`
- `pose_preset`

当前代码中 `speak` 动作已经明确标记为旧接口，不应继续作为可用调用方式。

## 调用示例

查询 TTS 状态，仅在启用 TTS 时有意义：

```bash
printf '{"action":"tts_player_status"}' | nc 127.0.0.1 8765
```

设置音量到 70%：

```bash
printf '{"action":"volume","params":{"action":"set","volume":0.7}}' | nc 127.0.0.1 8765
```

控制前进 2 秒：

```bash
printf '{"action":"move_for","params":{"vx":0.2,"vy":0,"vyaw":0,"duration":2.0}}' | nc 127.0.0.1 8765
```

切到 low-level 模式：

```bash
printf '{"action":"switch_mode","params":{"mode":"low"}}' | nc 127.0.0.1 8765
```

设置单关节位置：

```bash
printf '{"action":"joint_position","params":{"joint":"FR_1","position":1.2,"kp":60,"kd":5}}' | nc 127.0.0.1 8765
```

## 常见问题

- `module 'asyncio' has no attribute 'to_thread'`
  当前运行环境太老。请切到 Python 3.10 或更高版本。
- `CycloneDDSLoaderException` / `libddsc.so`
  `requirements.txt` 只装了 Python 绑定，CycloneDDS 原生共享库和对应环境变量仍需宿主机补齐。
- `缺少 unitree_webrtc_connect`
  当前解释器缺少 `unitree_webrtc_connect_leshy==2.0.7`，或 `TTS_PLAYER_ROBOT_PYTHONPATH` 没指到正确目录。
- `缺少豆包鉴权配置`
  检查 `DOUBAO_APP_ID` / `DOUBAO_ACCESS_KEY`。
- `日志转TTS桥接需要 root 运行`
  仅在启用日志桥接时关注。要么改权限策略，要么用 root 启动。
- `日志转TTS桥接缺少 speaker`
  仅在启用日志桥接时关注。请确认 `DOUBAO_DEFAULT_SPEAKER` 或 `TTS_LOG_BRIDGE_SPEAKER` 至少有一个有效值。
- `本地TTS播放后端都启动失败`
  仅在启用 TTS 且走本地回退链路时关注，请检查 `aplay` / `ffplay`、Jetson 音频路由以及 `TTS_PLAYER_AUDIO_DEVICE`。

## 相关文档

- Nuwax Docker Client 官方部署：<https://nuwax.com/docker-client-deploy.html>
- nuwax_robot_bridge 对接 Nuwax Docker Client 的项目内补充说明：`DEPLOY_NUWAX_DOCKER.md`
