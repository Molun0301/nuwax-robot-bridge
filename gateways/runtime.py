from __future__ import annotations

from dataclasses import asdict, is_dataclass
import logging
import math
import threading
import time
from typing import Dict, Optional, Tuple, Any

from contracts.artifacts import ArtifactRetentionPolicy
from contracts.capabilities import (
    CapabilityAvailability,
    CapabilityDescriptor,
    CapabilityExecutionMode,
    CapabilityMatrix,
    CapabilityRiskLevel,
)
from contracts.events import RuntimeEventCategory
from contracts.geometry import Pose, Quaternion, Twist, Vector3
from contracts.memory import MemoryPayloadFilter
from contracts.skills import SkillCategory, SkillDescriptor
from contracts.navigation import ExploreAreaRequest, NavigationGoal, NavigationStatus
from core import CapabilityRegistry, EventBus, ResourceLockManager, RuntimeResource, SafetyGuard, StateNamespace, StateStore, TaskManager
from drivers.robots.common.assembly import RobotAssemblyBase
from gateways.artifacts import LocalArtifactStore
from gateways.errors import GatewayError
from gateways.serialization import to_jsonable
from providers import ImageProvider, MotionControl, SafetyProvider, StateProvider
from services.artifact_service import ArtifactService
from services.audio import AudioService
from services.localization import LocalizationService
from services.mapping import MappingService
from services.memory import MemoryService
from services.navigation import NavigationService
from services.observation_service import ObservationService
from services.perception import PerceptionService
from services.perception import Basic2DTrackerBackend, DetectorPipeline, MetadataDrivenDetectorBackend, SimpleSceneDescriptionBackend
from services.robot_state_service import RobotStateService
from skills import SkillRegistry
from settings import NuwaxRobotBridgeConfig


RUNTIME_LOGGER = logging.getLogger("nuwax_robot_bridge.runtime")


def _schema_object(
    properties: Optional[Dict[str, Dict[str, object]]] = None,
    *,
    required: Tuple[str, ...] = (),
) -> Dict[str, object]:
    return {
        "type": "object",
        "properties": dict(properties or {}),
        "required": list(required),
        "additionalProperties": False,
    }


def _descriptor(
    name: str,
    display_name: str,
    description: str,
    *,
    execution_mode: CapabilityExecutionMode,
    risk_level: CapabilityRiskLevel,
    input_schema: Optional[Dict[str, object]] = None,
    output_schema: Optional[Dict[str, object]] = None,
    required_resources: Tuple[str, ...] = (),
    exposed_to_agent: bool = True,
    timeout_sec: Optional[int] = None,
    cancel_supported: bool = False,
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        name=name,
        display_name=display_name,
        description=description,
        execution_mode=execution_mode,
        risk_level=risk_level,
        input_schema=input_schema or _schema_object(),
        output_schema=output_schema or _schema_object(),
        required_resources=list(required_resources),
        timeout_sec=timeout_sec,
        cancel_supported=cancel_supported,
        exposed_to_agent=exposed_to_agent,
        tags=["gateway"],
    )


def _build_perception_components(robot: RobotAssemblyBase) -> Dict[str, Any]:
    """解析机器人入口提供的默认感知处理管线。"""

    default_components = {
        "pipeline_name": "default_perception_pipeline",
        "detector_pipeline": DetectorPipeline(
            (MetadataDrivenDetectorBackend(),),
            default_backend_name="metadata_detector",
        ),
        "tracker_backend": Basic2DTrackerBackend(),
        "scene_description_backend": SimpleSceneDescriptionBackend(),
    }
    factory = getattr(robot, "build_default_perception_pipeline", None)
    if not callable(factory):
        return default_components

    custom_components = factory()
    if not isinstance(custom_components, dict):
        return default_components
    return {
        "pipeline_name": custom_components.get("pipeline_name") or default_components["pipeline_name"],
        "detector_pipeline": custom_components.get("detector_pipeline") or default_components["detector_pipeline"],
        "tracker_backend": custom_components.get("tracker_backend") or default_components["tracker_backend"],
        "scene_description_backend": custom_components.get("scene_description_backend")
        or default_components["scene_description_backend"],
    }


class GatewayRuntime:
    """宿主机能力运行时。"""

    def __init__(
        self,
        *,
        config: NuwaxRobotBridgeConfig,
        robot: RobotAssemblyBase,
        capability_registry: Optional[CapabilityRegistry] = None,
        event_bus: Optional[EventBus] = None,
        resource_lock_manager: Optional[ResourceLockManager] = None,
        safety_guard: Optional[SafetyGuard] = None,
        state_store: Optional[StateStore] = None,
        task_manager: Optional[TaskManager] = None,
        artifact_store: Optional[LocalArtifactStore] = None,
        register_default_capabilities: bool = True,
    ) -> None:
        self.config = config
        self.robot = robot
        self.capability_registry = capability_registry or CapabilityRegistry()
        self.skill_registry = SkillRegistry()
        self.event_bus = event_bus or EventBus()
        self.resource_lock_manager = resource_lock_manager or ResourceLockManager()
        self.state_store = state_store or StateStore()
        self.safety_guard = safety_guard or SafetyGuard(
            event_bus=self.event_bus,
            resource_lock_manager=self.resource_lock_manager,
        )
        self.task_manager = task_manager or TaskManager(
            event_bus=self.event_bus,
            resource_lock_manager=self.resource_lock_manager,
        )
        self.artifact_store = artifact_store or LocalArtifactStore(
            config.gateway.artifact_dir,
            config.gateway.public_base_url,
        )
        self.artifact_service = ArtifactService(
            self.artifact_store,
            retention_policy=ArtifactRetentionPolicy(
                retention_days=config.runtime_data.artifact_retention_days,
                max_count=config.runtime_data.artifact_max_count,
                max_total_bytes=config.runtime_data.artifact_max_total_bytes,
                cleanup_batch_size=config.runtime_data.artifact_cleanup_batch_size,
            ),
        )
        self.robot_state_service = RobotStateService(
            robot,
            state_store=self.state_store,
            event_bus=self.event_bus,
            history_limit=config.runtime_data.state_history_limit,
            diagnostic_history_limit=config.runtime_data.diagnostic_history_limit,
        )
        self.observation_service = ObservationService(
            provider_owner=robot,
            artifact_service=self.artifact_service,
            state_store=self.state_store,
            event_bus=self.event_bus,
            history_limit=config.runtime_data.observation_history_limit,
        )
        self.localization_service = LocalizationService(
            provider_owner=robot,
            state_store=self.state_store,
            event_bus=self.event_bus,
            history_limit=config.runtime_data.localization_history_limit,
        )
        self.mapping_service = MappingService(
            provider_owner=robot,
            state_store=self.state_store,
            event_bus=self.event_bus,
            history_limit=config.runtime_data.map_history_limit,
        )
        perception_components = _build_perception_components(robot)
        self.perception_service = PerceptionService(
            provider_owner=robot,
            artifact_service=self.artifact_service,
            state_store=self.state_store,
            detector_pipeline=perception_components["detector_pipeline"],
            tracker_backend=perception_components["tracker_backend"],
            scene_description_backend=perception_components["scene_description_backend"],
            event_bus=self.event_bus,
            history_limit=config.runtime_data.perception_history_limit,
            pipeline_name=str(perception_components["pipeline_name"]),
        )
        self.audio_service = AudioService(
            config=config,
            robot=robot,
            event_bus=self.event_bus,
        )
        self.memory_service = MemoryService(
            localization_service=self.localization_service,
            mapping_service=self.mapping_service,
            observation_service=self.observation_service,
            perception_service=self.perception_service,
            state_store=self.state_store,
            event_bus=self.event_bus,
            artifact_store=self.artifact_store,
            history_limit=config.runtime_data.memory_history_limit,
            memory_db_path=config.runtime_data.memory_db_path,
            embedding_model=config.runtime_data.memory_embedding_model,
            embedding_dimension=config.runtime_data.memory_embedding_dimension,
            image_embedding_model=config.runtime_data.memory_image_embedding_model,
            image_embedding_dimension=config.runtime_data.memory_image_embedding_dimension,
        )
        self.navigation_service = NavigationService(
            provider_owner=robot,
            localization_service=self.localization_service,
            mapping_service=self.mapping_service,
            state_store=self.state_store,
            event_bus=self.event_bus,
            history_limit=config.runtime_data.navigation_history_limit,
        )
        self._lock = threading.RLock()
        self._started = False
        self._event_log_subscription_id = self.event_bus.subscribe(self._log_runtime_event)

        self.capability_registry.bind_robot_capability_matrix(robot.manifest.capability_matrix)
        self.safety_guard.register_stop_handler("robot_entry", self._safe_stop_robot)

        if register_default_capabilities:
            self.register_default_capabilities()
            self.register_default_skills()

    def start(self) -> None:
        """启动宿主机运行时。"""

        with self._lock:
            if self._started:
                RUNTIME_LOGGER.info("宿主机运行时已在运行，忽略重复启动请求。")
                return
            RUNTIME_LOGGER.info(
                "宿主机运行时开始启动 robot=%s model=%s",
                self.robot.manifest.robot_name,
                self.robot.manifest.robot_model,
            )
            self.robot.start()
            self._started = True
        try:
            self.robot_state_service.refresh()
        except Exception:
            pass
        try:
            if self.localization_service.is_available():
                self.localization_service.refresh()
        except Exception:
            pass
        try:
            if self.mapping_service.is_available():
                self.mapping_service.refresh()
        except Exception:
            pass
        try:
            if self.navigation_service.is_navigation_available():
                self.navigation_service.refresh_navigation()
        except Exception:
            pass
        try:
            if self.navigation_service.is_exploration_available():
                self.navigation_service.refresh_exploration()
        except Exception:
            pass

        self.event_bus.publish(
            self.event_bus.build_event(
                "system.gateway_started",
                category=RuntimeEventCategory.SYSTEM,
                source="gateway_runtime",
                message="宿主机网关运行时已启动。",
                payload={
                    "robot_name": self.robot.manifest.robot_name,
                    "robot_model": self.robot.manifest.robot_model,
                },
            )
        )

    def stop(self) -> None:
        """停止宿主机运行时。"""

        with self._lock:
            if not self._started:
                RUNTIME_LOGGER.info("宿主机运行时未启动，忽略停止请求。")
                return
            RUNTIME_LOGGER.info("宿主机运行时开始停止。")
            self.safety_guard.stop_all_motion("宿主机网关关闭。")
            self.task_manager.shutdown(wait=False)
            self.audio_service.stop()
            self.robot.stop()
            self._started = False

        self.event_bus.publish(
            self.event_bus.build_event(
                "system.gateway_stopped",
                category=RuntimeEventCategory.SYSTEM,
                source="gateway_runtime",
                message="宿主机网关运行时已停止。",
            )
        )

    def _log_runtime_event(self, event) -> None:
        """把运行时事件同步写入标准日志。"""

        try:
            level = {
                "info": logging.INFO,
                "warning": logging.WARNING,
                "error": logging.ERROR,
            }.get(str(event.severity), logging.INFO)
            payload_preview = self._build_log_preview(event.payload)
            metadata_preview = self._build_log_preview(event.metadata)
            RUNTIME_LOGGER.log(
                level,
                "运行时事件 category=%s type=%s source=%s task_id=%s subject_id=%s message=%s payload=%s metadata=%s",
                event.category,
                event.event_type,
                event.source,
                event.task_id or "-",
                event.subject_id or "-",
                event.message or "-",
                payload_preview,
                metadata_preview,
            )
        except Exception:
            RUNTIME_LOGGER.exception("写入运行时事件日志失败。")

    @staticmethod
    def _build_log_preview(value: Any) -> str:
        """把结构化载荷压缩成适合日志的一行文本。"""

        if not value:
            return "-"
        try:
            rendered = str(to_jsonable(value))
        except Exception:
            rendered = str(value)
        if len(rendered) > 600:
            return f"{rendered[:600]}...(truncated)"
        return rendered

    def register_default_capabilities(self) -> None:
        """注册首版网关能力。"""

        capabilities = (
            _descriptor(
                "get_robot_status",
                "获取机器人状态",
                "读取机器人当前状态、装配状态和安全状态。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.LOW,
                output_schema=_schema_object(
                    {
                        "robot_name": {"type": "string"},
                        "robot_model": {"type": "string"},
                        "assembly_status": {"type": "object"},
                        "robot_state": {"type": "object"},
                        "safety_state": {"type": "object"},
                    }
                ),
            ),
            _descriptor(
                "get_active_tasks",
                "获取活动任务",
                "返回当前活动任务列表和最近历史任务。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.LOW,
                input_schema=_schema_object({"history_limit": {"type": "integer", "minimum": 0}}),
                output_schema=_schema_object(
                    {
                        "active_tasks": {"type": "array"},
                        "history_tasks": {"type": "array"},
                    }
                ),
            ),
            _descriptor(
                "list_capabilities",
                "列出能力与技能",
                "返回当前运行时能力和技能工具面摘要，供上层智能体理解可调用动作集合。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.LOW,
                input_schema=_schema_object(
                    {
                        "include_unsupported": {"type": "boolean", "default": False},
                    }
                ),
                output_schema=_schema_object(
                    {
                        "skills": {"type": "array"},
                        "capabilities": {"type": "array"},
                    }
                ),
            ),
            _descriptor(
                "get_task_status",
                "获取任务状态",
                "按任务标识读取状态和事件。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.LOW,
                input_schema=_schema_object({"task_id": {"type": "string"}}, required=("task_id",)),
                output_schema=_schema_object({"task": {"type": "object"}}),
            ),
            _descriptor(
                "cancel_task",
                "取消任务",
                "取消一个仍在执行中的异步任务。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.MEDIUM,
                input_schema=_schema_object(
                    {
                        "task_id": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    required=("task_id",),
                ),
                output_schema=_schema_object({"cancelled": {"type": "boolean"}, "task": {"type": "object"}}),
            ),
            _descriptor(
                "stop_all_motion",
                "安全停止",
                "触发全局安全停止，立即停止底盘运动。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.MEDIUM,
                input_schema=_schema_object({"reason": {"type": "string"}}),
                output_schema=_schema_object({"stopped": {"type": "boolean"}}),
            ),
            _descriptor(
                "capture_image",
                "抓取图像",
                "抓取单帧图像并生成观察上下文与宿主机制品。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.LOW,
                input_schema=_schema_object({"camera_id": {"type": "string"}}),
                output_schema=_schema_object(
                    {
                        "artifact": {"type": "object"},
                        "observation_context": {"type": "object"},
                    }
                ),
            ),
            _descriptor(
                "get_joint_state",
                "获取关节状态",
                "读取当前全部关节状态，或按名称筛选单个关节。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.LOW,
                input_schema=_schema_object({"joint_name": {"type": "string"}}),
                output_schema=_schema_object({"joints": {"type": "array"}, "joint": {"type": "object"}}),
            ),
            _descriptor(
                "get_imu_state",
                "获取 IMU 状态",
                "读取当前 IMU 姿态、角速度与线加速度。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.LOW,
                output_schema=_schema_object({"imu_state": {"type": "object"}}),
            ),
            _descriptor(
                "get_latest_observation",
                "获取最新观察",
                "读取当前缓存的最新观察结果，可按相机筛选。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.LOW,
                input_schema=_schema_object({"camera_id": {"type": "string"}}),
                output_schema=_schema_object({"observation_context": {"type": "object"}}),
            ),
            _descriptor(
                "perceive_current_scene",
                "感知当前场景",
                "抓取当前图像并执行标准检测、跟踪与场景语义摘要。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.LOW,
                input_schema=_schema_object(
                    {
                        "camera_id": {"type": "string"},
                        "detector_backend": {"type": "string"},
                    }
                ),
                output_schema=_schema_object({"perception_context": {"type": "object"}}),
            ),
            _descriptor(
                "describe_current_scene",
                "描述当前场景",
                "返回场景摘要文本和结构化感知结果。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.LOW,
                input_schema=_schema_object(
                    {
                        "camera_id": {"type": "string"},
                        "detector_backend": {"type": "string"},
                        "refresh": {"type": "boolean", "default": True},
                    }
                ),
                output_schema=_schema_object(
                    {
                        "summary": {"type": "string"},
                        "scene_summary": {"type": "object"},
                        "perception_context": {"type": "object"},
                    }
                ),
            ),
            _descriptor(
                "get_latest_perception",
                "获取最新感知",
                "读取当前缓存的最新感知结果，可按相机筛选。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.LOW,
                input_schema=_schema_object({"camera_id": {"type": "string"}}),
                output_schema=_schema_object({"perception_context": {"type": "object"}}),
            ),
            _descriptor(
                "get_localization_snapshot",
                "获取定位快照",
                "读取当前定位与 TF 快照，必要时从本地提供器刷新。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.LOW,
                output_schema=_schema_object({"localization_snapshot": {"type": "object"}}),
            ),
            _descriptor(
                "get_map_snapshot",
                "获取地图快照",
                "读取当前地图快照，必要时从本地图提供器刷新。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.LOW,
                output_schema=_schema_object({"map_snapshot": {"type": "object"}}),
            ),
            _descriptor(
                "get_navigation_snapshot",
                "获取导航快照",
                "读取当前导航与探索上下文，必要时主动刷新后端状态。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.LOW,
                output_schema=_schema_object(
                    {
                        "navigation_context": {"type": "object"},
                        "exploration_context": {"type": "object"},
                    }
                ),
            ),
            _descriptor(
                "tag_location",
                "标记当前位置",
                "把当前位姿与最新观察信息写入命名地点记忆，可供后续命名导航复用。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.LOW,
                input_schema=_schema_object(
                    {
                        "name": {"type": "string"},
                        "aliases": {"type": "array"},
                        "description": {"type": "string"},
                        "camera_id": {"type": "string"},
                        "auto_create_memory": {"type": "boolean", "default": True},
                    },
                    required=("name",),
                ),
                output_schema=_schema_object(
                    {
                        "tagged_location": {"type": "object"},
                        "semantic_memory": {"type": "object"},
                    }
                ),
            ),
            _descriptor(
                "query_location",
                "查询地点记忆",
                "按名称、别名或描述查询已标记地点。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.LOW,
                input_schema=_schema_object(
                    {
                        "query": {"type": "string"},
                        "similarity_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.55},
                        "limit": {"type": "integer", "minimum": 1, "default": 5},
                        "map_version_id": {"type": "string"},
                        "camera_id": {"type": "string"},
                        "semantic_labels": {"type": "array"},
                        "visual_labels": {"type": "array"},
                        "vision_tags": {"type": "array"},
                        "topo_node_id": {"type": "string"},
                        "max_age_sec": {"type": "number", "minimum": 0.0},
                        "near_frame_id": {"type": "string"},
                        "near_x": {"type": "number"},
                        "near_y": {"type": "number"},
                        "near_z": {"type": "number", "default": 0.0},
                        "max_distance_m": {"type": "number", "minimum": 0.0},
                    },
                    required=("query",),
                ),
                output_schema=_schema_object({"query_result": {"type": "object"}}),
            ),
            _descriptor(
                "query_semantic_memory",
                "查询语义记忆",
                "按自然语言检索近期场景记忆和语义摘要。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.LOW,
                input_schema=_schema_object(
                    {
                        "query": {"type": "string"},
                        "similarity_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.25},
                        "limit": {"type": "integer", "minimum": 1, "default": 5},
                        "map_version_id": {"type": "string"},
                        "camera_id": {"type": "string"},
                        "linked_location_id": {"type": "string"},
                        "semantic_labels": {"type": "array"},
                        "visual_labels": {"type": "array"},
                        "vision_tags": {"type": "array"},
                        "topo_node_id": {"type": "string"},
                        "max_age_sec": {"type": "number", "minimum": 0.0},
                        "near_frame_id": {"type": "string"},
                        "near_x": {"type": "number"},
                        "near_y": {"type": "number"},
                        "near_z": {"type": "number", "default": 0.0},
                        "max_distance_m": {"type": "number", "minimum": 0.0},
                    },
                    required=("query",),
                ),
                output_schema=_schema_object({"query_result": {"type": "object"}}),
            ),
            _descriptor(
                "remember_current_scene",
                "记住当前场景",
                "把当前观察或感知结果写入语义记忆。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.LOW,
                input_schema=_schema_object(
                    {
                        "title": {"type": "string"},
                        "camera_id": {"type": "string"},
                        "summary": {"type": "string"},
                    }
                ),
                output_schema=_schema_object({"semantic_memory": {"type": "object"}}),
            ),
            _descriptor(
                "navigate_to_pose",
                "导航到位姿",
                "以统一导航接口把机器人导航到二维目标位姿。",
                execution_mode=CapabilityExecutionMode.ASYNC_TASK,
                risk_level=CapabilityRiskLevel.MEDIUM,
                input_schema=_schema_object(
                    {
                        "frame_id": {"type": "string"},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number", "default": 0.0},
                        "yaw_rad": {"type": "number", "default": 0.0},
                        "tolerance_position_m": {"type": "number", "minimum": 0.0, "default": 0.3},
                        "tolerance_yaw_rad": {"type": "number", "minimum": 0.0, "default": 0.3},
                        "poll_interval_sec": {"type": "number", "exclusiveMinimum": 0.0, "default": 0.1},
                        "timeout_sec": {"type": "number", "exclusiveMinimum": 0.0},
                    },
                    required=("frame_id", "x", "y"),
                ),
                output_schema=_schema_object({"task": {"type": "object"}, "status": {"type": "object"}}),
                required_resources=(RuntimeResource.BASE_MOTION.value, RuntimeResource.NAVIGATION.value),
                timeout_sec=180,
                cancel_supported=True,
            ),
            _descriptor(
                "navigate_to_named_location",
                "导航到命名位置",
                "优先使用已标记地点，其次使用语义记忆候选，把自然语言目标解析成导航目标并可在到点后复核。",
                execution_mode=CapabilityExecutionMode.ASYNC_TASK,
                risk_level=CapabilityRiskLevel.MEDIUM,
                input_schema=_schema_object(
                    {
                        "target_name": {"type": "string"},
                        "camera_id": {"type": "string"},
                        "verify_on_arrival": {"type": "boolean", "default": True},
                        "verify_similarity_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.55},
                        "poll_interval_sec": {"type": "number", "exclusiveMinimum": 0.0, "default": 0.1},
                        "timeout_sec": {"type": "number", "exclusiveMinimum": 0.0},
                    },
                    required=("target_name",),
                ),
                output_schema=_schema_object({"task": {"type": "object"}, "status": {"type": "object"}}),
                required_resources=(RuntimeResource.BASE_MOTION.value, RuntimeResource.NAVIGATION.value),
                timeout_sec=240,
                cancel_supported=True,
            ),
            _descriptor(
                "explore_area",
                "探索区域",
                "以统一探索接口发起前沿或覆盖式探索任务。",
                execution_mode=CapabilityExecutionMode.ASYNC_TASK,
                risk_level=CapabilityRiskLevel.MEDIUM,
                input_schema=_schema_object(
                    {
                        "target_name": {"type": "string"},
                        "frame_id": {"type": "string"},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number", "default": 0.0},
                        "radius_m": {"type": "number", "exclusiveMinimum": 0.0},
                        "strategy": {"type": "string", "default": "frontier"},
                        "max_duration_sec": {"type": "number", "exclusiveMinimum": 0.0},
                        "poll_interval_sec": {"type": "number", "exclusiveMinimum": 0.0, "default": 0.2},
                        "timeout_sec": {"type": "number", "exclusiveMinimum": 0.0},
                    }
                ),
                output_schema=_schema_object({"task": {"type": "object"}, "status": {"type": "object"}}),
                required_resources=(RuntimeResource.BASE_MOTION.value, RuntimeResource.NAVIGATION.value),
                timeout_sec=300,
                cancel_supported=True,
            ),
            _descriptor(
                "inspect_target",
                "检查目标",
                "检查当前目标是否出现在画面中；必要时可先导航到命名地点后再观察。",
                execution_mode=CapabilityExecutionMode.ASYNC_TASK,
                risk_level=CapabilityRiskLevel.MEDIUM,
                input_schema=_schema_object(
                    {
                        "query": {"type": "string"},
                        "camera_id": {"type": "string"},
                        "detector_backend": {"type": "string"},
                        "auto_navigate": {"type": "boolean", "default": False},
                        "poll_interval_sec": {"type": "number", "exclusiveMinimum": 0.0, "default": 0.1},
                        "timeout_sec": {"type": "number", "exclusiveMinimum": 0.0},
                    },
                    required=("query",),
                ),
                output_schema=_schema_object({"task": {"type": "object"}, "status": {"type": "object"}}),
                required_resources=(RuntimeResource.BASE_MOTION.value, RuntimeResource.NAVIGATION.value),
                timeout_sec=120,
                cancel_supported=True,
            ),
            _descriptor(
                "follow_target",
                "跟随目标",
                "基于当前感知结果对指定目标执行首版二维视觉跟随。",
                execution_mode=CapabilityExecutionMode.ASYNC_TASK,
                risk_level=CapabilityRiskLevel.MEDIUM,
                input_schema=_schema_object(
                    {
                        "query": {"type": "string"},
                        "camera_id": {"type": "string"},
                        "detector_backend": {"type": "string"},
                        "duration_sec": {"type": "number", "exclusiveMinimum": 0.0, "default": 2.0},
                        "interval_sec": {"type": "number", "exclusiveMinimum": 0.0, "default": 0.1},
                        "lost_timeout_sec": {"type": "number", "exclusiveMinimum": 0.0, "default": 0.8},
                        "timeout_sec": {"type": "number", "exclusiveMinimum": 0.0},
                    },
                    required=("query",),
                ),
                output_schema=_schema_object({"task": {"type": "object"}, "status": {"type": "object"}}),
                required_resources=(RuntimeResource.BASE_MOTION.value, RuntimeResource.CAMERA_OBSERVATION.value),
                timeout_sec=60,
                cancel_supported=True,
            ),
            _descriptor(
                "relative_move",
                "相对移动",
                "按给定速度和时长执行短时移动，返回异步任务。",
                execution_mode=CapabilityExecutionMode.ASYNC_TASK,
                risk_level=CapabilityRiskLevel.MEDIUM,
                input_schema=_schema_object(
                    {
                        "vx": {"type": "number"},
                        "vy": {"type": "number", "default": 0.0},
                        "vyaw": {"type": "number", "default": 0.0},
                        "duration_sec": {"type": "number", "exclusiveMinimum": 0.0},
                        "interval_sec": {"type": "number", "exclusiveMinimum": 0.0, "default": 0.05},
                        "timeout_sec": {"type": "number", "exclusiveMinimum": 0.0},
                    },
                    required=("vx", "duration_sec"),
                ),
                output_schema=_schema_object({"task": {"type": "object"}, "status": {"type": "object"}}),
                required_resources=(RuntimeResource.BASE_MOTION.value,),
                timeout_sec=30,
                cancel_supported=True,
            ),
            _descriptor(
                "execute_sport_command",
                "执行机身动作",
                "执行机器人入口已实现的高层动作命令。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.MEDIUM,
                input_schema=_schema_object(
                    {
                        "action": {"type": "string"},
                        "params": {"type": "object"},
                    },
                    required=("action",),
                ),
                output_schema=_schema_object({"code": {"type": "integer"}, "result": {"type": "object"}}),
            ),
            _descriptor(
                "set_body_pose",
                "设置机身姿态",
                "通过机身欧拉角控制机器人姿态。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.MEDIUM,
                input_schema=_schema_object(
                    {
                        "roll": {"type": "number", "default": 0.0},
                        "pitch": {"type": "number", "default": 0.0},
                        "yaw": {"type": "number", "default": 0.0},
                    }
                ),
                output_schema=_schema_object({"code": {"type": "integer"}, "result": {"type": "object"}}),
            ),
            _descriptor(
                "set_speed_level",
                "设置速度档位",
                "设置机器人运动速度档位。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.MEDIUM,
                input_schema=_schema_object(
                    {
                        "level": {"type": "integer", "minimum": 1, "maximum": 3},
                    },
                    required=("level",),
                ),
                output_schema=_schema_object({"code": {"type": "integer"}, "result": {"type": "object"}}),
            ),
            _descriptor(
                "get_volume",
                "获取音量",
                "读取当前机器人音量与软件播报音量状态。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.LOW,
                output_schema=_schema_object({"volume_state": {"type": "object"}}),
            ),
            _descriptor(
                "set_volume",
                "设置音量",
                "设置机器人音量与软件播报音量。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.MEDIUM,
                input_schema=_schema_object(
                    {
                        "volume": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    required=("volume",),
                ),
                output_schema=_schema_object({"volume_state": {"type": "object"}}),
            ),
            _descriptor(
                "speak_text",
                "播报文本",
                "提交一段中文文本到宿主机播报服务；未配置实时语音时退化为记录模式。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.LOW,
                input_schema=_schema_object(
                    {
                        "text": {"type": "string"},
                    },
                    required=("text",),
                ),
                output_schema=_schema_object({"speech": {"type": "object"}}),
            ),
            _descriptor(
                "switch_control_mode",
                "切换控制模式",
                "切换机器人高层/低层控制模式，默认仅管理员可见。",
                execution_mode=CapabilityExecutionMode.SYNC,
                risk_level=CapabilityRiskLevel.ADMIN,
                input_schema=_schema_object({"mode": {"type": "string", "enum": ["high", "low"]}}, required=("mode",)),
                output_schema=_schema_object({"code": {"type": "integer"}, "result": {"type": "object"}}),
                exposed_to_agent=False,
            ),
        )

        handlers = {
            "get_robot_status": self._handle_get_robot_status,
            "get_active_tasks": self._handle_get_active_tasks,
            "list_capabilities": self._handle_list_capabilities,
            "get_task_status": self._handle_get_task_status,
            "cancel_task": self._handle_cancel_task,
            "stop_all_motion": self._handle_stop_all_motion,
            "capture_image": self._handle_capture_image,
            "get_joint_state": self._handle_get_joint_state,
            "get_imu_state": self._handle_get_imu_state,
            "get_latest_observation": self._handle_get_latest_observation,
            "perceive_current_scene": self._handle_perceive_current_scene,
            "describe_current_scene": self._handle_describe_current_scene,
            "get_latest_perception": self._handle_get_latest_perception,
            "get_localization_snapshot": self._handle_get_localization_snapshot,
            "get_map_snapshot": self._handle_get_map_snapshot,
            "get_navigation_snapshot": self._handle_get_navigation_snapshot,
            "tag_location": self._handle_tag_location,
            "query_location": self._handle_query_location,
            "query_semantic_memory": self._handle_query_semantic_memory,
            "remember_current_scene": self._handle_remember_current_scene,
            "navigate_to_pose": self._handle_navigate_to_pose,
            "navigate_to_named_location": self._handle_navigate_to_named_location,
            "explore_area": self._handle_explore_area,
            "inspect_target": self._handle_inspect_target,
            "follow_target": self._handle_follow_target,
            "relative_move": self._handle_relative_move,
            "execute_sport_command": self._handle_execute_sport_command,
            "set_body_pose": self._handle_set_body_pose,
            "set_speed_level": self._handle_set_speed_level,
            "get_volume": self._handle_get_volume,
            "set_volume": self._handle_set_volume,
            "speak_text": self._handle_speak_text,
            "switch_control_mode": self._handle_switch_control_mode,
        }

        for descriptor in capabilities:
            self.capability_registry.register(
                descriptor,
                handlers[descriptor.name],
                owner="gateway_runtime",
                overwrite=True,
            )

        existing_matrix = self.robot.manifest.capability_matrix
        merged_capabilities = {item.name: item for item in existing_matrix.capabilities}
        for descriptor in capabilities:
            merged_capabilities[descriptor.name] = self._build_gateway_capability_availability(
                descriptor.name,
                fallback=merged_capabilities.get(descriptor.name),
            )
        self.capability_registry.bind_robot_capability_matrix(
            CapabilityMatrix(
                robot_model=self.robot.manifest.robot_model,
                capabilities=list(merged_capabilities.values()),
                metadata={
                    **dict(existing_matrix.metadata),
                    "runtime_gateway": "enabled",
                },
            )
        )

    def register_default_skills(self) -> None:
        """注册首版技能工具面。"""

        skills = (
            SkillDescriptor(
                name="list_capabilities",
                display_name="列出能力与技能",
                description="返回当前机器人对上层智能体可见的工具集合与运行时能力摘要。",
                category=SkillCategory.SYSTEM,
                capability_name="list_capabilities",
            ),
            SkillDescriptor(
                name="get_robot_status",
                display_name="获取机器人状态",
                description="读取机器人当前状态、装配状态和安全状态。",
                category=SkillCategory.SYSTEM,
                capability_name="get_robot_status",
            ),
            SkillDescriptor(
                name="get_active_tasks",
                display_name="获取活动任务",
                description="返回当前活动任务与最近历史任务。",
                category=SkillCategory.SYSTEM,
                capability_name="get_active_tasks",
            ),
            SkillDescriptor(
                name="cancel_task",
                display_name="取消任务",
                description="取消一个仍在执行中的后台任务。",
                category=SkillCategory.SYSTEM,
                capability_name="cancel_task",
            ),
            SkillDescriptor(
                name="stop_all_motion",
                display_name="安全停止",
                description="立即触发全局安全停止。",
                category=SkillCategory.SYSTEM,
                capability_name="stop_all_motion",
            ),
            SkillDescriptor(
                name="get_localization_snapshot",
                display_name="获取定位快照",
                description="读取当前定位和 TF 快照。",
                category=SkillCategory.SYSTEM,
                capability_name="get_localization_snapshot",
            ),
            SkillDescriptor(
                name="get_map_snapshot",
                display_name="获取地图快照",
                description="读取当前地图快照和地图版本信息。",
                category=SkillCategory.SYSTEM,
                capability_name="get_map_snapshot",
            ),
            SkillDescriptor(
                name="get_navigation_snapshot",
                display_name="获取导航快照",
                description="读取当前导航和探索上下文。",
                category=SkillCategory.SYSTEM,
                capability_name="get_navigation_snapshot",
            ),
            SkillDescriptor(
                name="capture_image",
                display_name="抓取图像",
                description="抓取当前单帧图像并返回观察上下文与图像制品。",
                category=SkillCategory.OBSERVATION,
                capability_name="capture_image",
            ),
            SkillDescriptor(
                name="describe_current_scene",
                display_name="描述当前场景",
                description="返回当前场景的中文摘要和结构化感知结果。",
                category=SkillCategory.OBSERVATION,
                capability_name="describe_current_scene",
            ),
            SkillDescriptor(
                name="get_joint_state",
                display_name="获取关节状态",
                description="读取当前全部关节状态，或按名称筛选。",
                category=SkillCategory.OBSERVATION,
                capability_name="get_joint_state",
            ),
            SkillDescriptor(
                name="get_imu_state",
                display_name="获取 IMU 状态",
                description="读取当前 IMU 姿态、角速度和线加速度。",
                category=SkillCategory.OBSERVATION,
                capability_name="get_imu_state",
            ),
            SkillDescriptor(
                name="relative_move",
                display_name="相对移动",
                description="按给定速度和时长执行短时底盘移动。",
                category=SkillCategory.MOTION,
                capability_name="relative_move",
            ),
            SkillDescriptor(
                name="execute_sport_command",
                display_name="执行机身动作",
                description="执行机器人入口支持的高层动作命令。",
                category=SkillCategory.MOTION,
                capability_name="execute_sport_command",
            ),
            SkillDescriptor(
                name="set_body_pose",
                display_name="设置机身姿态",
                description="通过欧拉角控制机身姿态。",
                category=SkillCategory.MOTION,
                capability_name="set_body_pose",
            ),
            SkillDescriptor(
                name="set_speed_level",
                display_name="设置速度档位",
                description="设置机器人运动速度档位。",
                category=SkillCategory.MOTION,
                capability_name="set_speed_level",
            ),
            SkillDescriptor(
                name="get_volume",
                display_name="获取音量",
                description="读取机器人音量和软件播报音量。",
                category=SkillCategory.AUDIO,
                capability_name="get_volume",
            ),
            SkillDescriptor(
                name="set_volume",
                display_name="设置音量",
                description="设置机器人音量和软件播报音量。",
                category=SkillCategory.AUDIO,
                capability_name="set_volume",
            ),
            SkillDescriptor(
                name="speak_text",
                display_name="播报文本",
                description="把一段文本提交给宿主机播报服务。",
                category=SkillCategory.AUDIO,
                capability_name="speak_text",
            ),
            SkillDescriptor(
                name="tag_location",
                display_name="标记当前位置",
                description="把当前位姿与最新观察写入命名地点记忆。",
                category=SkillCategory.MEMORY,
                capability_name="tag_location",
            ),
            SkillDescriptor(
                name="query_location",
                display_name="查询地点记忆",
                description="按名称、别名或描述查询已标记地点。",
                category=SkillCategory.MEMORY,
                capability_name="query_location",
            ),
            SkillDescriptor(
                name="query_semantic_memory",
                display_name="查询语义记忆",
                description="按自然语言检索语义记忆。",
                category=SkillCategory.MEMORY,
                capability_name="query_semantic_memory",
            ),
            SkillDescriptor(
                name="remember_current_scene",
                display_name="记住当前场景",
                description="把当前观察或感知结果写入语义记忆。",
                category=SkillCategory.MEMORY,
                capability_name="remember_current_scene",
            ),
            SkillDescriptor(
                name="navigate_to_pose",
                display_name="导航到位姿",
                description="把机器人导航到指定二维目标位姿。",
                category=SkillCategory.TASK,
                capability_name="navigate_to_pose",
            ),
            SkillDescriptor(
                name="navigate_to_named_location",
                display_name="导航到命名位置",
                description="优先使用地点记忆，其次使用语义地图，导航到命名位置。",
                category=SkillCategory.TASK,
                capability_name="navigate_to_named_location",
            ),
            SkillDescriptor(
                name="explore_area",
                display_name="探索区域",
                description="发起一次前沿或覆盖式探索任务。",
                category=SkillCategory.TASK,
                capability_name="explore_area",
            ),
            SkillDescriptor(
                name="inspect_target",
                display_name="检查目标",
                description="检查目标是否出现在当前场景中，必要时先导航后观察。",
                category=SkillCategory.TASK,
                capability_name="inspect_target",
            ),
            SkillDescriptor(
                name="follow_target",
                display_name="跟随目标",
                description="基于当前相机感知对目标执行首版二维视觉跟随。",
                category=SkillCategory.TASK,
                capability_name="follow_target",
            ),
            SkillDescriptor(
                name="switch_control_mode",
                display_name="切换控制模式",
                description="管理员技能，切换高层与低层控制模式。",
                category=SkillCategory.ADMIN,
                capability_name="switch_control_mode",
                exposed_to_agent=False,
            ),
        )

        for descriptor in skills:
            self.skill_registry.register(descriptor, overwrite=True)

    def list_capabilities(self, *, exposed_only: Optional[bool] = None) -> Tuple[CapabilityDescriptor, ...]:
        """列出运行时能力描述。"""

        views = self.capability_registry.build_runtime_views(
            robot_model=self.robot.manifest.robot_model,
            exposed_only=exposed_only,
        )
        return tuple(view.descriptor for view in views if view.supported and view.runnable)

    def list_tools(
        self,
        *,
        exposed_only: Optional[bool] = None,
        include_unsupported: bool = False,
    ):
        """列出面向上层智能体的技能工具视图。"""

        return self.skill_registry.build_runtime_views(
            self.capability_registry,
            robot_model=self.robot.manifest.robot_model,
            exposed_only=exposed_only,
            include_unsupported=include_unsupported,
        )

    def get_tool_descriptor(self, tool_name: str) -> SkillDescriptor:
        """读取工具描述。"""

        return self.skill_registry.get_descriptor(tool_name)

    def invoke_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        requested_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """按技能工具名调用运行时能力。"""

        tool_descriptor = self.skill_registry.get_descriptor(tool_name)
        return self.invoke_capability(
            tool_descriptor.capability_name,
            arguments,
            requested_by=requested_by,
        )

    def invoke_capability(
        self,
        capability_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        requested_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """调用一个网关能力。"""

        registration = self.capability_registry.get_registration(capability_name)
        self.capability_registry.assert_supported(capability_name, robot_model=self.robot.manifest.robot_model)
        if registration.handler is None:
            raise GatewayError(f"能力 {capability_name} 尚未绑定处理器。")

        descriptor = registration.descriptor
        payload = dict(arguments or {})
        result = registration.handler(payload, requested_by=requested_by)

        if descriptor.execution_mode == CapabilityExecutionMode.ASYNC_TASK:
            status = self.task_manager.get_task_status(result.task_id)
            return {
                "capability_name": descriptor.name,
                "mode": descriptor.execution_mode.value,
                "task": result,
                "status": status,
            }

        self.event_bus.publish(
            self.event_bus.build_event(
                "capability.invoked",
                category=RuntimeEventCategory.SYSTEM,
                source="gateway_runtime",
                subject_id=descriptor.name,
                message=f"同步能力 {descriptor.name} 调用完成。",
                payload={"requested_by": requested_by or "unknown"},
            )
        )
        return {
            "capability_name": descriptor.name,
            "mode": descriptor.execution_mode.value,
            "result": result,
        }

    def get_health_summary(self) -> Dict[str, Any]:
        """导出健康摘要。"""

        return {
            "runtime_started": self._started,
            "robot_name": self.robot.manifest.robot_name,
            "robot_model": self.robot.manifest.robot_model,
            "assembly_status": self._serialize(self.robot.get_status()),
            "latest_robot_snapshot": self._serialize(self.robot_state_service.get_latest_snapshot()),
            "latest_localization": self._serialize(self.localization_service.get_latest_snapshot()),
            "latest_map": self._serialize(self.mapping_service.get_latest_snapshot()),
            "latest_observation": self._serialize(self.observation_service.get_latest_observation()),
            "latest_perception": self._serialize(self.perception_service.get_latest_perception()),
            "latest_navigation": self._serialize(self.navigation_service.get_latest_navigation_context()),
            "latest_exploration": self._serialize(self.navigation_service.get_latest_exploration_context()),
            "memory_summary": self.memory_service.get_summary().model_dump(mode="json"),
            "audio_state": self.audio_service.get_volume_state(),
            "active_task_count": len(self.task_manager.list_active_tasks()),
            "latest_event_cursor": self.event_bus.latest_cursor(),
            "artifact_dir": str(self.artifact_store.base_dir),
            "public_base_url": self.config.gateway.public_base_url,
            "artifact_summary": self.artifact_service.get_summary().model_dump(mode="json"),
            "resource_leases": [lease.model_dump(mode="json") for lease in self.resource_lock_manager.snapshot()],
        }

    def get_task_snapshot(self, task_id: str) -> Dict[str, Any]:
        """读取任务状态快照。"""

        return {
            "task_id": task_id,
            "status": self.task_manager.get_task_status(task_id),
            "events": self.task_manager.get_task_events(task_id),
            "result": self.task_manager.get_task_result(task_id),
            "error": self.task_manager.get_task_error(task_id),
        }

    def _handle_get_robot_status(self, _: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        del requested_by
        snapshot = self.robot_state_service.refresh()
        return snapshot.model_dump(mode="json")

    def _handle_get_active_tasks(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        del requested_by
        history_limit = int(arguments.get("history_limit", 20))
        return {
            "active_tasks": self.task_manager.list_active_tasks(),
            "history_tasks": self.task_manager.list_history(limit=history_limit),
        }

    def _handle_list_capabilities(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        include_unsupported = bool(arguments.get("include_unsupported", False))
        exposed_only = None if requested_by == "admin" else True
        skill_views = self.list_tools(exposed_only=exposed_only, include_unsupported=include_unsupported)
        capability_views = self.capability_registry.build_runtime_views(
            robot_model=self.robot.manifest.robot_model,
            exposed_only=exposed_only,
        )
        if not include_unsupported:
            capability_views = tuple(view for view in capability_views if view.supported and view.runnable)
        return {
            "skills": [view.model_dump(mode="json") for view in skill_views],
            "capabilities": [view.model_dump(mode="json") for view in capability_views],
        }

    def _handle_get_task_status(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        del requested_by
        task_id = self._require_str(arguments, "task_id")
        return self.get_task_snapshot(task_id)

    def _handle_cancel_task(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        task_id = self._require_str(arguments, "task_id")
        message = arguments.get("message")
        cancelled = self.task_manager.cancel_task(task_id, f"{requested_by or 'unknown'} 请求取消任务。{message or ''}".strip())
        return {
            "cancelled": cancelled,
            "task": self.get_task_snapshot(task_id),
        }

    def _handle_stop_all_motion(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        reason = arguments.get("reason") or f"{requested_by or 'unknown'} 请求安全停止。"
        self.safety_guard.stop_all_motion(reason)
        return {"stopped": True, "reason": reason}

    def _handle_capture_image(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        camera_id = arguments.get("camera_id")
        context = self.observation_service.capture_observation(
            camera_id=camera_id,
            requested_by=requested_by,
        )
        artifact = context.image_artifact
        if artifact is None:
            raise GatewayError("观察上下文未生成图像制品。")
        self.event_bus.publish(
            self.event_bus.build_event(
                "artifact.created",
                category=RuntimeEventCategory.PERCEPTION,
                source="gateway_runtime",
                subject_id=artifact.artifact_id,
                message="图像制品已写入宿主机。",
                payload={"artifact_id": artifact.artifact_id, "camera_id": context.camera_id},
            )
        )
        return {
            "artifact": artifact,
            "observation_context": context,
            "camera_id": context.camera_id,
            "frame_id": context.observation.frame_id,
            "encoding": context.observation.metadata.get("encoding"),
        }

    def _handle_get_joint_state(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        del requested_by
        state_provider = self._get_provider(StateProvider, "状态提供器")
        joints = list(state_provider.get_joint_states())
        joint_name = str(arguments.get("joint_name", "")).strip()
        if not joint_name:
            return {"joints": joints, "joint": None}
        joint = next((item for item in joints if item.name == joint_name), None)
        return {"joints": joints, "joint": joint}

    def _handle_get_imu_state(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        del arguments, requested_by
        state_provider = self._get_provider(StateProvider, "状态提供器")
        return {"imu_state": state_provider.get_imu_state()}

    def _handle_get_latest_observation(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        del requested_by
        camera_id = arguments.get("camera_id")
        context = self.observation_service.get_latest_observation(camera_id)
        if context is None:
            raise GatewayError("当前还没有可用的观察缓存。")
        return {"observation_context": context}

    def _handle_perceive_current_scene(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        camera_id = arguments.get("camera_id")
        detector_backend = arguments.get("detector_backend")
        context = self.perception_service.perceive_current_scene(
            camera_id=camera_id,
            requested_by=requested_by,
            detector_backend_name=detector_backend,
        )
        return {"perception_context": context}

    def _handle_describe_current_scene(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        camera_id = arguments.get("camera_id")
        detector_backend = arguments.get("detector_backend")
        refresh = bool(arguments.get("refresh", True))
        context = self.perception_service.describe_current_scene(
            camera_id=camera_id,
            refresh=refresh,
            requested_by=requested_by,
            detector_backend_name=detector_backend,
        )
        return {
            "summary": context.scene_summary.headline,
            "scene_summary": context.scene_summary,
            "perception_context": context,
        }

    def _handle_get_latest_perception(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        del requested_by
        camera_id = arguments.get("camera_id")
        context = self.perception_service.get_latest_perception(camera_id)
        if context is None:
            raise GatewayError("当前还没有可用的感知缓存。")
        return {"perception_context": context}

    def _handle_get_localization_snapshot(
        self,
        _: Dict[str, Any],
        *,
        requested_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        del requested_by
        snapshot = self.localization_service.get_latest_snapshot()
        if snapshot is None:
            snapshot = self.localization_service.refresh()
        return {"localization_snapshot": snapshot}

    def _handle_get_map_snapshot(self, _: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        del requested_by
        snapshot = self.mapping_service.get_latest_snapshot()
        if snapshot is None:
            snapshot = self.mapping_service.refresh()
        return {"map_snapshot": snapshot}

    def _handle_get_navigation_snapshot(
        self,
        _: Dict[str, Any],
        *,
        requested_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        del requested_by
        navigation_context = self.navigation_service.get_latest_navigation_context()
        exploration_context = self.navigation_service.get_latest_exploration_context()

        if navigation_context is None and self.navigation_service.is_navigation_available():
            navigation_context = self.navigation_service.refresh_navigation()
        if exploration_context is None and self.navigation_service.is_exploration_available():
            exploration_context = self.navigation_service.refresh_exploration()
        if navigation_context is None and exploration_context is None:
            raise GatewayError("当前机器人入口未提供导航或探索后端。")
        return {
            "navigation_context": navigation_context,
            "exploration_context": exploration_context,
        }

    def _handle_tag_location(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        name = self._require_str(arguments, "name")
        aliases = arguments.get("aliases") or []
        if not isinstance(aliases, list):
            raise GatewayError(
                "aliases 必须为数组。",
                error_code="invalid_params",
                http_status=422,
                jsonrpc_code=-32602,
            )
        tagged_location, semantic_memory = self.memory_service.tag_location(
            name,
            aliases=aliases,
            description=str(arguments.get("description", "")).strip() or None,
            camera_id=str(arguments.get("camera_id", "")).strip() or None,
            auto_create_memory=bool(arguments.get("auto_create_memory", True)),
            metadata={"requested_by": requested_by or "unknown"},
        )
        return {
            "tagged_location": tagged_location,
            "semantic_memory": semantic_memory,
        }

    def _handle_query_location(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        del requested_by
        query = self._require_str(arguments, "query")
        payload_filter = self._build_memory_payload_filter(arguments)
        return {
            "query_result": self.memory_service.query_location(
                query,
                similarity_threshold=self._float(arguments, "similarity_threshold", 0.55),
                limit=self._int(arguments, "limit", 5),
                payload_filter=payload_filter,
                max_age_sec=payload_filter.max_age_sec if payload_filter is not None else None,
                near_pose=payload_filter.near_pose if payload_filter is not None else None,
                max_distance_m=payload_filter.max_distance_m if payload_filter is not None else None,
            )
        }

    def _handle_query_semantic_memory(
        self,
        arguments: Dict[str, Any],
        *,
        requested_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        del requested_by
        query = self._require_str(arguments, "query")
        payload_filter = self._build_memory_payload_filter(arguments)
        return {
            "query_result": self.memory_service.query_semantic_memory(
                query,
                similarity_threshold=self._float(arguments, "similarity_threshold", 0.25),
                limit=self._int(arguments, "limit", 5),
                payload_filter=payload_filter,
                max_age_sec=payload_filter.max_age_sec if payload_filter is not None else None,
                near_pose=payload_filter.near_pose if payload_filter is not None else None,
                max_distance_m=payload_filter.max_distance_m if payload_filter is not None else None,
            )
        }

    def _handle_remember_current_scene(
        self,
        arguments: Dict[str, Any],
        *,
        requested_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        entry = self.memory_service.remember_current_scene(
            title=str(arguments.get("title", "")).strip() or None,
            camera_id=str(arguments.get("camera_id", "")).strip() or None,
            summary_override=str(arguments.get("summary", "")).strip() or None,
            metadata={"requested_by": requested_by or "unknown"},
        )
        return {"semantic_memory": entry}

    def _handle_navigate_to_pose(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None):
        pose = self._build_pose_from_arguments(arguments)
        goal = NavigationGoal(
            goal_id=self._build_runtime_request_id("nav_pose"),
            target_pose=pose,
            tolerance_position_m=self._float(arguments, "tolerance_position_m", 0.3),
            tolerance_yaw_rad=self._float(arguments, "tolerance_yaw_rad", 0.3),
            metadata={"requested_by": requested_by or "unknown"},
        )
        poll_interval_sec = self._float(arguments, "poll_interval_sec", 0.1)
        timeout_sec = self._resolve_async_timeout("navigate_to_pose", arguments, minimum_sec=0.1)
        return self._submit_navigation_task(
            capability_name="navigate_to_pose",
            goal=goal,
            poll_interval_sec=poll_interval_sec,
            timeout_sec=timeout_sec,
            requested_by=requested_by,
        )

    def _handle_navigate_to_named_location(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None):
        target_name = self._require_str(arguments, "target_name")
        camera_id = str(arguments.get("camera_id", "")).strip() or None
        verify_on_arrival = bool(arguments.get("verify_on_arrival", True))
        verify_similarity_threshold = self._float(arguments, "verify_similarity_threshold", 0.55)
        navigation_candidate = None
        tagged_location = self.memory_service.resolve_tagged_location(target_name)
        if tagged_location is not None:
            navigation_candidate = self.memory_service.resolve_navigation_candidate(
                target_name,
                camera_id=camera_id,
            )
            goal = NavigationGoal(
                goal_id=self._build_runtime_request_id("nav_memory"),
                target_pose=tagged_location.pose,
                target_name=tagged_location.name,
                metadata={
                    "resolved_from": "memory",
                    "location_id": tagged_location.location_id,
                    "requested_by": requested_by or "unknown",
                },
            )
        else:
            navigation_candidate = self.memory_service.resolve_navigation_candidate(
                target_name,
                camera_id=camera_id,
            )
            if navigation_candidate is not None:
                goal = NavigationGoal(
                    goal_id=self._build_runtime_request_id("nav_memory_query"),
                    target_pose=navigation_candidate.target_pose,
                    target_name=navigation_candidate.target_name,
                    metadata={
                        "resolved_from": "semantic_memory",
                        "record_id": navigation_candidate.record_id,
                        "record_kind": navigation_candidate.record_kind.value,
                        "requested_by": requested_by or "unknown",
                    },
                )
            else:
                goal = self.navigation_service.resolve_named_goal(target_name)
        goal = goal.model_copy(
            update={
                "goal_id": self._build_runtime_request_id("nav_named"),
                "metadata": {
                    **dict(goal.metadata),
                    "requested_by": requested_by or "unknown",
                },
            },
            deep=True,
        )
        poll_interval_sec = self._float(arguments, "poll_interval_sec", 0.1)
        timeout_sec = self._resolve_async_timeout("navigate_to_named_location", arguments, minimum_sec=0.1)
        return self._submit_navigation_task(
            capability_name="navigate_to_named_location",
            goal=goal,
            poll_interval_sec=poll_interval_sec,
            timeout_sec=timeout_sec,
            requested_by=requested_by,
            completion_callback=(
                (lambda _navigation_context: {
                    "arrival_verification": self.memory_service.verify_arrival(
                        target_name,
                        navigation_candidate=navigation_candidate,
                        camera_id=camera_id,
                        similarity_threshold=verify_similarity_threshold,
                    )
                })
                if verify_on_arrival
                else None
            ),
        )

    def _handle_explore_area(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None):
        target_name = str(arguments.get("target_name", "")).strip() or None
        center_pose = None
        if "frame_id" in arguments or "x" in arguments or "y" in arguments:
            center_pose = self._build_pose_from_arguments(arguments)
        if target_name is None and center_pose is None:
            raise GatewayError(
                "explore_area 至少需要提供 target_name 或 frame_id/x/y。",
                error_code="invalid_params",
                http_status=422,
                jsonrpc_code=-32602,
            )
        request = ExploreAreaRequest(
            request_id=self._build_runtime_request_id("explore"),
            center_pose=center_pose,
            target_name=target_name,
            radius_m=self._float(arguments, "radius_m", 0.0) or None,
            strategy=str(arguments.get("strategy") or "frontier"),
            max_duration_sec=self._float(arguments, "max_duration_sec", 0.0) or None,
            metadata={"requested_by": requested_by or "unknown"},
        )
        poll_interval_sec = self._float(arguments, "poll_interval_sec", 0.2)
        timeout_sec = self._resolve_async_timeout("explore_area", arguments, minimum_sec=0.1)
        return self._submit_exploration_task(
            request=request,
            poll_interval_sec=poll_interval_sec,
            timeout_sec=timeout_sec,
            requested_by=requested_by,
        )

    def _handle_inspect_target(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None):
        query = self._require_str(arguments, "query")
        camera_id = str(arguments.get("camera_id", "")).strip() or None
        detector_backend = str(arguments.get("detector_backend", "")).strip() or None
        auto_navigate = bool(arguments.get("auto_navigate", False))
        poll_interval_sec = self._float(arguments, "poll_interval_sec", 0.1)
        timeout_sec = self._resolve_async_timeout("inspect_target", arguments, minimum_sec=0.1)
        descriptor = self.capability_registry.get_descriptor("inspect_target")

        def runner(context) -> Dict[str, Any]:
            context.update(progress=0.05, stage="inspect_observe", message="开始检查目标。")
            perception_context = self.perception_service.describe_current_scene(
                camera_id=camera_id,
                refresh=True,
                requested_by=requested_by,
                detector_backend_name=detector_backend,
            )
            matched_object = self._find_scene_object(query, perception_context)
            if matched_object is None and auto_navigate:
                context.update(progress=0.3, stage="inspect_navigate", message="当前画面未找到目标，尝试导航到命名位置。")
                tagged_location = self.memory_service.resolve_tagged_location(query)
                if tagged_location is not None:
                    goal = NavigationGoal(
                        goal_id=self._build_runtime_request_id("inspect_nav"),
                        target_pose=tagged_location.pose,
                        target_name=tagged_location.name,
                        metadata={"resolved_from": "memory", "location_id": tagged_location.location_id},
                    )
                else:
                    goal = self.navigation_service.resolve_named_goal(query)
                self.navigation_service.navigate_until_complete(
                    goal,
                    poll_interval_sec=max(0.02, poll_interval_sec),
                    on_progress=lambda nav_context: context.update(
                        progress=0.3 + 0.5 * self._estimate_navigation_progress(nav_context),
                        stage=f"inspect_navigation_{nav_context.navigation_state.status.value}",
                        message=nav_context.navigation_state.message or "导航检查目标中。",
                    ),
                )
                context.ensure_active()
                perception_context = self.perception_service.describe_current_scene(
                    camera_id=camera_id,
                    refresh=True,
                    requested_by=requested_by,
                    detector_backend_name=detector_backend,
                )
                matched_object = self._find_scene_object(query, perception_context)

            if matched_object is None:
                raise GatewayError(f"未在当前场景中找到目标：{query}")

            memory_entry = self.memory_service.remember_current_scene(
                title=f"检查结果：{query}",
                camera_id=camera_id,
                summary_override=perception_context.scene_summary.headline,
                tags=[query, matched_object.label],
                metadata={"requested_by": requested_by or "unknown", "source": "inspect_target"},
            )
            context.update(progress=1.0, stage="completed", message="目标检查完成。")
            return {
                "query": query,
                "matched_object": matched_object,
                "perception_context": perception_context,
                "semantic_memory": memory_entry,
            }

        return self.task_manager.submit(
            "inspect_target",
            runner,
            input_payload={
                "query": query,
                "camera_id": camera_id,
                "auto_navigate": auto_navigate,
                "detector_backend": detector_backend,
            },
            requested_by=requested_by,
            required_resources=list(descriptor.required_resources),
            timeout_sec=timeout_sec,
            metadata={"requested_by": requested_by or "unknown"},
        )

    def _handle_follow_target(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None):
        query = self._require_str(arguments, "query")
        camera_id = str(arguments.get("camera_id", "")).strip() or None
        detector_backend = str(arguments.get("detector_backend", "")).strip() or None
        duration_sec = self._float(arguments, "duration_sec", 2.0)
        interval_sec = self._float(arguments, "interval_sec", 0.1)
        lost_timeout_sec = self._float(arguments, "lost_timeout_sec", 0.8)
        timeout_sec = self._resolve_async_timeout("follow_target", arguments, minimum_sec=0.1)
        descriptor = self.capability_registry.get_descriptor("follow_target")
        motion_control = self._get_provider(MotionControl, "运动控制提供器")
        if duration_sec <= 0 or interval_sec <= 0 or lost_timeout_sec <= 0:
            raise GatewayError(
                "duration_sec、interval_sec、lost_timeout_sec 必须大于 0。",
                error_code="invalid_params",
                http_status=422,
                jsonrpc_code=-32602,
            )

        def runner(context) -> Dict[str, Any]:
            start_time = time.monotonic()
            last_seen_time = 0.0
            last_detection = None
            try:
                while True:
                    context.ensure_active()
                    elapsed = time.monotonic() - start_time
                    if elapsed >= duration_sec:
                        break
                    perception_context = self.perception_service.perceive_current_scene(
                        camera_id=camera_id,
                        requested_by=requested_by,
                        detector_backend_name=detector_backend,
                    )
                    detection = self._find_best_detection(query, perception_context)
                    if detection is None:
                        if last_seen_time and time.monotonic() - last_seen_time > lost_timeout_sec:
                            raise GatewayError(f"目标 {query} 已丢失。")
                        motion_control.stop_motion()
                        context.update(
                            progress=min(0.95, elapsed / duration_sec),
                            stage="follow_searching",
                            message=f"正在搜索目标 {query}。",
                        )
                        time.sleep(interval_sec)
                        continue

                    last_seen_time = time.monotonic()
                    last_detection = detection
                    follow_command = self._build_follow_twist(perception_context, detection)
                    motion_control.send_twist(follow_command)
                    context.update(
                        progress=min(0.99, elapsed / duration_sec),
                        stage="follow_tracking",
                        message=f"正在跟随目标 {query}。",
                        payload={
                            "label": detection.label,
                            "score": detection.score,
                        },
                    )
                    time.sleep(interval_sec)
            finally:
                motion_control.stop_motion()

            context.update(progress=1.0, stage="completed", message="目标跟随完成。")
            return {
                "query": query,
                "duration_sec": duration_sec,
                "last_detection": last_detection,
            }

        return self.task_manager.submit(
            "follow_target",
            runner,
            input_payload={
                "query": query,
                "camera_id": camera_id,
                "duration_sec": duration_sec,
                "interval_sec": interval_sec,
                "lost_timeout_sec": lost_timeout_sec,
            },
            requested_by=requested_by,
            required_resources=list(descriptor.required_resources),
            timeout_sec=timeout_sec,
            metadata={"requested_by": requested_by or "unknown"},
        )

    def _handle_relative_move(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None):
        motion_control = self._get_provider(MotionControl, "运动控制提供器")
        descriptor = self.capability_registry.get_descriptor("relative_move")
        vx = self._require_float(arguments, "vx")
        vy = self._float(arguments, "vy", 0.0)
        vyaw = self._float(arguments, "vyaw", 0.0)
        duration_sec = self._require_float(arguments, "duration_sec")
        interval_sec = self._float(arguments, "interval_sec", 0.05)
        timeout_override = self._float(arguments, "timeout_sec", 0.0)

        if duration_sec <= 0 or interval_sec <= 0:
            raise GatewayError(
                "duration_sec 和 interval_sec 必须大于 0。",
                error_code="invalid_params",
                http_status=422,
                jsonrpc_code=-32602,
            )

        timeout_sec = max(timeout_override, duration_sec + 2.0, float(descriptor.timeout_sec or 0))

        def runner(context) -> Dict[str, float]:
            start_time = time.monotonic()
            try:
                while True:
                    context.ensure_active()
                    elapsed = time.monotonic() - start_time
                    if elapsed >= duration_sec:
                        break
                    motion_control.send_twist(
                        Twist(
                            frame_id=self.robot.defaults.frame_ids.get("base", "world/base"),
                            linear=Vector3(x=vx, y=vy, z=0.0),
                            angular=Vector3(x=0.0, y=0.0, z=vyaw),
                        )
                    )
                    progress = min(0.99, elapsed / duration_sec)
                    context.update(
                        progress=progress,
                        stage="moving",
                        message="机器人正在执行相对移动。",
                        payload={"elapsed_sec": round(elapsed, 3)},
                    )
                    time.sleep(interval_sec)
            finally:
                motion_control.stop_motion()

            context.update(progress=1.0, stage="completed", message="机器人已完成相对移动。")
            return {
                "vx": vx,
                "vy": vy,
                "vyaw": vyaw,
                "duration_sec": duration_sec,
            }

        return self.task_manager.submit(
            "relative_move",
            runner,
            input_payload={
                "vx": vx,
                "vy": vy,
                "vyaw": vyaw,
                "duration_sec": duration_sec,
                "interval_sec": interval_sec,
            },
            requested_by=requested_by,
            required_resources=[RuntimeResource.BASE_MOTION.value],
            timeout_sec=timeout_sec,
            metadata={"requested_by": requested_by or "unknown"},
        )

    def _handle_execute_sport_command(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        del requested_by
        if not hasattr(self.robot, "execute_action"):
            raise GatewayError("当前机器人入口未实现 execute_action。")
        action = self._require_str(arguments, "action")
        params = arguments.get("params") or {}
        if not isinstance(params, dict):
            raise GatewayError(
                "params 必须为对象。",
                error_code="invalid_params",
                http_status=422,
                jsonrpc_code=-32602,
            )
        code, result = self.robot.execute_action(action, params)
        if code is None:
            raise GatewayError(str(result or f"动作 {action} 执行失败。"))
        return {"code": code, "result": result or {}, "action": action}

    def _handle_set_body_pose(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        del requested_by
        if not hasattr(self.robot, "execute_action"):
            raise GatewayError("当前机器人入口未实现 execute_action。")
        code, result = self.robot.execute_action(
            "euler",
            {
                "roll": self._float(arguments, "roll", 0.0),
                "pitch": self._float(arguments, "pitch", 0.0),
                "yaw": self._float(arguments, "yaw", 0.0),
            },
        )
        if code is None:
            raise GatewayError(str(result or "机身姿态设置失败。"))
        return {"code": code, "result": result or {}}

    def _handle_set_speed_level(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        del requested_by
        if not hasattr(self.robot, "execute_action"):
            raise GatewayError("当前机器人入口未实现 execute_action。")
        level = self._int(arguments, "level", 1)
        code, result = self.robot.execute_action("speed_level", {"level": level})
        if code is None:
            raise GatewayError(str(result or "速度档位设置失败。"))
        return {"code": code, "result": result or {}, "level": level}

    def _handle_get_volume(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        del arguments, requested_by
        return {"volume_state": self.audio_service.get_volume_state()}

    def _handle_set_volume(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        del requested_by
        volume = self._require_float(arguments, "volume")
        return {"volume_state": self.audio_service.set_volume(volume)}

    def _handle_speak_text(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        text = self._require_str(arguments, "text")
        return {
            "speech": self.audio_service.speak_text(
                text,
                metadata={"requested_by": requested_by or "unknown"},
            )
        }

    def _handle_switch_control_mode(self, arguments: Dict[str, Any], *, requested_by: Optional[str] = None) -> Dict[str, Any]:
        del requested_by
        if not hasattr(self.robot, "switch_mode"):
            raise GatewayError("当前机器人入口未实现 switch_mode。")
        mode = self._require_str(arguments, "mode")
        if mode not in {"high", "low"}:
            raise GatewayError(
                "mode 仅支持 high 或 low。",
                error_code="invalid_params",
                http_status=422,
                jsonrpc_code=-32602,
            )
        code, result = self.robot.switch_mode(mode)
        return {"code": code, "result": result or {}, "mode": mode}

    def _safe_stop_robot(self, reason: Optional[str]) -> None:
        providers = getattr(self.robot, "providers", None)
        if isinstance(providers, SafetyProvider):
            providers.request_safe_stop(reason)
            return
        if isinstance(providers, MotionControl):
            providers.stop_motion()

    def _get_provider(self, provider_type: type, provider_label: str):
        providers = getattr(self.robot, "providers", None)
        if providers is None or not isinstance(providers, provider_type):
            raise GatewayError(f"当前机器人入口未提供{provider_label}。")
        if not providers.is_available():
            raise GatewayError(f"当前{provider_label}暂不可用。")
        return providers

    def _submit_navigation_task(
        self,
        *,
        capability_name: str,
        goal: NavigationGoal,
        poll_interval_sec: float,
        timeout_sec: float,
        requested_by: Optional[str],
        completion_callback=None,
    ):
        descriptor = self.capability_registry.get_descriptor(capability_name)
        poll_interval_sec = max(0.02, poll_interval_sec)

        def runner(context) -> Dict[str, Any]:
            last_context = self.navigation_service.set_goal(goal)
            context.update(
                progress=0.05,
                stage="accepted",
                message=f"导航目标 {goal.goal_id} 已提交。",
                payload={"goal_id": goal.goal_id},
            )
            try:
                while True:
                    context.ensure_active()
                    last_context = self.navigation_service.refresh_navigation()
                    status = last_context.navigation_state.status
                    context.update(
                        progress=self._estimate_navigation_progress(last_context),
                        stage=f"navigation_{status.value}",
                        message=last_context.navigation_state.message or "导航执行中。",
                        payload={
                            "goal_id": goal.goal_id,
                            "remaining_distance_m": last_context.navigation_state.remaining_distance_m,
                            "goal_reached": last_context.goal_reached,
                        },
                    )
                    if last_context.goal_reached or status == NavigationStatus.SUCCEEDED:
                        context.update(progress=1.0, stage="completed", message="导航任务已完成。")
                        result = {
                            "goal_id": goal.goal_id,
                            "navigation_context": last_context,
                        }
                        if completion_callback is not None:
                            result.update(dict(completion_callback(last_context) or {}))
                        return result
                    if status == NavigationStatus.FAILED:
                        raise GatewayError(last_context.navigation_state.message or "导航执行失败。")
                    if status == NavigationStatus.CANCELLED:
                        raise GatewayError(last_context.navigation_state.message or "导航已取消。")
                    time.sleep(poll_interval_sec)
            finally:
                if context.is_cancel_requested() or context.is_timed_out():
                    try:
                        self.navigation_service.cancel_goal()
                    except Exception:
                        pass

        return self.task_manager.submit(
            capability_name,
            runner,
            input_payload={
                "goal_id": goal.goal_id,
                "target_name": goal.target_name,
                "target_pose": goal.target_pose.model_dump(mode="json") if goal.target_pose is not None else None,
                "poll_interval_sec": poll_interval_sec,
            },
            requested_by=requested_by,
            required_resources=list(descriptor.required_resources),
            timeout_sec=timeout_sec,
            metadata={"requested_by": requested_by or "unknown"},
        )

    def _submit_exploration_task(
        self,
        *,
        request: ExploreAreaRequest,
        poll_interval_sec: float,
        timeout_sec: float,
        requested_by: Optional[str],
    ):
        descriptor = self.capability_registry.get_descriptor("explore_area")
        poll_interval_sec = max(0.05, poll_interval_sec)

        def runner(context) -> Dict[str, Any]:
            last_context = self.navigation_service.start_exploration(request)
            context.update(
                progress=0.05,
                stage="accepted",
                message=f"探索任务 {request.request_id} 已提交。",
                payload={"request_id": request.request_id},
            )
            try:
                while True:
                    context.ensure_active()
                    last_context = self.navigation_service.refresh_exploration()
                    status = last_context.exploration_state.status
                    context.update(
                        progress=self._estimate_exploration_progress(last_context),
                        stage=f"exploration_{status.value}",
                        message=last_context.exploration_state.message or "探索执行中。",
                        payload={
                            "request_id": request.request_id,
                            "covered_ratio": last_context.exploration_state.covered_ratio,
                            "frontier_count": last_context.exploration_state.frontier_count,
                        },
                    )
                    if status.value == "succeeded":
                        context.update(progress=1.0, stage="completed", message="探索任务已完成。")
                        return {
                            "request_id": request.request_id,
                            "exploration_context": last_context,
                        }
                    if status.value == "failed":
                        raise GatewayError(last_context.exploration_state.message or "探索执行失败。")
                    if status.value == "cancelled":
                        raise GatewayError(last_context.exploration_state.message or "探索已取消。")
                    time.sleep(poll_interval_sec)
            finally:
                if context.is_cancel_requested() or context.is_timed_out():
                    try:
                        self.navigation_service.stop_exploration()
                    except Exception:
                        pass

        return self.task_manager.submit(
            "explore_area",
            runner,
            input_payload={
                "request_id": request.request_id,
                "target_name": request.target_name,
                "center_pose": request.center_pose.model_dump(mode="json") if request.center_pose is not None else None,
                "radius_m": request.radius_m,
                "strategy": request.strategy,
                "max_duration_sec": request.max_duration_sec,
                "poll_interval_sec": poll_interval_sec,
            },
            requested_by=requested_by,
            required_resources=list(descriptor.required_resources),
            timeout_sec=timeout_sec,
            metadata={"requested_by": requested_by or "unknown"},
        )

    def _find_scene_object(self, query: str, perception_context):
        normalized_query = self._normalize_query(query)
        for item in perception_context.scene_summary.objects:
            if normalized_query == self._normalize_query(item.label):
                return item
        for item in perception_context.scene_summary.objects:
            candidates = {self._normalize_query(item.label)}
            candidates.update(self._normalize_query(camera_id) for camera_id in item.camera_ids)
            if normalized_query in candidates:
                return item
        if normalized_query and normalized_query in self._normalize_query(perception_context.scene_summary.headline):
            return perception_context.scene_summary.objects[0] if perception_context.scene_summary.objects else None
        return None

    def _find_best_detection(self, query: str, perception_context):
        normalized_query = self._normalize_query(query)
        detections = [
            detection
            for detection in perception_context.observation.detections_2d
            if normalized_query == self._normalize_query(detection.label)
        ]
        if not detections:
            return None
        detections.sort(key=lambda item: item.score, reverse=True)
        return detections[0]

    def _build_follow_twist(self, perception_context, detection) -> Twist:
        camera_info = perception_context.camera_info
        scene_metadata = dict(perception_context.scene_summary.metadata)
        width = float(
            camera_info.width_px
            if camera_info is not None
            else scene_metadata.get("image_width_px") or detection.bbox.width_px or 1.0
        )
        height = float(
            camera_info.height_px
            if camera_info is not None
            else scene_metadata.get("image_height_px") or detection.bbox.height_px or 1.0
        )
        bbox = detection.bbox
        center_x = bbox.x_px + bbox.width_px / 2.0
        normalized_error_x = ((center_x - width / 2.0) / max(width / 2.0, 1.0))
        bbox_ratio = bbox.height_px / max(height, 1.0)
        desired_ratio = 0.45
        distance_error = desired_ratio - bbox_ratio

        linear_x = max(-0.18, min(0.22, distance_error * 0.6))
        angular_z = max(-0.5, min(0.5, -normalized_error_x * 0.8))
        if abs(distance_error) < 0.05:
            linear_x = 0.0
        if abs(normalized_error_x) < 0.05:
            angular_z = 0.0
        return Twist(
            frame_id=self.robot.defaults.frame_ids.get("base", "world/base"),
            linear=Vector3(x=linear_x, y=0.0, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=angular_z),
        )

    def _build_gateway_capability_availability(
        self,
        capability_name: str,
        *,
        fallback: Optional[CapabilityAvailability] = None,
    ) -> CapabilityAvailability:
        metadata = dict(fallback.metadata) if fallback is not None else {}
        reason = fallback.reason if fallback is not None else None
        supported = fallback.supported if fallback is not None else True

        dynamic_support: Dict[str, Tuple[bool, Optional[str]]] = {
            "capture_image": (
                self._has_provider(ImageProvider),
                "当前机器人入口未提供图像采集提供器。",
            ),
            "get_latest_observation": (
                self._has_provider(ImageProvider),
                "当前机器人入口未提供图像采集提供器。",
            ),
            "perceive_current_scene": (
                self._has_provider(ImageProvider),
                "当前机器人入口未提供图像采集提供器。",
            ),
            "describe_current_scene": (
                self._has_provider(ImageProvider),
                "当前机器人入口未提供图像采集提供器。",
            ),
            "get_latest_perception": (
                self._has_provider(ImageProvider),
                "当前机器人入口未提供图像采集提供器。",
            ),
            "get_localization_snapshot": (
                self.localization_service.is_available(),
                "当前机器人入口未提供定位提供器，且还没有外部定位快照。",
            ),
            "get_map_snapshot": (
                self.mapping_service.is_available(),
                "当前机器人入口未提供地图提供器，且还没有外部地图快照。",
            ),
            "get_navigation_snapshot": (
                self.navigation_service.is_navigation_available() or self.navigation_service.is_exploration_available(),
                "当前机器人入口未提供导航或探索后端。",
            ),
            "tag_location": (
                self.localization_service.is_available(),
                "当前机器人入口未提供定位结果，无法标记地点。",
            ),
            "query_location": (
                True,
                None,
            ),
            "query_semantic_memory": (
                True,
                None,
            ),
            "remember_current_scene": (
                self._has_provider(ImageProvider),
                "当前机器人入口未提供图像采集提供器。",
            ),
            "navigate_to_pose": (
                self.navigation_service.is_navigation_available(),
                "当前机器人入口未提供导航后端。",
            ),
            "navigate_to_named_location": (
                self.navigation_service.is_navigation_available()
                and (self.mapping_service.is_available() or bool(self.memory_service.list_tagged_locations(limit=1))),
                "当前机器人入口未提供导航后端，且也没有可用地点记忆或语义地图。",
            ),
            "explore_area": (
                self.navigation_service.is_exploration_available(),
                "当前机器人入口未提供探索后端。",
            ),
            "inspect_target": (
                self._has_provider(ImageProvider),
                "当前机器人入口未提供图像采集提供器。",
            ),
            "follow_target": (
                self._has_provider(ImageProvider) and self._has_provider(MotionControl),
                "当前机器人入口未同时提供图像采集与运动控制提供器。",
            ),
            "relative_move": (
                self._has_provider(MotionControl),
                "当前机器人入口未提供运动控制提供器。",
            ),
            "execute_sport_command": (
                hasattr(self.robot, "execute_action"),
                "当前机器人入口未实现 execute_action。",
            ),
            "set_body_pose": (
                hasattr(self.robot, "execute_action"),
                "当前机器人入口未实现 execute_action。",
            ),
            "set_speed_level": (
                hasattr(self.robot, "execute_action"),
                "当前机器人入口未实现 execute_action。",
            ),
            "get_joint_state": (
                self._has_provider(StateProvider),
                "当前机器人入口未提供状态提供器。",
            ),
            "get_imu_state": (
                self._has_provider(StateProvider),
                "当前机器人入口未提供状态提供器。",
            ),
            "get_volume": (
                self.audio_service.is_volume_available(),
                "当前机器人入口未提供机器人音量控制，仅保留软件音量。",
            ),
            "set_volume": (
                self.audio_service.is_speech_available(),
                "当前播报与音量服务不可用。",
            ),
            "speak_text": (
                self.audio_service.is_speech_available(),
                "当前播报服务不可用。",
            ),
            "stop_all_motion": (
                self._has_provider(SafetyProvider) or self._has_provider(MotionControl),
                "当前机器人入口未提供安全停止或运动控制提供器。",
            ),
            "switch_control_mode": (
                hasattr(self.robot, "switch_mode"),
                "当前机器人入口未实现 switch_mode。",
            ),
            "get_robot_status": (
                self._has_provider(StateProvider),
                "当前机器人入口未提供状态提供器。",
            ),
        }

        if capability_name in dynamic_support:
            supported, reason = dynamic_support[capability_name]
        metadata["runtime_gateway"] = "enabled"
        metadata["dynamic_availability"] = capability_name in dynamic_support
        return CapabilityAvailability(
            name=capability_name,
            supported=supported,
            reason=None if supported else reason,
            metadata=metadata,
        )

    def _has_provider(self, provider_type: type) -> bool:
        providers = getattr(self.robot, "providers", None)
        return providers is not None and isinstance(providers, provider_type)

    def _build_pose_from_arguments(self, arguments: Dict[str, Any]) -> Pose:
        frame_id = self._require_str(arguments, "frame_id")
        x = self._require_float(arguments, "x")
        y = self._require_float(arguments, "y")
        z = self._float(arguments, "z", 0.0)
        yaw_rad = self._float(arguments, "yaw_rad", 0.0)
        half_yaw = yaw_rad / 2.0
        return Pose(
            frame_id=frame_id,
            position=Vector3(x=x, y=y, z=z),
            orientation=Quaternion(z=math.sin(half_yaw), w=math.cos(half_yaw)),
        )

    def _estimate_navigation_progress(self, context) -> float:
        remaining_distance = context.navigation_state.remaining_distance_m
        if context.goal_reached:
            return 1.0
        if remaining_distance is None:
            return 0.35
        estimated_total = max(remaining_distance + 0.5, 0.5)
        return max(0.05, min(0.95, 1.0 - (remaining_distance / estimated_total)))

    def _estimate_exploration_progress(self, context) -> float:
        covered_ratio = context.exploration_state.covered_ratio
        if covered_ratio is not None:
            return max(0.05, min(0.99, covered_ratio))
        frontier_count = context.exploration_state.frontier_count
        if frontier_count is None:
            return 0.2
        if frontier_count <= 0:
            return 0.99
        return max(0.1, min(0.9, 1.0 / float(frontier_count + 1)))

    def _build_runtime_request_id(self, prefix: str) -> str:
        return f"{prefix}_{time.time_ns()}"

    def _resolve_async_timeout(self, capability_name: str, payload: Dict[str, Any], *, minimum_sec: float) -> float:
        descriptor = self.capability_registry.get_descriptor(capability_name)
        timeout_override = self._float(payload, "timeout_sec", 0.0)
        default_timeout = float(descriptor.timeout_sec or 0)
        selected_timeout = timeout_override if timeout_override > 0 else default_timeout
        return max(minimum_sec, selected_timeout)

    def _normalize_query(self, value: Optional[str]) -> str:
        return "".join(char.lower() if char.isalnum() or "\u4e00" <= char <= "\u9fff" else " " for char in str(value or "")).strip()

    def _serialize(self, value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        return to_jsonable(value)

    def _require_str(self, payload: Dict[str, Any], key: str) -> str:
        value = str(payload.get(key, "")).strip()
        if not value:
            raise GatewayError(
                f"参数 {key} 不能为空。",
                error_code="invalid_params",
                http_status=422,
                jsonrpc_code=-32602,
            )
        return value

    def _require_float(self, payload: Dict[str, Any], key: str) -> float:
        if key not in payload:
            raise GatewayError(
                f"缺少参数 {key}。",
                error_code="invalid_params",
                http_status=422,
                jsonrpc_code=-32602,
            )
        return self._float(payload, key, 0.0)

    def _float(self, payload: Dict[str, Any], key: str, default: float) -> float:
        if key not in payload:
            return default
        try:
            return float(payload[key])
        except (TypeError, ValueError) as exc:
            raise GatewayError(
                f"参数 {key} 必须为数字。",
                error_code="invalid_params",
                http_status=422,
                jsonrpc_code=-32602,
            ) from exc

    def _int(self, payload: Dict[str, Any], key: str, default: int) -> int:
        if key not in payload:
            return default
        try:
            return int(payload[key])
        except (TypeError, ValueError) as exc:
            raise GatewayError(
                f"参数 {key} 必须为整数。",
                error_code="invalid_params",
                http_status=422,
                jsonrpc_code=-32602,
            ) from exc

    def _build_memory_payload_filter(self, payload: Dict[str, Any]) -> Optional[MemoryPayloadFilter]:
        semantic_labels = payload.get("semantic_labels")
        if isinstance(semantic_labels, list):
            normalized_semantic_labels = [str(item).strip() for item in semantic_labels if str(item).strip()]
        else:
            normalized_semantic_labels = []
        visual_labels = payload.get("visual_labels")
        if isinstance(visual_labels, list):
            normalized_visual_labels = [str(item).strip() for item in visual_labels if str(item).strip()]
        else:
            normalized_visual_labels = []
        vision_tags = payload.get("vision_tags")
        if isinstance(vision_tags, list):
            normalized_vision_tags = [str(item).strip() for item in vision_tags if str(item).strip()]
        else:
            normalized_vision_tags = []
        topo_node_id = str(payload.get("topo_node_id", "")).strip()
        near_pose = None
        if "near_frame_id" in payload or "near_x" in payload or "near_y" in payload:
            near_pose = Pose(
                frame_id=str(payload.get("near_frame_id") or "map"),
                position=Vector3(
                    x=self._float(payload, "near_x", 0.0),
                    y=self._float(payload, "near_y", 0.0),
                    z=self._float(payload, "near_z", 0.0),
                ),
                orientation=Quaternion(w=1.0),
            )
        filter_payload = MemoryPayloadFilter(
            map_version_id=str(payload.get("map_version_id", "")).strip() or None,
            linked_location_id=str(payload.get("linked_location_id", "")).strip() or None,
            camera_ids=[str(payload.get("camera_id")).strip()] if str(payload.get("camera_id", "")).strip() else [],
            semantic_labels_any=normalized_semantic_labels,
            visual_labels_any=normalized_visual_labels,
            vision_tags_any=normalized_vision_tags,
            topo_node_ids=[topo_node_id] if topo_node_id else [],
            max_age_sec=self._float(payload, "max_age_sec", 0.0) or None,
            near_pose=near_pose,
            max_distance_m=self._float(payload, "max_distance_m", 0.0) or None,
        )
        has_filter = any(
            [
                filter_payload.map_version_id,
                filter_payload.linked_location_id,
                filter_payload.camera_ids,
                filter_payload.semantic_labels_any,
                filter_payload.visual_labels_any,
                filter_payload.vision_tags_any,
                filter_payload.topo_node_ids,
                filter_payload.max_age_sec is not None,
                filter_payload.near_pose is not None,
                filter_payload.max_distance_m is not None,
            ]
        )
        if not has_filter:
            return None
        return filter_payload


def create_default_gateway_runtime(
    config: NuwaxRobotBridgeConfig,
    robot: RobotAssemblyBase,
) -> GatewayRuntime:
    """创建默认宿主机运行时。"""

    return GatewayRuntime(config=config, robot=robot)
