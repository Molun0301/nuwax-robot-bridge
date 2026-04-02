from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable

from pydantic import Field

from contracts.base import ContractModel, MetadataDict
from contracts.capabilities import (
    CapabilityAvailability,
    CapabilityDescriptor,
    CapabilityExecutionMode,
    CapabilityMatrix,
)
from contracts.naming import validate_capability_name

CapabilityHandler = Callable[..., Any]


class CapabilityRuntimeView(ContractModel):
    """运行时能力视图。"""

    name: str = Field(description="能力名称。")
    owner: str = Field(description="能力归属模块。")
    descriptor: CapabilityDescriptor = Field(description="能力描述。")
    execution_mode: CapabilityExecutionMode = Field(description="执行模式。")
    supported: bool = Field(description="当前是否支持。")
    reason: Optional[str] = Field(default=None, description="不支持原因。")
    runnable: bool = Field(description="当前是否存在处理器。")
    robot_model: Optional[str] = Field(default=None, description="关联机器人型号。")
    metadata: MetadataDict = Field(default_factory=dict, description="附加元数据。")


@dataclass
class RegisteredCapability:
    """单条能力注册记录。"""

    descriptor: CapabilityDescriptor
    handler: Optional[CapabilityHandler] = None
    owner: str = "runtime"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def runnable(self) -> bool:
        return self.handler is not None


class CapabilityNotRegisteredError(KeyError):
    """能力未注册异常。"""


class CapabilityRegistry:
    """能力注册表。"""

    def __init__(self) -> None:
        self._registrations: Dict[str, RegisteredCapability] = {}
        self._robot_matrices: Dict[str, CapabilityMatrix] = {}
        self._lock = threading.RLock()

    def register(
        self,
        descriptor: CapabilityDescriptor,
        handler: Optional[CapabilityHandler] = None,
        *,
        owner: str = "runtime",
        metadata: Optional[MetadataDict] = None,
        overwrite: bool = False,
    ) -> RegisteredCapability:
        """注册一项能力。"""

        with self._lock:
            if descriptor.name in self._registrations and not overwrite:
                raise ValueError(f"能力 {descriptor.name} 已存在。")
            registration = RegisteredCapability(
                descriptor=descriptor,
                handler=handler,
                owner=owner,
                metadata=dict(metadata or {}),
            )
            self._registrations[descriptor.name] = registration
            return registration

    def bind_robot_capability_matrix(self, matrix: CapabilityMatrix) -> None:
        """绑定机器人能力矩阵。"""

        with self._lock:
            self._robot_matrices[matrix.robot_model] = matrix

    def list_names(self) -> Tuple[str, ...]:
        """列出全部能力名称。"""

        with self._lock:
            return tuple(sorted(self._registrations))

    def get_registration(self, capability_name: str) -> RegisteredCapability:
        """获取能力注册记录。"""

        capability_name = validate_capability_name(capability_name)
        with self._lock:
            if capability_name not in self._registrations:
                raise CapabilityNotRegisteredError(capability_name)
            return self._registrations[capability_name]

    def get_descriptor(self, capability_name: str) -> CapabilityDescriptor:
        """获取能力描述。"""

        return self.get_registration(capability_name).descriptor

    def get_handler(self, capability_name: str) -> Optional[CapabilityHandler]:
        """获取能力处理器。"""

        return self.get_registration(capability_name).handler

    def supports_async(self, capability_name: str) -> bool:
        """判断能力是否为异步任务。"""

        descriptor = self.get_descriptor(capability_name)
        return descriptor.execution_mode == CapabilityExecutionMode.ASYNC_TASK

    def get_availability(self, capability_name: str, robot_model: Optional[str] = None) -> CapabilityAvailability:
        """获取某项能力在指定机器人上的可用性。"""

        capability_name = validate_capability_name(capability_name)
        with self._lock:
            registration = self._registrations.get(capability_name)
            if registration is None:
                return CapabilityAvailability(name=capability_name, supported=False, reason="能力未注册。")

            if robot_model is None:
                if registration.runnable:
                    return CapabilityAvailability(name=capability_name, supported=True)
                return CapabilityAvailability(name=capability_name, supported=False, reason="能力已声明但未注册处理器。")

            matrix = self._robot_matrices.get(robot_model)
            if matrix is None:
                return CapabilityAvailability(name=capability_name, supported=False, reason=f"机器人 {robot_model} 未绑定能力矩阵。")

            matrix_item = next((item for item in matrix.capabilities if item.name == capability_name), None)
            if matrix_item is None:
                return CapabilityAvailability(name=capability_name, supported=False, reason="能力未在机器人矩阵中声明。")
            if not matrix_item.supported:
                return matrix_item
            if not registration.runnable:
                return CapabilityAvailability(name=capability_name, supported=False, reason="能力已在矩阵声明，但运行时未注册处理器。")
            return CapabilityAvailability(
                name=capability_name,
                supported=True,
                metadata=dict(matrix_item.metadata),
            )

    def build_runtime_views(
        self,
        *,
        robot_model: Optional[str] = None,
        exposed_only: Optional[bool] = None,
    ) -> Tuple[CapabilityRuntimeView, ...]:
        """构造可直接暴露给网关的运行时能力视图。"""

        with self._lock:
            views: List[CapabilityRuntimeView] = []
            for name, registration in sorted(self._registrations.items()):
                descriptor = registration.descriptor
                if exposed_only is not None and descriptor.exposed_to_agent != exposed_only:
                    continue
                availability = self.get_availability(name, robot_model=robot_model)
                views.append(
                    CapabilityRuntimeView(
                        name=name,
                        owner=registration.owner,
                        descriptor=descriptor,
                        execution_mode=descriptor.execution_mode,
                        supported=availability.supported,
                        reason=availability.reason,
                        runnable=registration.runnable,
                        robot_model=robot_model,
                        metadata=dict(registration.metadata),
                    )
                )
            return tuple(views)

    def assert_supported(self, capability_name: str, robot_model: Optional[str] = None) -> None:
        """断言某项能力当前可用。"""

        availability = self.get_availability(capability_name, robot_model=robot_model)
        if not availability.supported:
            raise RuntimeError(availability.reason or f"能力 {capability_name} 当前不可用。")
