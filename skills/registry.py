from __future__ import annotations

import threading

from contracts.capabilities import CapabilityDescriptor
from contracts.skills import SkillDescriptor, SkillRuntimeView
from core import CapabilityRegistry
from typing import Dict, List, Optional, Tuple


class SkillRegistry:
    """技能注册表。"""

    def __init__(self) -> None:
        self._descriptors: Dict[str, SkillDescriptor] = {}
        self._lock = threading.RLock()

    def register(self, descriptor: SkillDescriptor, *, overwrite: bool = False) -> SkillDescriptor:
        """注册技能。"""

        with self._lock:
            if descriptor.name in self._descriptors and not overwrite:
                raise ValueError(f"技能 {descriptor.name} 已存在。")
            self._descriptors[descriptor.name] = descriptor
            return descriptor

    def get_descriptor(self, skill_name: str) -> SkillDescriptor:
        """读取技能描述。"""

        with self._lock:
            return self._descriptors[skill_name]

    def list_descriptors(self) -> Tuple[SkillDescriptor, ...]:
        """列出全部技能描述。"""

        with self._lock:
            return tuple(self._descriptors[name] for name in sorted(self._descriptors))

    def build_runtime_views(
        self,
        capability_registry: CapabilityRegistry,
        *,
        robot_model: str,
        exposed_only: Optional[bool] = None,
        include_unsupported: bool = False,
    ) -> Tuple[SkillRuntimeView, ...]:
        """构造技能运行时视图。"""

        views: List[SkillRuntimeView] = []
        with self._lock:
            descriptors = [self._descriptors[name] for name in sorted(self._descriptors)]
        for descriptor in descriptors:
            if exposed_only is not None and descriptor.exposed_to_agent != exposed_only:
                continue
            availability = capability_registry.get_availability(descriptor.capability_name, robot_model=robot_model)
            registration = capability_registry.get_registration(descriptor.capability_name)
            capability_descriptor: CapabilityDescriptor = registration.descriptor
            view = SkillRuntimeView(
                descriptor=descriptor,
                supported=availability.supported,
                runnable=registration.runnable,
                reason=availability.reason,
                capability_name=descriptor.capability_name,
                input_schema=capability_descriptor.input_schema,
                output_schema=capability_descriptor.output_schema,
                metadata={
                    "execution_mode": capability_descriptor.execution_mode.value,
                    "risk_level": capability_descriptor.risk_level.value,
                },
            )
            if include_unsupported or (view.supported and view.runnable):
                views.append(view)
        return tuple(views)
