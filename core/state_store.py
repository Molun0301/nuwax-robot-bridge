from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from compat import StrEnum
import threading
from typing import Dict, Optional, Tuple, Any

from contracts.base import MetadataDict, utc_now


class StateNamespace(StrEnum):
    """状态存储命名空间。"""

    ROBOT_STATE = "robot_state"
    LOCALIZATION = "localization"
    OBSERVATION = "observation"
    PERCEPTION = "perception"
    NAVIGATION = "navigation"
    MAP = "map"
    MEMORY = "memory"
    SAFETY = "safety"
    SYSTEM = "system"


@dataclass(frozen=True)
class StateEntry:
    """状态存储条目。"""

    namespace: StateNamespace
    key: str
    source: str
    updated_at: datetime
    value: Any
    metadata: MetadataDict = field(default_factory=dict)


class StateStore:
    """线程安全的最新状态缓存。"""

    def __init__(self) -> None:
        self._store: Dict[StateNamespace, Dict[str, StateEntry]] = {namespace: {} for namespace in StateNamespace}
        self._lock = threading.RLock()

    def write(
        self,
        namespace: StateNamespace,
        key: str,
        value: Any,
        *,
        source: str,
        metadata: Optional[MetadataDict] = None,
        updated_at: Optional[datetime] = None,
    ) -> StateEntry:
        """写入状态条目。"""

        entry = StateEntry(
            namespace=namespace,
            key=key,
            source=source,
            updated_at=updated_at or utc_now(),
            value=value,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._store[namespace][key] = entry
        return entry

    def write_from_provider(
        self,
        provider_name: str,
        namespace: StateNamespace,
        key: str,
        value: Any,
        *,
        metadata: Optional[MetadataDict] = None,
    ) -> StateEntry:
        """按提供器路径写入状态。"""

        return self.write(namespace, key, value, source=provider_name, metadata=metadata)

    def read(self, namespace: StateNamespace, key: str) -> Optional[StateEntry]:
        """读取单条状态。"""

        with self._lock:
            return self._store[namespace].get(key)

    def read_latest(self, namespace: StateNamespace) -> Optional[StateEntry]:
        """读取某命名空间最新条目。"""

        with self._lock:
            entries = list(self._store[namespace].values())
        if not entries:
            return None
        return max(entries, key=lambda item: item.updated_at)

    def snapshot(self, namespace: StateNamespace) -> Tuple[StateEntry, ...]:
        """读取某命名空间完整快照。"""

        with self._lock:
            entries = list(self._store[namespace].values())
        entries.sort(key=lambda item: item.updated_at)
        return tuple(entries)
