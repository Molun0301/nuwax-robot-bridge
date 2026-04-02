from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BaseProvider(Protocol):
    """所有提供器接口的共同最小边界。"""

    provider_name: str
    provider_version: str

    def is_available(self) -> bool:
        """返回当前提供器是否可用。"""

