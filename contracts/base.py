from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field

MetadataDict = Dict[str, Any]


def utc_now() -> datetime:
    """返回带 UTC 时区的当前时间。"""

    return datetime.now(timezone.utc)


class ContractModel(BaseModel):
    """平台统一契约模型基类。"""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
        use_enum_values=False,
    )


class TimestampedContract(ContractModel):
    """带统一时间戳的契约基类。"""

    timestamp: datetime = Field(default_factory=utc_now, description="UTC 时间戳。")
