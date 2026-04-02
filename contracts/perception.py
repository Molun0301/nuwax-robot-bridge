from __future__ import annotations

from compat import StrEnum

from pydantic import Field, field_validator

from contracts.base import MetadataDict, TimestampedContract
from contracts.geometry import Pose
from contracts.naming import validate_frame_id, validate_observation_id
from typing import List, Optional


class BoundingBox2D(TimestampedContract):
    """二维检测框。"""

    x_px: float = Field(ge=0.0, description="左上角 X。")
    y_px: float = Field(ge=0.0, description="左上角 Y。")
    width_px: float = Field(gt=0.0, description="宽度。")
    height_px: float = Field(gt=0.0, description="高度。")


class Detection2D(TimestampedContract):
    """二维检测结果。"""

    label: str = Field(description="检测标签。")
    score: float = Field(ge=0.0, le=1.0, description="置信度。")
    bbox: BoundingBox2D = Field(description="二维检测框。")
    camera_id: Optional[str] = Field(default=None, description="来源相机。")
    track_id: Optional[str] = Field(default=None, description="关联轨迹。")
    attributes: MetadataDict = Field(default_factory=dict, description="附加属性。")


class Detection3D(TimestampedContract):
    """三维检测结果。"""

    label: str = Field(description="检测标签。")
    score: float = Field(ge=0.0, le=1.0, description="置信度。")
    pose: Pose = Field(description="三维目标位姿。")
    size_x_m: Optional[float] = Field(default=None, gt=0.0, description="X 尺寸。")
    size_y_m: Optional[float] = Field(default=None, gt=0.0, description="Y 尺寸。")
    size_z_m: Optional[float] = Field(default=None, gt=0.0, description="Z 尺寸。")
    attributes: MetadataDict = Field(default_factory=dict, description="附加属性。")


class TrackState(StrEnum):
    """轨迹状态。"""

    TENTATIVE = "tentative"
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


class Track(TimestampedContract):
    """目标轨迹。"""

    track_id: str = Field(description="轨迹标识。")
    label: str = Field(description="目标标签。")
    state: TrackState = Field(default=TrackState.TENTATIVE, description="轨迹状态。")
    score: float = Field(ge=0.0, le=1.0, description="置信度。")
    pose: Optional[Pose] = Field(default=None, description="三维位姿。")
    bbox: Optional[BoundingBox2D] = Field(default=None, description="二维检测框。")
    attributes: MetadataDict = Field(default_factory=dict, description="附加属性。")


class Observation(TimestampedContract):
    """观察结果聚合。"""

    observation_id: str = Field(description="观察标识。")
    frame_id: str = Field(description="观察所属坐标系。")
    summary: Optional[str] = Field(default=None, description="面向上层的中文摘要。")
    detections_2d: List[Detection2D] = Field(default_factory=list, description="二维检测结果。")
    detections_3d: List[Detection3D] = Field(default_factory=list, description="三维检测结果。")
    tracks: List[Track] = Field(default_factory=list, description="轨迹列表。")
    artifact_ids: List[str] = Field(default_factory=list, description="关联制品标识。")
    metadata: MetadataDict = Field(default_factory=dict, description="附加元数据。")

    @field_validator("frame_id")
    @classmethod
    def _validate_frame_id(cls, value: str) -> str:
        return validate_frame_id(value)

    @field_validator("observation_id")
    @classmethod
    def _validate_observation_id(cls, value: str) -> str:
        return validate_observation_id(value)
