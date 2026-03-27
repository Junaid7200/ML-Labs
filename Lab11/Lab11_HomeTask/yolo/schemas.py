from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    POSE = "pose"
    OBB = "obb"

    @property
    def label(self) -> str:
        return {
            TaskType.DETECTION: "Object Detection",
            TaskType.SEGMENTATION: "Instance Segmentation",
            TaskType.CLASSIFICATION: "Image Classification",
            TaskType.POSE: "Pose Estimation",
            TaskType.OBB: "Oriented Bounding Boxes",
        }[self]


class InputMetadata(BaseModel):
    filename: str
    content_type: str
    width: int
    height: int
    size_bytes: int


class SummaryMetadata(BaseModel):
    inference_time_ms: float
    prediction_count: int
    headline: str


class ArtifactLinks(BaseModel):
    annotated_image_url: str
    result_json_url: str


class InferenceResponse(BaseModel):
    status: str = "success"
    task: TaskType
    input: InputMetadata
    summary: SummaryMetadata
    predictions: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: ArtifactLinks


class HealthResponse(BaseModel):
    status: str
    ready: bool
    loaded_tasks: list[str]
