from __future__ import annotations

import io
import time
from pathlib import Path

import pytest
from fastapi import UploadFile
from PIL import Image
from starlette.datastructures import Headers

from config import Settings
from exceptions import ArtifactNotFoundError, ValidationError
from schemas import TaskType
from services.artifacts import ArtifactManager
from services.inference import InferenceService


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class FakeBox:
    def __init__(self):
        self.cls = [1]
        self.conf = [0.87]
        self.xyxy = [FakeTensor([10.0, 12.0, 42.0, 58.0])]


class FakeMasks:
    xy = [FakeTensor([[1, 2], [3, 4], [5, 6]])]


class FakeProbs:
    top5 = [0, 1, 2]
    top5conf = FakeTensor([0.91, 0.05, 0.03])


class FakeKeypoints:
    xy = FakeTensor([[[10, 20], [12, 24]]])
    conf = FakeTensor([[0.92, 0.88]])


class FakeObb:
    xyxyxyxy = FakeTensor([[[0, 0], [10, 0], [10, 8], [0, 8]]])
    conf = FakeTensor([0.66])
    cls = FakeTensor([0])


class FakeResult:
    names = {0: "plane", 1: "person", 2: "car"}
    boxes = [FakeBox()]
    masks = FakeMasks()
    probs = FakeProbs()
    keypoints = FakeKeypoints()
    obb = FakeObb()

    def plot(self):
        image = Image.new("RGB", (64, 64), color=(10, 20, 30))
        return image


class DummyRegistry:
    def get(self, task):
        return None


@pytest.fixture
def service(tmp_path: Path) -> InferenceService:
    settings = Settings(load_models_on_startup=False, artifact_ttl_seconds=1)
    settings.runtime_dir = tmp_path / "runtime"
    settings.artifacts_dir = settings.runtime_dir / "artifacts"
    settings.ensure_runtime_dirs()
    return InferenceService(settings, DummyRegistry(), ArtifactManager(settings))


@pytest.mark.asyncio
async def test_validation_rejects_empty_upload(service: InferenceService) -> None:
    upload = UploadFile(filename="empty.png", file=io.BytesIO(b""), headers=Headers({"content-type": "image/png"}))
    with pytest.raises(ValidationError):
        await service._read_and_validate_image(upload)


@pytest.mark.asyncio
async def test_validation_rejects_unsupported_type(service: InferenceService) -> None:
    upload = UploadFile(filename="sample.gif", file=io.BytesIO(b"gif"), headers=Headers({"content-type": "image/gif"}))
    with pytest.raises(ValidationError):
        await service._read_and_validate_image(upload)


@pytest.mark.asyncio
async def test_validation_rejects_oversized_upload(service: InferenceService) -> None:
    service.settings.max_upload_size_bytes = 4
    upload = UploadFile(
        filename="large.png",
        file=io.BytesIO(b"0123456789"),
        headers=Headers({"content-type": "image/png"}),
    )
    with pytest.raises(ValidationError):
        await service._read_and_validate_image(upload)


def test_artifact_manager_expires_files(tmp_path: Path) -> None:
    settings = Settings(load_models_on_startup=False, artifact_ttl_seconds=1)
    settings.runtime_dir = tmp_path / "runtime"
    settings.artifacts_dir = settings.runtime_dir / "artifacts"
    settings.ensure_runtime_dirs()
    manager = ArtifactManager(settings)
    artifact_id = manager.create_artifact(
        content=b"hello",
        filename="hello.txt",
        media_type="text/plain",
        suffix=".txt",
    )
    time.sleep(2)
    with pytest.raises(ArtifactNotFoundError):
        manager.get_artifact(artifact_id)


def test_prediction_serializers(service: InferenceService) -> None:
    result = FakeResult()
    assert service._serialize_predictions(TaskType.DETECTION, result)[0]["class_name"] == "person"
    assert service._serialize_predictions(TaskType.SEGMENTATION, result)[0]["mask_points"]
    assert service._serialize_predictions(TaskType.CLASSIFICATION, result)[0]["rank"] == 1
    assert service._serialize_predictions(TaskType.POSE, result)[0]["keypoint_count"] == 2
    assert service._serialize_predictions(TaskType.OBB, result)[0]["class_name"] == "plane"
