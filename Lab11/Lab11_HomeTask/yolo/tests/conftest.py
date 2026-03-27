from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import create_app
from config import Settings
from schemas import TaskType
from services.artifacts import ArtifactManager


class FakeInferenceService:
    def __init__(self, artifact_manager: ArtifactManager) -> None:
        self.artifact_manager = artifact_manager

    async def infer(self, *, task: TaskType, upload, request):
        image_id = self.artifact_manager.create_artifact(
            content=b"fakepng",
            filename=f"{task.value}.png",
            media_type="image/png",
            suffix=".png",
        )
        json_id = self.artifact_manager.create_artifact(
            content=json.dumps({"task": task.value}).encode("utf-8"),
            filename=f"{task.value}.json",
            media_type="application/json",
            suffix=".json",
        )
        return {
            "status": "success",
            "task": task,
            "input": {
                "filename": upload.filename,
                "content_type": upload.content_type,
                "width": 64,
                "height": 64,
                "size_bytes": 512,
            },
            "summary": {
                "inference_time_ms": 12.3,
                "prediction_count": 1,
                "headline": f"{task.value} headline",
            },
            "predictions": [{"class_name": "sample", "confidence": 0.99}],
            "artifacts": {
                "annotated_image_url": str(request.url_for("serve_artifact", artifact_id=image_id)),
                "result_json_url": str(request.url_for("serve_artifact", artifact_id=json_id)),
            },
        }


def create_image_bytes(fmt: str = "PNG") -> bytes:
    image = Image.new("RGB", (64, 64), color=(32, 140, 216))
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    settings = Settings(load_models_on_startup=False)
    settings.runtime_dir = tmp_path / "runtime"
    settings.artifacts_dir = settings.runtime_dir / "artifacts"
    settings.ensure_runtime_dirs()
    return settings


@pytest.fixture
def client(settings: Settings) -> TestClient:
    artifact_manager = ArtifactManager(settings)
    app = create_app(settings=settings, inference_service=FakeInferenceService(artifact_manager))
    return TestClient(app)
