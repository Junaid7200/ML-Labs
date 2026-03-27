from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


@dataclass(slots=True)
class Settings:
    project_name: str = "YOLO Vision Studio"
    environment: str = os.getenv("YOLO_STUDIO_ENV", "development")
    app_base_url: str = os.getenv("YOLO_STUDIO_BASE_URL", "").rstrip("/")
    port: int = int(os.getenv("PORT", "8000"))
    load_models_on_startup: bool = os.getenv("YOLO_STUDIO_LOAD_MODELS_ON_STARTUP", "true").lower() == "true"
    max_upload_size_bytes: int = int(os.getenv("YOLO_STUDIO_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
    artifact_ttl_seconds: int = int(os.getenv("YOLO_STUDIO_ARTIFACT_TTL_SECONDS", "3600"))
    allowed_image_types: tuple[str, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
    )
    base_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    templates_dir: Path = field(init=False)
    static_dir: Path = field(init=False)
    runtime_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.base_dir = Path(__file__).resolve().parent
        self.models_dir = self.base_dir / "models"
        self.templates_dir = self.base_dir / "templates"
        self.static_dir = self.base_dir / "static"
        self.runtime_dir = self.base_dir / "runtime"
        self.artifacts_dir = self.runtime_dir / "artifacts"

    def ensure_runtime_dirs(self) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
