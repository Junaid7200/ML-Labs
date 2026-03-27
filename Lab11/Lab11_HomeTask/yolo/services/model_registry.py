from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from config import Settings
from schemas import TaskType


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._models: dict[TaskType, YOLO] = {}
        self._paths = {
            TaskType.DETECTION: self.settings.models_dir / "yolo11n.pt",
            TaskType.SEGMENTATION: self.settings.models_dir / "yolo11n-seg.pt",
            TaskType.CLASSIFICATION: self.settings.models_dir / "yolo11n-cls.pt",
            TaskType.POSE: self.settings.models_dir / "yolo11n-pose.pt",
            TaskType.OBB: self.settings.models_dir / "yolo11n-obb.pt",
        }

    @property
    def is_ready(self) -> bool:
        return len(self._models) == len(self._paths)

    def loaded_tasks(self) -> list[str]:
        return [task.value for task in self._models]

    def load_all(self) -> None:
        for task in self._paths:
            self.get(task)

    def get(self, task: TaskType) -> YOLO:
        if task not in self._models:
            path = self._paths[task]
            self._validate_model_path(path)
            self._models[task] = YOLO(str(path))
        return self._models[task]

    @staticmethod
    def _validate_model_path(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Missing model weights: {path}")
