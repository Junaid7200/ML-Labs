from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

from fastapi import Request, UploadFile
from PIL import Image

from config import Settings
from exceptions import ModelInferenceError, ValidationError
from schemas import ArtifactLinks, InferenceResponse, InputMetadata, SummaryMetadata, TaskType
from services.artifacts import ArtifactManager
from services.model_registry import ModelRegistry
from services.rendering import render_annotated_image

POSE_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


class InferenceService:
    def __init__(self, settings: Settings, registry: ModelRegistry, artifact_manager: ArtifactManager) -> None:
        self.settings = settings
        self.registry = registry
        self.artifact_manager = artifact_manager

    async def infer(self, *, task: TaskType, upload: UploadFile, request: Request) -> InferenceResponse:
        image_bytes, image = await self._read_and_validate_image(upload)
        filename = upload.filename or "upload"
        model = self.registry.get(task)

        started_at = time.perf_counter()
        try:
            results = model.predict(image, verbose=False)
        except Exception as exc:  # pragma: no cover - defensive path
            raise ModelInferenceError(str(exc)) from exc

        inference_time_ms = round((time.perf_counter() - started_at) * 1000, 2)
        result = results[0]
        predictions = self._serialize_predictions(task, result)
        annotated_image = render_annotated_image(task, image, result, predictions)

        input_meta = InputMetadata(
            filename=filename,
            content_type=upload.content_type or "application/octet-stream",
            width=image.width,
            height=image.height,
            size_bytes=len(image_bytes),
        )
        summary = SummaryMetadata(
            inference_time_ms=inference_time_ms,
            prediction_count=len(predictions),
            headline=self._headline(task, predictions),
        )

        base_name = Path(filename).stem or "upload"
        annotated_id = self.artifact_manager.create_artifact(
            content=annotated_image,
            filename=f"{base_name}-{task.value}.png",
            media_type="image/png",
            suffix=".png",
        )
        payload = {
            "status": "success",
            "task": task.value,
            "input": input_meta.model_dump(),
            "summary": summary.model_dump(),
            "predictions": predictions,
        }
        json_id = self.artifact_manager.create_artifact(
            content=json.dumps(payload, indent=2).encode("utf-8"),
            filename=f"{base_name}-{task.value}.json",
            media_type="application/json",
            suffix=".json",
        )

        return InferenceResponse(
            status="success",
            task=task,
            input=input_meta,
            summary=summary,
            predictions=predictions,
            artifacts=ArtifactLinks(
                annotated_image_url=self._artifact_url(request, annotated_id),
                result_json_url=self._artifact_url(request, json_id),
            ),
        )

    async def _read_and_validate_image(self, upload: UploadFile) -> tuple[bytes, Image.Image]:
        if upload is None:
            raise ValidationError("A file upload is required.")
        if not upload.filename:
            raise ValidationError("Uploaded file must include a filename.")
        if upload.content_type not in self.settings.allowed_image_types:
            raise ValidationError(
                f"Unsupported content type '{upload.content_type}'. Allowed types: {', '.join(self.settings.allowed_image_types)}."
            )

        image_bytes = await upload.read()
        if not image_bytes:
            raise ValidationError("Uploaded file is empty.")
        if len(image_bytes) > self.settings.max_upload_size_bytes:
            raise ValidationError(
                f"Uploaded file exceeds the {self.settings.max_upload_size_bytes // (1024 * 1024)} MB limit."
            )

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValidationError("Uploaded file could not be parsed as an image.") from exc
        return image_bytes, image

    def _artifact_url(self, request: Request, artifact_id: str) -> str:
        relative = str(request.url_for("serve_artifact", artifact_id=artifact_id))
        if self.settings.app_base_url:
            return f"{self.settings.app_base_url}/artifacts/{artifact_id}"
        return relative

    def _serialize_predictions(self, task: TaskType, result: Any) -> list[dict[str, Any]]:
        if task is TaskType.DETECTION:
            return self._serialize_detection(result)
        if task is TaskType.SEGMENTATION:
            return self._serialize_segmentation(result)
        if task is TaskType.CLASSIFICATION:
            return self._serialize_classification(result)
        if task is TaskType.POSE:
            return self._serialize_pose(result)
        if task is TaskType.OBB:
            return self._serialize_obb(result)
        return []

    def _headline(self, task: TaskType, predictions: list[dict[str, Any]]) -> str:
        if task is TaskType.CLASSIFICATION:
            if not predictions:
                return "No classes predicted"
            top = predictions[0]
            return f"Top class: {top['class_name']} ({top['confidence'] * 100:.1f}%)"
        if task is TaskType.POSE:
            return f"Detected {len(predictions)} pose(s)"
        if task is TaskType.SEGMENTATION:
            return f"Segmented {len(predictions)} object(s)"
        if task is TaskType.OBB:
            return f"Detected {len(predictions)} oriented object(s)"
        return f"Detected {len(predictions)} object(s)"

    @staticmethod
    def _to_list(value: Any) -> Any:
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    def _serialize_detection(self, result: Any) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        if result.boxes is None:
            return predictions

        for box in result.boxes:
            class_id = int(box.cls[0])
            predictions.append(
                {
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": [round(float(value), 2) for value in self._to_list(box.xyxy[0])],
                }
            )
        return predictions

    def _serialize_segmentation(self, result: Any) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        if result.masks is None:
            return predictions

        polygons = getattr(result.masks, "xy", [])
        for index, polygon in enumerate(polygons):
            box = result.boxes[index]
            class_id = int(box.cls[0])
            predictions.append(
                {
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": [round(float(value), 2) for value in self._to_list(box.xyxy[0])],
                    "mask_points": [[round(float(x), 2), round(float(y), 2)] for x, y in self._to_list(polygon)],
                }
            )
        return predictions

    def _serialize_classification(self, result: Any) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        top5_indices = list(result.probs.top5)
        top5_conf = list(self._to_list(result.probs.top5conf))
        for rank, (class_id, confidence) in enumerate(zip(top5_indices, top5_conf), start=1):
            predictions.append(
                {
                    "rank": rank,
                    "class_id": int(class_id),
                    "class_name": result.names[int(class_id)],
                    "confidence": round(float(confidence), 4),
                }
            )
        return predictions

    def _serialize_pose(self, result: Any) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        if result.keypoints is None:
            return predictions

        points = self._to_list(result.keypoints.xy)
        confidences = self._to_list(result.keypoints.conf) if result.keypoints.conf is not None else None
        for person_index, person_points in enumerate(points):
            keypoints = []
            point_confidences = confidences[person_index] if confidences else [1.0] * len(person_points)
            for idx, (coords, confidence) in enumerate(zip(person_points, point_confidences)):
                keypoints.append(
                    {
                        "name": POSE_KEYPOINT_NAMES[idx] if idx < len(POSE_KEYPOINT_NAMES) else f"kp_{idx}",
                        "x": round(float(coords[0]), 2),
                        "y": round(float(coords[1]), 2),
                        "confidence": round(float(confidence), 4),
                    }
                )
            bbox = None
            if result.boxes is not None and len(result.boxes) > person_index:
                bbox = [round(float(value), 2) for value in self._to_list(result.boxes[person_index].xyxy[0])]
            predictions.append(
                {
                    "person_index": person_index,
                    "bbox": bbox,
                    "keypoint_count": len(keypoints),
                    "keypoints": keypoints,
                }
            )
        return predictions

    def _serialize_obb(self, result: Any) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        if result.obb is None:
            return predictions

        polygons = self._to_list(result.obb.xyxyxyxy)
        confidences = self._to_list(result.obb.conf)
        class_ids = self._to_list(result.obb.cls)
        for index, polygon in enumerate(polygons):
            class_id = int(class_ids[index])
            predictions.append(
                {
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": round(float(confidences[index]), 4),
                    "rotated_bbox": [[round(float(x), 2), round(float(y), 2)] for x, y in polygon],
                }
            )
        return predictions
