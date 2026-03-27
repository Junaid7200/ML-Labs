from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from schemas import TaskType


def render_annotated_image(task: TaskType, image: Image.Image, result: Any, predictions: list[dict[str, Any]]) -> bytes:
    if task in {TaskType.DETECTION, TaskType.SEGMENTATION, TaskType.POSE, TaskType.OBB}:
        plotted = result.plot()
        if isinstance(plotted, np.ndarray):
            if plotted.ndim == 3 and plotted.shape[2] == 3:
                rendered = Image.fromarray(plotted[:, :, ::-1])
            else:
                rendered = Image.fromarray(plotted)
        else:
            rendered = image.copy()
    else:
        rendered = _render_classification_overlay(image.copy(), predictions)

    buffer = io.BytesIO()
    rendered.save(buffer, format="PNG")
    return buffer.getvalue()


def _render_classification_overlay(image: Image.Image, predictions: list[dict[str, Any]]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    top = predictions[:5]
    x1, y1 = 16, 16
    x2, y2 = min(canvas.width - 16, 400), min(canvas.height - 16, 180)
    draw.rounded_rectangle((x1, y1, x2, y2), radius=16, fill=(17, 24, 39, 220))
    draw.text((x1 + 16, y1 + 14), "Top Predictions", fill="white")
    line_y = y1 + 48
    for index, pred in enumerate(top, start=1):
        label = f"{index}. {pred['class_name']} - {pred['confidence'] * 100:.1f}%"
        draw.text((x1 + 16, line_y), label, fill=(220, 252, 231))
        line_y += 24
    return canvas
