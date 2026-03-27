from __future__ import annotations

import io

from PIL import Image


def create_image_bytes(fmt: str = "PNG") -> bytes:
    image = Image.new("RGB", (64, 64), color=(32, 140, 216))
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def test_frontend_route_renders(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "YOLO Vision Studio" in response.text


def test_health_route(client) -> None:
    response = client.get("/health")
    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ok"


def test_infer_works_for_all_tasks(client) -> None:
    for task in ["detection", "segmentation", "classification", "pose", "obb"]:
        response = client.post(
            "/api/v1/infer",
            data={"task": task},
            files={"file": ("sample.png", create_image_bytes(), "image/png")},
        )
        payload = response.json()
        assert response.status_code == 200
        assert payload["status"] == "success"
        assert payload["task"] == task
        assert "annotated_image_url" in payload["artifacts"]
        assert "result_json_url" in payload["artifacts"]


def test_infer_rejects_missing_file(client) -> None:
    response = client.post("/api/v1/infer", data={"task": "detection"})
    assert response.status_code == 422


def test_infer_rejects_invalid_task(client) -> None:
    response = client.post(
        "/api/v1/infer",
        data={"task": "unknown"},
        files={"file": ("sample.png", create_image_bytes(), "image/png")},
    )
    assert response.status_code == 422


def test_artifact_download_routes_work(client) -> None:
    response = client.post(
        "/api/v1/infer",
        data={"task": "detection"},
        files={"file": ("sample.png", create_image_bytes(), "image/png")},
    )
    payload = response.json()

    image_response = client.get(payload["artifacts"]["annotated_image_url"])
    json_response = client.get(payload["artifacts"]["result_json_url"])

    assert image_response.status_code == 200
    assert json_response.status_code == 200
