# YOLO Vision Studio

YOLO Vision Studio is a FastAPI-based computer vision workbench built around five YOLO tasks in one interface:

- Object detection
- Instance segmentation
- Image classification
- Pose estimation
- Oriented bounding boxes

It is designed as a portfolio-grade project rather than a classroom demo. Users upload one image, choose a task, run inference, inspect structured results, and download both the annotated output image and the raw JSON payload.

## Highlights

- Single studio workflow instead of separate mini demos
- FastAPI backend with typed response models and automatic docs
- Unified `POST /api/v1/infer` endpoint for all five tasks
- Annotated artifact generation for every run
- Temporary artifact storage with automatic expiration
- Public-demo-ready Docker setup for Render or Railway style hosting
- Tests covering API routes, artifact lifecycle, and serializer behavior

## Architecture

- `app.py`
  FastAPI app factory, routes, lifespan wiring, and exception handling
- `services/model_registry.py`
  Lazy model loading and per-task weight resolution
- `services/inference.py`
  Upload validation, model execution, normalization, and artifact creation
- `services/artifacts.py`
  Ephemeral artifact storage and TTL cleanup
- `services/rendering.py`
  Annotated image generation

## API

### `POST /api/v1/infer`

Multipart form fields:

- `task`
  One of `detection`, `segmentation`, `classification`, `pose`, `obb`
- `file`
  A PNG, JPEG, or WEBP image

Returns:

- task metadata
- input metadata
- summary metrics
- normalized predictions
- temporary URLs for the annotated image and raw JSON payload

### `GET /artifacts/{artifact_id}`

Downloads a generated artifact if it has not expired yet.

### `GET /health`

Returns service readiness and loaded task state.

## Local setup

From this folder:

```bash
uv sync
uv run uvicorn app:create_app --factory --reload
```

Then open:

- `http://localhost:8000/`
- `http://localhost:8000/docs`

## Docker

Build from the `Lab11/Lab11_HomeTask/yolo` directory:

```bash
docker build -t yolo-vision-studio .
docker run -p 8000:8000 yolo-vision-studio
```

## Deployment notes

This project is intended for straightforward container deployment on platforms like Render or Railway.

Recommended production command:

```bash
uvicorn app:create_app --factory --host 0.0.0.0 --port $PORT
```

Environment variables are documented in `.env.example`.

## Resume-ready framing

- Built a FastAPI-based multi-task computer vision studio supporting detection, segmentation, classification, pose estimation, and oriented bounding box inference through a unified upload workflow
- Designed a normalized inference pipeline with typed API responses, annotated artifact generation, and expiring temporary file storage for public-demo-friendly result delivery
- Added Docker deployment, automatic API documentation, and automated tests to harden a YOLO prototype into a portfolio-grade application

## Media placeholders

Add these later for portfolio polish:

- `docs/studio-home.png`
- `docs/detection-output.png`
- `docs/pose-output.png`
- `docs/demo.gif`
