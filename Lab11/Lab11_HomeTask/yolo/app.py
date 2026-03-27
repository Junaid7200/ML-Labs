from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import Settings, get_settings
from exceptions import ArtifactNotFoundError, StudioError
from schemas import HealthResponse, InferenceResponse, TaskType
from services.artifacts import ArtifactManager
from services.inference import InferenceService
from services.model_registry import ModelRegistry


def create_app(
    settings: Settings | None = None,
    registry: ModelRegistry | None = None,
    artifact_manager: ArtifactManager | None = None,
    inference_service: InferenceService | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_runtime_dirs()

    registry = registry or ModelRegistry(settings)
    artifact_manager = artifact_manager or ArtifactManager(settings)
    inference_service = inference_service or InferenceService(settings, registry, artifact_manager)

    templates = Jinja2Templates(directory=str(settings.templates_dir))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.registry = registry
        app.state.artifact_manager = artifact_manager
        app.state.inference_service = inference_service
        app.state.templates = templates
        artifact_manager.cleanup_expired()
        if settings.load_models_on_startup:
            registry.load_all()
        yield

    app = FastAPI(
        title="YOLO Vision Studio",
        description="A multi-task computer vision studio for YOLO-based detection, segmentation, classification, pose estimation, and oriented boxes.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")

    @app.exception_handler(StudioError)
    async def studio_error_handler(_: Request, exc: StudioError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"status": "error", "error": exc.message})

    @app.exception_handler(ArtifactNotFoundError)
    async def artifact_error_handler(_: Request, exc: ArtifactNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"status": "error", "error": exc.message})

    @app.get("/", response_class=HTMLResponse)
    async def studio(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "page_title": "YOLO Vision Studio",
                "task_options": [{"value": task.value, "label": task.label} for task in TaskType],
                "max_upload_mb": settings.max_upload_size_bytes // (1024 * 1024),
            },
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            ready=registry.is_ready,
            loaded_tasks=registry.loaded_tasks(),
        )

    @app.post("/api/v1/infer", response_model=InferenceResponse)
    async def infer(
        request: Request,
        task: TaskType = Form(...),
        file: UploadFile = File(...),
    ) -> InferenceResponse:
        return await inference_service.infer(task=task, upload=file, request=request)

    @app.get("/artifacts/{artifact_id}")
    async def serve_artifact(artifact_id: str) -> FileResponse:
        artifact = artifact_manager.get_artifact(artifact_id)
        return FileResponse(
            artifact.path,
            media_type=artifact.media_type,
            filename=artifact.filename,
        )

    @app.get("/artifacts/{artifact_id}/json")
    async def serve_artifact_json(artifact_id: str) -> JSONResponse:
        artifact = artifact_manager.get_artifact(artifact_id)
        if artifact.media_type != "application/json":
            raise ArtifactNotFoundError("Artifact is not a JSON payload.")
        return JSONResponse(content=json.loads(Path(artifact.path).read_text(encoding="utf-8")))

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:create_app", factory=True, host="0.0.0.0", port=8000, reload=True)
