from __future__ import annotations


class StudioError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ValidationError(StudioError):
    pass


class ArtifactNotFoundError(StudioError):
    def __init__(self, message: str = "Artifact not found or expired.") -> None:
        super().__init__(message=message, status_code=404)


class ModelInferenceError(StudioError):
    def __init__(self, message: str = "Model inference failed.") -> None:
        super().__init__(message=message, status_code=500)
