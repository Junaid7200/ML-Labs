from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from config import Settings
from exceptions import ArtifactNotFoundError


@dataclass(slots=True)
class ArtifactRecord:
    artifact_id: str
    path: Path
    filename: str
    media_type: str
    expires_at: int


class ArtifactManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_runtime_dirs()

    def create_artifact(
        self,
        *,
        content: bytes,
        filename: str,
        media_type: str,
        suffix: str,
    ) -> str:
        self.cleanup_expired()
        artifact_id = f"{uuid.uuid4().hex}{suffix}"
        path = self.settings.artifacts_dir / artifact_id
        expires_at = int(time.time()) + self.settings.artifact_ttl_seconds
        path.write_bytes(content)
        meta = {
            "filename": filename,
            "media_type": media_type,
            "expires_at": expires_at,
        }
        path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return artifact_id

    def get_artifact(self, artifact_id: str) -> ArtifactRecord:
        self.cleanup_expired()
        path = self.settings.artifacts_dir / artifact_id
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if not path.exists() or not meta_path.exists():
            raise ArtifactNotFoundError()

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["expires_at"] < int(time.time()):
            self._delete_pair(path)
            raise ArtifactNotFoundError()

        return ArtifactRecord(
            artifact_id=artifact_id,
            path=path,
            filename=meta["filename"],
            media_type=meta["media_type"],
            expires_at=meta["expires_at"],
        )

    def cleanup_expired(self) -> None:
        now = int(time.time())
        for meta_path in self.settings.artifacts_dir.glob("*.meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta_path.unlink(missing_ok=True)
                continue

            if meta.get("expires_at", 0) < now:
                artifact_name = meta_path.name.removesuffix(".meta.json")
                artifact_path = self.settings.artifacts_dir / artifact_name
                self._delete_pair(artifact_path)

    def _delete_pair(self, artifact_path: Path) -> None:
        artifact_path.unlink(missing_ok=True)
        artifact_path.with_suffix(artifact_path.suffix + ".meta.json").unlink(missing_ok=True)
