import hashlib
import os
from pathlib import Path
import re
import uuid

from app.errors import AppError

# Only allow letters, numbers, dots, underscores, and hyphens in model/version names
IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")

# Map each supported framework to the file extensions it is allowed to use
FRAMEWORK_EXTENSIONS: dict[str, set[str]] = {
    "sklearn": {".pkl", ".joblib"},
    "onnx": {".onnx"},
    "custom": {".py"},
}


def validate_identifier(value: str, field_name: str = "identifier") -> str:
    # Block empty names, weird characters, or '..' so nobody can sneak out of our folder
    if not value or ".." in value or not IDENTIFIER_RE.match(value):
        raise AppError(
            code="INVALID_IDENTIFIER",
            message=f"Invalid {field_name}: '{value}'. Must match ^[a-zA-Z0-9][a-zA-Z0-9._-]{{0,63}}$ without '..'.",
            status_code=400,
        )
    return value


class ArtifactStore:
    def __init__(self, base_dir: Path) -> None:
        # Resolve the storage folder once and create it if it isn't there yet
        self.base_dir = base_dir.resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _safe_target_dir(self, model_name: str, version: str) -> Path:
        # Build the folder path <base_dir>/<model_name>/<version> after checking both names
        validate_identifier(model_name, "model_name")
        validate_identifier(version, "version")
        target = (self.base_dir / model_name / version).resolve()
        # Extra safety check to make sure the final path is still inside our storage folder
        if not target.is_relative_to(self.base_dir):
            raise AppError(
                code="INVALID_IDENTIFIER",
                message="Resolved artifact path escapes storage root",
                status_code=400,
            )
        return target

    def save_bytes(
        self,
        model_name: str,
        version: str,
        framework: str,
        original_filename: str | None,
        data: bytes,
        max_bytes: int,
    ) -> tuple[str, int, str]:
        # Make sure the framework is one we support
        if framework not in FRAMEWORK_EXTENSIONS:
            raise AppError(
                code="UNSUPPORTED_FRAMEWORK",
                message=f"Unsupported framework '{framework}'. Allowed: {sorted(FRAMEWORK_EXTENSIONS)}",
                status_code=400,
            )
        # Reject empty uploads and files that are larger than our size limit
        if not data:
            raise AppError(code="MISSING_ARTIFACT", message="Uploaded artifact file is empty", status_code=400)
        if len(data) > max_bytes:
            raise AppError(
                code="ARTIFACT_TOO_LARGE",
                message=f"Artifact size {len(data)} bytes exceeds limit of {max_bytes} bytes",
                status_code=413,
            )

        # Make sure the uploaded file extension matches the chosen framework
        ext = Path(original_filename or "").suffix.lower()
        allowed_exts = FRAMEWORK_EXTENSIONS[framework]
        if ext not in allowed_exts:
            raise AppError(
                code="UNSUPPORTED_FRAMEWORK",
                message=f"File extension '{ext or '(none)'}' does not match framework '{framework}' (expected {sorted(allowed_exts)})",
                status_code=400,
            )

        # Create the version folder and pick a temporary name before moving into place
        target_dir = self._safe_target_dir(model_name, version)
        target_dir.mkdir(parents=True, exist_ok=True)
        final_path = target_dir / f"artifact{ext}"
        tmp_path = target_dir / f".tmp_{uuid.uuid4().hex}{ext}"

        # Write to a temp file first, then swap it in so a crash never leaves a half-written file
        sha256 = hashlib.sha256(data).hexdigest()
        try:
            tmp_path.write_bytes(data)
            os.replace(tmp_path, final_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        return str(final_path), len(data), sha256

    def resolve_download_path(self, artifact_location: str) -> Path:
        # Check that the file actually exists on disk before sending it back to the caller
        path = Path(artifact_location).resolve()
        if not path.is_file():
            raise AppError(
                code="ARTIFACT_NOT_FOUND",
                message="Model artifact file is not available locally for download",
                status_code=404,
            )
        return path

    def delete_artifact(self, artifact_location: str) -> None:
        # Skip remote URLs; we only delete files that live inside our local storage folder
        if artifact_location.startswith(("http://", "https://", "s3://")):
            return
        path = Path(artifact_location).resolve()
        if not (path.is_relative_to(self.base_dir) and path.exists()):
            return
        # Remove the file and also remove its version folder if it's now empty
        path.unlink(missing_ok=True)
        parent = path.parent
        if parent != self.base_dir and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
