import hashlib
import json
from pathlib import Path
import re
import uuid

from jsonschema import Draft7Validator, SchemaError
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import Deployment, Model, ModelVersion, utcnow
from app.db.queries import require_model, require_model_version
from app.errors import AppError
from app.schemas.registry import (
    DeleteVersionResponse,
    ModelMetadataInput,
    ModelSummary,
    ModelUploadResponse,
    ModelVersionDetail,
)
from app.storage.artifact_store import ArtifactStore, FRAMEWORK_EXTENSIONS, validate_identifier

# Matches auto-generated version names like "v1", "v2", "v10"
AUTO_VERSION_RE = re.compile(r"^v(\d+)$")


def _validate_json_schema(schema: dict, label: str) -> None:
    # Make sure the user's input_schema / output_schema is valid JSON Schema
    if not isinstance(schema, dict):
        raise AppError(code="INVALID_METADATA", message=f"{label} must be a JSON object", status_code=422)
    try:
        Draft7Validator.check_schema(schema)
    except SchemaError as exc:
        raise AppError(code="INVALID_METADATA", message=f"Invalid {label}: {exc.message}", status_code=422) from exc


def _next_version_string(db: Session, model_name: str) -> str:
    # Look at all existing vN versions for this model and pick the next number
    existing = db.scalars(select(ModelVersion.version).where(ModelVersion.model_name == model_name)).all()
    nums = [int(m.group(1)) for v in existing if (m := AUTO_VERSION_RE.match(v))]
    return f"v{max(nums, default=0) + 1}"


def _resolve_reference_artifact(artifact_url: str, framework: str) -> tuple[str, int, str]:
    allowed_exts = FRAMEWORK_EXTENSIONS[framework]
    # If the caller passed a remote URL, check its extension and store the URL directly
    if artifact_url.startswith(("http://", "https://", "s3://")):
        ext = Path(artifact_url.split("?", 1)[0]).suffix.lower()
        if ext and ext not in allowed_exts:
            raise AppError(
                code="UNSUPPORTED_FRAMEWORK",
                message=f"Reference URL extension '{ext}' does not match framework '{framework}'",
                status_code=400,
            )
        return artifact_url, 0, hashlib.sha256(artifact_url.encode("utf-8")).hexdigest()

    # Otherwise treat it as a local file path, verify it exists, and hash its contents
    local_path = Path(artifact_url).expanduser().resolve()
    if not local_path.is_file():
        raise AppError(
            code="MISSING_ARTIFACT",
            message=f"Referenced artifact path does not exist: '{artifact_url}'",
            status_code=400,
        )
    if local_path.suffix.lower() not in allowed_exts:
        raise AppError(
            code="UNSUPPORTED_FRAMEWORK",
            message=f"Referenced file extension '{local_path.suffix.lower()}' does not match framework '{framework}'",
            status_code=400,
        )
    raw = local_path.read_bytes()
    return str(local_path), len(raw), hashlib.sha256(raw).hexdigest()


class RegistryService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = ArtifactStore(settings.artifact_storage_dir)

    def _parse_metadata(self, metadata_raw: str) -> ModelMetadataInput:
        # Parse the form's JSON metadata string into a Python dict
        try:
            parsed = json.loads(metadata_raw)
        except json.JSONDecodeError as exc:
            raise AppError(code="INVALID_METADATA", message=f"metadata must be valid JSON: {exc.msg}", status_code=422) from exc

        # Validate metadata fields with Pydantic (and return 400 if framework is unsupported)
        try:
            meta = ModelMetadataInput.model_validate(parsed)
        except ValidationError as exc:
            if any(err.get("loc") == ("framework",) for err in exc.errors()):
                raise AppError(
                    code="UNSUPPORTED_FRAMEWORK",
                    message=f"Unsupported framework. Allowed: {sorted(FRAMEWORK_EXTENSIONS)}",
                    status_code=400,
                ) from exc
            raise AppError(
                code="INVALID_METADATA",
                message="Metadata failed schema validation",
                status_code=422,
                details={"errors": exc.errors()},
            ) from exc

        # Also validate that input_schema and output_schema are valid JSON Schemas
        _validate_json_schema(meta.input_schema, "input_schema")
        _validate_json_schema(meta.output_schema, "output_schema")
        return meta

    def register_version(
        self,
        db: Session,
        model_name: str,
        metadata_raw: str,
        artifact_filename: str | None,
        artifact_bytes: bytes | None,
    ) -> ModelUploadResponse:
        # Check model name and parse metadata before touching disk or the database
        validate_identifier(model_name, "model_name")
        meta = self._parse_metadata(metadata_raw)

        # Pick the version name (explicit or auto v1/v2/...) and reject duplicates
        version = meta.version or _next_version_string(db, model_name)
        validate_identifier(version, "version")
        duplicate = db.scalar(
            select(ModelVersion).where(ModelVersion.model_name == model_name, ModelVersion.version == version)
        )
        if duplicate is not None:
            raise AppError(
                code="VERSION_ALREADY_EXISTS",
                message=f"Model '{model_name}' version '{version}' already exists",
                status_code=409,
            )

        # Save the uploaded file to disk, or validate the artifact_url reference
        saved_managed_file = False
        if artifact_bytes is not None:
            location, size_bytes, sha256 = self.store.save_bytes(
                model_name=model_name,
                version=version,
                framework=meta.framework,
                original_filename=artifact_filename,
                data=artifact_bytes,
                max_bytes=self.settings.max_upload_size_bytes,
            )
            saved_managed_file = True
        elif meta.artifact_url:
            location, size_bytes, sha256 = _resolve_reference_artifact(meta.artifact_url, meta.framework)
        else:
            raise AppError(
                code="MISSING_ARTIFACT",
                message="Either an 'artifact' file upload or 'artifact_url' in metadata is required",
                status_code=400,
            )

        # Create/update the parent Model row and insert the new ModelVersion row
        try:
            model = db.get(Model, model_name)
            if model is None:
                db.add(Model(name=model_name, description=meta.description))
            else:
                model.updated_at = utcnow()
                if meta.description:
                    model.description = meta.description

            mv = ModelVersion(
                id=f"mv_{uuid.uuid4().hex[:12]}",
                model_name=model_name,
                version=version,
                artifact_location=location,
                artifact_size_bytes=size_bytes,
                artifact_sha256=sha256,
                framework=meta.framework,
                input_schema=meta.input_schema,
                output_schema=meta.output_schema,
                description=meta.description,
                status="registered",
            )
            db.add(mv)
            db.commit()
            db.refresh(mv)
        except Exception:
            # If saving to the DB fails, roll back and delete the file we just wrote
            db.rollback()
            if saved_managed_file:
                self.store.delete_artifact(location)
            raise

        return ModelUploadResponse(
            model=mv.model_name,
            version=mv.version,
            status=mv.status,
            artifact_size_bytes=mv.artifact_size_bytes,
            created_at=mv.created_at,
        )

    def list_models(self, db: Session) -> list[ModelSummary]:
        # Fetch every model and build a summary with version count and live versions
        models = db.scalars(select(Model).order_by(Model.created_at.asc())).all()
        summaries: list[ModelSummary] = []
        for m in models:
            versions = db.scalars(
                select(ModelVersion)
                .where(ModelVersion.model_name == m.name)
                .order_by(ModelVersion.created_at.asc())
            ).all()
            summaries.append(
                ModelSummary(
                    name=m.name,
                    description=m.description,
                    versions_count=len(versions),
                    latest_version=versions[-1].version if versions else None,
                    deployed_versions=[v.version for v in versions if v.status == "deployed"],
                    created_at=m.created_at,
                    updated_at=m.updated_at,
                )
            )
        return summaries

    def list_versions(self, db: Session, model_name: str) -> list[ModelVersionDetail]:
        # Make sure the model exists, then return all of its versions oldest-first
        require_model(db, model_name)
        versions = db.scalars(
            select(ModelVersion)
            .where(ModelVersion.model_name == model_name)
            .order_by(ModelVersion.created_at.asc())
        ).all()
        return [self._to_detail(v) for v in versions]

    def get_version(self, db: Session, model_name: str, version: str) -> ModelVersionDetail:
        return self._to_detail(require_model_version(db, model_name, version))

    def get_download_path(self, db: Session, model_name: str, version: str) -> tuple[Path, str]:
        # Find the file on disk and give it a friendly download filename like fraud_v1.pkl
        mv = require_model_version(db, model_name, version)
        path = self.store.resolve_download_path(mv.artifact_location)
        return path, f"{mv.model_name}_{mv.version}{path.suffix}"

    def delete_version(self, db: Session, model_name: str, version: str) -> DeleteVersionResponse:
        mv = require_model_version(db, model_name, version)

        # Refuse to delete a version if it is still running or deploying
        active_dep = db.scalar(
            select(Deployment).where(
                Deployment.model_version_id == mv.id,
                Deployment.status.in_(["deploying", "running"]),
            )
        )
        if active_dep is not None or mv.status == "deployed":
            raise AppError(
                code="VERSION_DEPLOYED",
                message=f"Cannot delete model '{model_name}' version '{version}' while deployed; undeploy first",
                status_code=409,
            )

        # Remove the DB record first, then delete the artifact file from disk
        artifact_location = mv.artifact_location
        db.delete(mv)
        db.commit()
        self.store.delete_artifact(artifact_location)
        return DeleteVersionResponse(model=model_name, version=version, status="deleted")

    @staticmethod
    def _to_detail(mv: ModelVersion) -> ModelVersionDetail:
        return ModelVersionDetail(
            model=mv.model_name,
            version=mv.version,
            framework=mv.framework,
            status=mv.status,
            artifact_location=mv.artifact_location,
            artifact_size_bytes=mv.artifact_size_bytes,
            artifact_sha256=mv.artifact_sha256,
            input_schema=mv.input_schema,
            output_schema=mv.output_schema,
            description=mv.description,
            created_at=mv.created_at,
        )
