from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.registry import (
    DeleteVersionResponse,
    ModelSummary,
    ModelUploadResponse,
    ModelVersionDetail,
)
from app.services.registry_service import RegistryService

router = APIRouter(prefix="/models", tags=["registry"])


def _get_registry_service(request: Request) -> RegistryService:
    # Pull the shared RegistryService instance from app.state
    return request.app.state.registry_service


@router.post("/{model_name}/versions", response_model=ModelUploadResponse, status_code=201)
async def upload_model_version(
    model_name: str,
    metadata: str = Form(...),
    artifact: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
    service: RegistryService = Depends(_get_registry_service),
) -> ModelUploadResponse:
    # Read the uploaded file bytes (if provided) and register the new model version
    artifact_bytes = await artifact.read() if artifact is not None else None
    artifact_filename = artifact.filename if artifact is not None else None
    return service.register_version(
        db=db,
        model_name=model_name,
        metadata_raw=metadata,
        artifact_filename=artifact_filename,
        artifact_bytes=artifact_bytes,
    )


@router.get("", response_model=list[ModelSummary])
def list_models(
    db: Session = Depends(get_db),
    service: RegistryService = Depends(_get_registry_service),
) -> list[ModelSummary]:
    # Return a summary of every registered model
    return service.list_models(db)


@router.get("/{model_name}/versions", response_model=list[ModelVersionDetail])
def list_model_versions(
    model_name: str,
    db: Session = Depends(get_db),
    service: RegistryService = Depends(_get_registry_service),
) -> list[ModelVersionDetail]:
    # Return all versions registered under this model name
    return service.list_versions(db, model_name)


@router.get("/{model_name}/versions/{version}", response_model=ModelVersionDetail)
def get_model_version(
    model_name: str,
    version: str,
    db: Session = Depends(get_db),
    service: RegistryService = Depends(_get_registry_service),
) -> ModelVersionDetail:
    # Return full metadata and schemas for one specific model version
    return service.get_version(db, model_name, version)


@router.get("/{model_name}/versions/{version}/download")
def download_model_artifact(
    model_name: str,
    version: str,
    db: Session = Depends(get_db),
    service: RegistryService = Depends(_get_registry_service),
) -> FileResponse:
    # Stream the saved model file back to the caller as a binary download
    path, filename = service.get_download_path(db, model_name, version)
    return FileResponse(path=path, filename=filename, media_type="application/octet-stream")


@router.delete("/{model_name}/versions/{version}", response_model=DeleteVersionResponse)
def delete_model_version(
    model_name: str,
    version: str,
    db: Session = Depends(get_db),
    service: RegistryService = Depends(_get_registry_service),
) -> DeleteVersionResponse:
    # Delete a model version and its file (blocked if it is still deployed)
    return service.delete_version(db, model_name, version)
