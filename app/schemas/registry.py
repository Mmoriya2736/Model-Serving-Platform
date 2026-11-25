from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# JSON metadata sent inside the multipart form when uploading a model version
class ModelMetadataInput(BaseModel):
    framework: Literal["sklearn", "onnx", "custom"]
    version: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})
    output_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})
    description: str | None = None
    artifact_url: str | None = None


# Response returned by POST /models/{model_name}/versions
class ModelUploadResponse(BaseModel):
    model: str
    version: str
    status: str
    artifact_size_bytes: int
    created_at: datetime


# Full details returned when inspecting one or more versions of a model
class ModelVersionDetail(BaseModel):
    model: str
    version: str
    framework: str
    status: str
    artifact_location: str
    artifact_size_bytes: int
    artifact_sha256: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    description: str | None = None
    created_at: datetime


# High-level summary returned for each model in GET /models
class ModelSummary(BaseModel):
    name: str
    description: str | None = None
    versions_count: int
    latest_version: str | None = None
    deployed_versions: list[str]
    created_at: datetime
    updated_at: datetime


# Confirmation returned after deleting an undeployed model version
class DeleteVersionResponse(BaseModel):
    model: str
    version: str
    status: str = "deleted"
