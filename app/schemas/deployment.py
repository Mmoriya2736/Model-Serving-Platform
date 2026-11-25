from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# Memory and CPU limits requested for a model deployment
class ResourceConfig(BaseModel):
    memory: str = "512Mi"
    cpu: str = "0.5"


# Request body for POST /models/{model_name}/versions/{version}/deploy
class DeployRequest(BaseModel):
    replicas: int = Field(default=1, ge=1, le=16)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)


# Response body returned after deploying a model version
class DeployResponse(BaseModel):
    deployment_id: str
    status: str
    endpoint: str


# Request body for PATCH /deployments/{deployment_id} to change replica count or resources
class ScaleRequest(BaseModel):
    replicas: int = Field(..., ge=1, le=16)
    resources: ResourceConfig | None = None


# Full deployment state plus live worker health returned by GET /deployments
class DeploymentDetail(BaseModel):
    deployment_id: str
    model: str
    version: str
    replicas: int
    resources: dict[str, Any]
    status: str
    endpoint: str
    error_message: str | None = None
    health: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


# Live health summary returned by GET /models/{model_name}/versions/{version}/health
class DeploymentHealthResponse(BaseModel):
    model: str
    version: str
    status: str
    healthy: bool
    replicas_total: int
    replicas_healthy: int
    queue_depth: int
