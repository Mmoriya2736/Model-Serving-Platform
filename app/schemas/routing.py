from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# One version + percentage weight pair (0 to 100)
class RoutingRule(BaseModel):
    version: str
    weight: int = Field(..., ge=0, le=100)


# Request body for PUT /models/{model_name}/routing
class RoutingUpdateRequest(BaseModel):
    rules: list[RoutingRule]


# Current routing rules, previous snapshot for rollback, and active canary info
class RoutingResponse(BaseModel):
    model: str
    rules: list[RoutingRule]
    previous_rules: list[RoutingRule] | None = None
    canary: dict[str, Any] | None = None
    updated_at: datetime


# Request body for POST /models/{model_name}/rollout
class RolloutRequest(BaseModel):
    target_version: str
    weight: int = Field(..., ge=1, le=100)
    max_error_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    min_requests: int = Field(default=5, ge=1)


# Optional request body for POST /models/{model_name}/rollback
class RollbackRequest(BaseModel):
    target_version: str | None = None
