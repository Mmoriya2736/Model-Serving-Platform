from datetime import datetime
from typing import Any

from pydantic import BaseModel


# Request body for single or async predictions (version is optional; uses routing if omitted)
class PredictRequest(BaseModel):
    inputs: dict[str, Any]
    version: str | None = None


# Standard response returned by POST /predict/{model_name}
class PredictResponse(BaseModel):
    model: str
    version: str
    outputs: dict[str, Any]
    latency_ms: float
    request_id: str


# Request body for POST /predict/{model_name}/batch
class BatchPredictRequest(BaseModel):
    inputs: list[dict[str, Any]]
    version: str | None = None


# One item inside a batch prediction response
class BatchPredictItem(BaseModel):
    outputs: dict[str, Any]
    latency_ms: float


# Response returned by POST /predict/{model_name}/batch
class BatchPredictResponse(BaseModel):
    model: str
    version: str
    results: list[BatchPredictItem]
    total_latency_ms: float
    request_id: str


# Immediate 202 response returned when an async prediction job is queued
class AsyncPredictSubmitResponse(BaseModel):
    job_id: str
    model: str
    version: str | None = None
    status: str


# Status and final result returned by GET /jobs/{job_id}
class AsyncJobResponse(BaseModel):
    job_id: str
    model: str
    version: str | None = None
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
