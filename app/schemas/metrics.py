from typing import Any

from pydantic import BaseModel


# Latency percentiles in milliseconds
class LatencyMetrics(BaseModel):
    p50_ms: float
    p95_ms: float
    p99_ms: float


# Per-version traffic, error, and median latency summary
class VersionMetrics(BaseModel):
    requests: int
    errors: int
    p50_ms: float


# Summary stats and 5 histogram buckets for classifier confidence scores
class ConfidenceDistribution(BaseModel):
    count: int
    mean: float
    min: float
    max: float
    buckets: dict[str, int]


# Full response returned by GET /models/{model_name}/metrics
class ModelMetricsResponse(BaseModel):
    model: str
    period: str
    requests_total: int
    errors_total: int
    error_rate: float
    throughput_rps: float
    latency: LatencyMetrics
    by_version: dict[str, VersionMetrics]
    confidence_distribution: ConfidenceDistribution | None = None
    drift: dict[str, Any] | None = None
