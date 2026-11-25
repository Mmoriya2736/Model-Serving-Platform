from collections import defaultdict
from datetime import timedelta
import re

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import InferenceLog, utcnow
from app.db.queries import require_model
from app.errors import AppError
from app.schemas.metrics import (
    ConfidenceDistribution,
    LatencyMetrics,
    ModelMetricsResponse,
    VersionMetrics,
)
from app.services.inference_service import DriftTracker

# Matches time windows like "30s", "5m", "1h", "7d"
PERIOD_RE = re.compile(r"^(\d+)([smhd])$")
UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}

# Five histogram buckets used to group classifier confidence scores from 0.0 to 1.0
CONFIDENCE_BUCKETS = (
    (0.2, "0.0-0.2"),
    (0.4, "0.2-0.4"),
    (0.6, "0.4-0.6"),
    (0.8, "0.6-0.8"),
    (1.1, "0.8-1.0"),
)


def parse_period_seconds(period: str) -> int | None:
    # Return None for "all", or convert strings like "1h" into seconds
    if period == "all":
        return None
    match = PERIOD_RE.match(period.strip())
    if not match or int(match.group(1)) <= 0:
        raise AppError(
            code="INVALID_PERIOD",
            message=f"Invalid period '{period}'. Expected e.g. '5m', '1h', '24h', '7d', or 'all'.",
            status_code=400,
        )
    return int(match.group(1)) * UNIT_SECONDS[match.group(2)]


def _percentile(values: list[float], q: float) -> float:
    # Safely return 0.0 when there are no requests yet
    return round(float(np.percentile(values, q)), 3) if values else 0.0


def _build_confidence_distribution(confidences: list[float]) -> ConfidenceDistribution | None:
    # Skip confidence stats for models (like regressors) that don't output a confidence score
    if not confidences:
        return None
    buckets = {label: 0 for _, label in CONFIDENCE_BUCKETS}
    for c in confidences:
        for upper, label in CONFIDENCE_BUCKETS:
            if c < upper:
                buckets[label] += 1
                break
    return ConfidenceDistribution(
        count=len(confidences),
        mean=round(sum(confidences) / len(confidences), 4),
        min=round(min(confidences), 4),
        max=round(max(confidences), 4),
        buckets=buckets,
    )


class MetricsService:
    def __init__(self, drift_tracker: DriftTracker) -> None:
        self.drift_tracker = drift_tracker

    def get_model_metrics(self, db: Session, model_name: str, period: str = "1h") -> ModelMetricsResponse:
        # Verify the model exists and parse the requested time window
        require_model(db, model_name)
        window_seconds = parse_period_seconds(period)

        # Fetch all inference logs for this model inside the time window
        stmt = select(InferenceLog).where(InferenceLog.model_name == model_name)
        if window_seconds is not None:
            stmt = stmt.where(InferenceLog.timestamp >= utcnow() - timedelta(seconds=window_seconds))
        logs = db.scalars(stmt.order_by(InferenceLog.timestamp.asc())).all()

        # Pull out totals, latencies, and confidence scores
        total = len(logs)
        errors = sum(1 for row in logs if row.error is not None)
        latencies = [float(row.latency_ms) for row in logs]
        confidences = [float(row.confidence) for row in logs if row.confidence is not None]

        # Pick the time span used to compute requests-per-second (throughput)
        if window_seconds is not None:
            duration_s = float(window_seconds)
        elif total >= 2:
            duration_s = max(1.0, (logs[-1].timestamp - logs[0].timestamp).total_seconds())
        else:
            duration_s = 1.0

        # Group logs by version so we can compare v1 vs v2 side by side
        grouped: dict[str, list[InferenceLog]] = defaultdict(list)
        for row in logs:
            grouped[row.version].append(row)
        by_version = {
            ver: VersionMetrics(
                requests=len(v_logs),
                errors=sum(1 for r in v_logs if r.error is not None),
                p50_ms=_percentile([float(r.latency_ms) for r in v_logs], 50),
            )
            for ver, v_logs in sorted(grouped.items())
        }

        return ModelMetricsResponse(
            model=model_name,
            period=period,
            requests_total=total,
            errors_total=errors,
            error_rate=round(errors / total, 4) if total > 0 else 0.0,
            throughput_rps=round(total / duration_s, 4) if total > 0 else 0.0,
            latency=LatencyMetrics(
                p50_ms=_percentile(latencies, 50),
                p95_ms=_percentile(latencies, 95),
                p99_ms=_percentile(latencies, 99),
            ),
            by_version=by_version,
            confidence_distribution=_build_confidence_distribution(confidences),
            drift=self.drift_tracker.summarize(model_name),
        )
