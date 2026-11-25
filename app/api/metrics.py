from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.metrics import ModelMetricsResponse
from app.services.metrics_service import MetricsService

router = APIRouter(prefix="/models", tags=["metrics"])


def _get_metrics_service(request: Request) -> MetricsService:
    # Pull the shared MetricsService from app.state
    return request.app.state.metrics_service


@router.get("/{model_name}/metrics", response_model=ModelMetricsResponse)
def get_model_metrics(
    model_name: str,
    period: str = Query(default="1h"),
    db: Session = Depends(get_db),
    service: MetricsService = Depends(_get_metrics_service),
) -> ModelMetricsResponse:
    # Return latency, throughput, error rate, confidence buckets, and drift for the given window
    return service.get_model_metrics(db, model_name, period=period)
