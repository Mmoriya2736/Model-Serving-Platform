import asyncio

from fastapi import APIRouter, Depends, Header, Request
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.inference import (
    AsyncJobResponse,
    AsyncPredictSubmitResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)
from app.services.inference_service import InferenceService

router = APIRouter(tags=["inference"])


def _get_inference_service(request: Request) -> InferenceService:
    # Pull the shared InferenceService from app.state
    return request.app.state.inference_service


@router.post("/predict/{model_name}", response_model=PredictResponse)
async def predict(
    model_name: str,
    req: PredictRequest,
    request: Request,
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
    db: Session = Depends(get_db),
    service: InferenceService = Depends(_get_inference_service),
) -> PredictResponse:
    # Run single prediction off the event loop so slow models never block /health or other models
    return await asyncio.to_thread(
        service.predict, db, model_name, req, request.state.request_id, x_session_id
    )


@router.post("/predict/{model_name}/batch", response_model=BatchPredictResponse)
async def predict_batch(
    model_name: str,
    req: BatchPredictRequest,
    request: Request,
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
    db: Session = Depends(get_db),
    service: InferenceService = Depends(_get_inference_service),
) -> BatchPredictResponse:
    # Run a batch of inputs in one worker call and return per-item and total latency
    return await asyncio.to_thread(
        service.predict_batch, db, model_name, req, request.state.request_id, x_session_id
    )


@router.post("/predict/{model_name}/async", response_model=AsyncPredictSubmitResponse, status_code=202)
async def predict_async(
    model_name: str,
    req: PredictRequest,
    request: Request,
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
    db: Session = Depends(get_db),
    service: InferenceService = Depends(_get_inference_service),
) -> AsyncPredictSubmitResponse:
    # Queue a slow prediction in the background and return a job_id right away (202 Accepted)
    return await asyncio.to_thread(
        service.submit_async_predict, db, model_name, req, request.state.request_id, x_session_id
    )


@router.get("/jobs/{job_id}", response_model=AsyncJobResponse)
def get_async_job(
    job_id: str,
    db: Session = Depends(get_db),
    service: InferenceService = Depends(_get_inference_service),
) -> AsyncJobResponse:
    # Check the status and result of a previously submitted async prediction job
    return service.get_job(db, job_id)
