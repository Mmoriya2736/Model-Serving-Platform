import asyncio

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.deployment import (
    DeployRequest,
    DeployResponse,
    DeploymentDetail,
    DeploymentHealthResponse,
    ScaleRequest,
)
from app.services.deployment_service import DeploymentService

router = APIRouter(tags=["deployments"])


def _get_deployment_service(request: Request) -> DeploymentService:
    # Pull the shared DeploymentService from app.state
    return request.app.state.deployment_service


@router.post("/models/{model_name}/versions/{version}/deploy", response_model=DeployResponse)
async def deploy_model_version(
    model_name: str,
    version: str,
    req: DeployRequest = DeployRequest(),
    db: Session = Depends(get_db),
    service: DeploymentService = Depends(_get_deployment_service),
) -> DeployResponse:
    # Run worker startup in a background thread so the FastAPI event loop stays responsive
    return await asyncio.to_thread(service.deploy_version, db, model_name, version, req)


@router.delete("/models/{model_name}/versions/{version}/deploy", response_model=DeploymentDetail)
@router.post("/models/{model_name}/versions/{version}/undeploy", response_model=DeploymentDetail)
async def undeploy_model_version(
    model_name: str,
    version: str,
    db: Session = Depends(get_db),
    service: DeploymentService = Depends(_get_deployment_service),
) -> DeploymentDetail:
    # Stop the model's worker processes and mark the deployment as stopped
    return await asyncio.to_thread(service.undeploy_version, db, model_name, version)


@router.get("/models/{model_name}/versions/{version}/health", response_model=DeploymentHealthResponse)
async def model_version_health(
    model_name: str,
    version: str,
    db: Session = Depends(get_db),
    service: DeploymentService = Depends(_get_deployment_service),
) -> DeploymentHealthResponse:
    # Ping the deployed workers and return replica health and current queue depth
    return await asyncio.to_thread(service.get_version_health, db, model_name, version)


@router.get("/deployments", response_model=list[DeploymentDetail])
def list_deployments(
    model_name: str | None = Query(default=None),
    status: str | None = Query(default=None),
    db: Session = Depends(get_db),
    service: DeploymentService = Depends(_get_deployment_service),
) -> list[DeploymentDetail]:
    # List deployments, optionally filtered by model name or status (e.g. "running")
    return service.list_deployments(db, model_name=model_name, status=status)


@router.get("/deployments/{deployment_id}", response_model=DeploymentDetail)
def get_deployment(
    deployment_id: str,
    db: Session = Depends(get_db),
    service: DeploymentService = Depends(_get_deployment_service),
) -> DeploymentDetail:
    # Return details and live health for one specific deployment ID
    return service.get_deployment(db, deployment_id)


@router.patch("/deployments/{deployment_id}", response_model=DeploymentDetail)
async def scale_deployment(
    deployment_id: str,
    req: ScaleRequest,
    db: Session = Depends(get_db),
    service: DeploymentService = Depends(_get_deployment_service),
) -> DeploymentDetail:
    # Add or remove worker replicas (and optionally update memory/CPU settings)
    return await asyncio.to_thread(service.scale_deployment, db, deployment_id, req)
