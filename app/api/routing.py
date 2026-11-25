from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.routing import (
    RollbackRequest,
    RolloutRequest,
    RoutingResponse,
    RoutingUpdateRequest,
)
from app.services.routing_service import RoutingService

router = APIRouter(prefix="/models", tags=["routing"])


def _get_routing_service(request: Request) -> RoutingService:
    # Pull the shared RoutingService from app.state
    return request.app.state.routing_service


@router.put("/{model_name}/routing", response_model=RoutingResponse)
def configure_routing(
    model_name: str,
    req: RoutingUpdateRequest,
    db: Session = Depends(get_db),
    service: RoutingService = Depends(_get_routing_service),
) -> RoutingResponse:
    # Set exact percentage weights for splitting traffic across running versions
    return service.update_routing(db, model_name, req)


@router.get("/{model_name}/routing", response_model=RoutingResponse)
def get_routing(
    model_name: str,
    db: Session = Depends(get_db),
    service: RoutingService = Depends(_get_routing_service),
) -> RoutingResponse:
    # Return the active routing weights, last saved rules, and any active canary config
    return service.get_routing(db, model_name)


@router.post("/{model_name}/rollout", response_model=RoutingResponse)
def canary_rollout(
    model_name: str,
    req: RolloutRequest,
    db: Session = Depends(get_db),
    service: RoutingService = Depends(_get_routing_service),
) -> RoutingResponse:
    # Gradually send a percentage of traffic to a new version with automatic error rollback
    return service.start_or_advance_rollout(db, model_name, req)


@router.post("/{model_name}/rollback", response_model=RoutingResponse)
def rollback_routing(
    model_name: str,
    req: RollbackRequest | None = None,
    db: Session = Depends(get_db),
    service: RoutingService = Depends(_get_routing_service),
) -> RoutingResponse:
    # Immediately revert traffic back to the previous routing rules (or a chosen version)
    return service.rollback(db, model_name, req)
