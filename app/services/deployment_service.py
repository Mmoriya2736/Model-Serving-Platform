import logging
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import Deployment, ModelVersion, RoutingConfig, utcnow
from app.db.queries import require_model_version
from app.errors import AppError
from app.runtime.manager import ServingManager, parse_cpu_cores, parse_memory_bytes
from app.schemas.deployment import (
    DeployRequest,
    DeployResponse,
    DeploymentDetail,
    DeploymentHealthResponse,
    ScaleRequest,
)
from app.services.routing_service import normalize_weights

logger = logging.getLogger(__name__)


class DeploymentService:
    def __init__(self, settings: Settings, serving_manager: ServingManager) -> None:
        self.settings = settings
        self.serving_manager = serving_manager

    def deploy_version(
        self,
        db: Session,
        model_name: str,
        version: str,
        req: DeployRequest,
    ) -> DeployResponse:
        # Look up the model version and check if custom Python models are allowed
        mv = require_model_version(db, model_name, version)
        if mv.framework == "custom" and not self.settings.enable_custom_python_models:
            raise AppError(
                code="CUSTOM_MODELS_DISABLED",
                message="Custom Python models are disabled by server configuration",
                status_code=403,
            )

        # Validate the memory ("512Mi") and CPU ("0.5") settings before starting workers
        memory_bytes = parse_memory_bytes(req.resources.memory)
        parse_cpu_cores(req.resources.cpu)

        # Reject if this exact model version is already running or deploying
        existing_active = db.scalar(
            select(Deployment).where(
                Deployment.model_version_id == mv.id,
                Deployment.status.in_(["deploying", "running"]),
            )
        )
        if existing_active is not None:
            raise AppError(
                code="ALREADY_DEPLOYED",
                message=f"Model '{model_name}' version '{version}' is already deployed ({existing_active.deployment_id})",
                status_code=409,
            )

        # Save a new Deployment row in "deploying" state so we can track failures
        dep = Deployment(
            deployment_id=f"dep_{uuid.uuid4().hex[:12]}",
            model_version_id=mv.id,
            model_name=model_name,
            version=version,
            replicas=req.replicas,
            resources=req.resources.model_dump(),
            status="deploying",
            endpoint=f"/predict/{model_name}",
        )
        db.add(dep)
        db.commit()

        # Spin up the worker processes and run a health check; mark "failed" if loading breaks
        try:
            self.serving_manager.deploy(
                model_name=model_name,
                version=version,
                framework=mv.framework,
                artifact_path=mv.artifact_location,
                replicas=req.replicas,
                memory_bytes=memory_bytes,
            )
        except AppError as exc:
            dep.status = "failed"
            dep.error_message = exc.message
            dep.updated_at = utcnow()
            db.commit()
            raise

        # Workers are healthy: mark deployment "running" and set up default routing if needed
        dep.status = "running"
        dep.updated_at = utcnow()
        mv.status = "deployed"
        self._ensure_initial_routing(db, model_name, version)
        db.commit()

        return DeployResponse(deployment_id=dep.deployment_id, status=dep.status, endpoint=dep.endpoint)

    def _ensure_initial_routing(self, db: Session, model_name: str, version: str) -> None:
        # Send 100% of traffic to the first version; keep existing traffic unchanged for v2+ (dark launch)
        cfg = db.get(RoutingConfig, model_name)
        if cfg is None:
            db.add(RoutingConfig(model_name=model_name, rules=[{"version": version, "weight": 100}], previous_rules=None))
        elif not cfg.rules:
            cfg.rules = [{"version": version, "weight": 100}]
            cfg.updated_at = utcnow()

    def undeploy_version(self, db: Session, model_name: str, version: str) -> DeploymentDetail:
        # Find the active deployment for this model version or return 404
        dep = db.scalar(
            select(Deployment).where(
                Deployment.model_name == model_name,
                Deployment.version == version,
                Deployment.status.in_(["deploying", "running"]),
            )
        )
        if dep is None:
            raise AppError(
                code="NOT_FOUND",
                message=f"No active deployment found for '{model_name}' version '{version}'",
                status_code=404,
            )

        # Stop all worker processes and update DB statuses back to stopped/registered
        self.serving_manager.undeploy(model_name, version)
        dep.status = "stopped"
        dep.updated_at = utcnow()

        mv = db.get(ModelVersion, dep.model_version_id)
        if mv is not None:
            mv.status = "registered"

        # Remove this version from active routing rules so no requests try to hit it
        self._remove_version_from_routing(db, model_name, version)
        db.commit()
        db.refresh(dep)
        return self._to_detail(dep)

    def _remove_version_from_routing(self, db: Session, model_name: str, version: str) -> None:
        # Drop the stopped version from routing and rescale the remaining versions back to 100%
        cfg = db.get(RoutingConfig, model_name)
        if cfg is None or not cfg.rules:
            return
        remaining = [r for r in cfg.rules if r["version"] != version]
        if len(remaining) == len(cfg.rules):
            return
        cfg.previous_rules = list(cfg.rules)
        cfg.rules = normalize_weights(remaining, target_total=100)
        cfg.canary_config = None
        cfg.updated_at = utcnow()

    def list_deployments(
        self,
        db: Session,
        model_name: str | None = None,
        status: str | None = None,
    ) -> list[DeploymentDetail]:
        # Return deployments newest-first, applying optional model_name and status filters
        stmt = select(Deployment).order_by(Deployment.created_at.desc())
        if model_name:
            stmt = stmt.where(Deployment.model_name == model_name)
        if status:
            stmt = stmt.where(Deployment.status == status)
        return [self._to_detail(d) for d in db.scalars(stmt).all()]

    def get_deployment(self, db: Session, deployment_id: str) -> DeploymentDetail:
        dep = db.get(Deployment, deployment_id)
        if dep is None:
            raise AppError(code="NOT_FOUND", message=f"Deployment '{deployment_id}' not found", status_code=404)
        return self._to_detail(dep)

    def scale_deployment(self, db: Session, deployment_id: str, req: ScaleRequest) -> DeploymentDetail:
        # Only running deployments can be scaled
        dep = db.get(Deployment, deployment_id)
        if dep is None or dep.status != "running":
            raise AppError(code="NOT_FOUND", message=f"Running deployment '{deployment_id}' not found", status_code=404)

        # Validate new memory/CPU settings if the caller provided them
        memory_bytes = None
        if req.resources is not None:
            memory_bytes = parse_memory_bytes(req.resources.memory)
            parse_cpu_cores(req.resources.cpu)
            dep.resources = req.resources.model_dump()

        # Add or remove worker processes live and save the updated replica count
        self.serving_manager.scale(dep.model_name, dep.version, replicas=req.replicas, memory_bytes=memory_bytes)
        dep.replicas = req.replicas
        dep.updated_at = utcnow()
        db.commit()
        db.refresh(dep)
        return self._to_detail(dep)

    def get_version_health(self, db: Session, model_name: str, version: str) -> DeploymentHealthResponse:
        # Look up the latest deployment record and ping its live worker pool
        dep = db.scalar(
            select(Deployment)
            .where(Deployment.model_name == model_name, Deployment.version == version)
            .order_by(Deployment.created_at.desc())
        )
        if dep is None:
            raise AppError(
                code="NOT_FOUND",
                message=f"No deployment found for '{model_name}' version '{version}'",
                status_code=404,
            )
        summary = self.serving_manager.health(model_name, version)
        return DeploymentHealthResponse(
            model=model_name,
            version=version,
            status=dep.status,
            healthy=bool(summary["healthy"]) and dep.status == "running",
            replicas_total=summary["replicas_total"],
            replicas_healthy=summary["replicas_healthy"],
            queue_depth=summary["queue_depth"],
        )

    def warm_reload_running_deployments(self, db: Session) -> None:
        # On server startup, re-launch worker processes for every model marked "running" in SQLite
        running = db.scalars(select(Deployment).where(Deployment.status == "running")).all()
        for dep in running:
            mv = db.get(ModelVersion, dep.model_version_id)
            if mv is None:
                continue
            try:
                mem = parse_memory_bytes(dep.resources.get("memory", "512Mi"))
                self.serving_manager.deploy(
                    model_name=dep.model_name,
                    version=dep.version,
                    framework=mv.framework,
                    artifact_path=mv.artifact_location,
                    replicas=dep.replicas,
                    memory_bytes=mem,
                )
            except Exception as exc:
                # If a model file went missing or broke, mark it failed without stopping startup
                logger.error("Failed to warm-reload %s:%s: %s", dep.model_name, dep.version, exc)
                dep.status = "failed"
                dep.error_message = f"Warm reload failed: {exc}"
                mv.status = "registered"
        db.commit()

    def _to_detail(self, dep: Deployment) -> DeploymentDetail:
        health = self.serving_manager.health(dep.model_name, dep.version) if dep.status == "running" else None
        return DeploymentDetail(
            deployment_id=dep.deployment_id,
            model=dep.model_name,
            version=dep.version,
            replicas=dep.replicas,
            resources=dep.resources,
            status=dep.status,
            endpoint=dep.endpoint,
            error_message=dep.error_message,
            health=health,
            created_at=dep.created_at,
            updated_at=dep.updated_at,
        )
