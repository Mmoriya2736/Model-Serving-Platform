from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
import uuid

from fastapi import FastAPI, Request, Response

from app.api import deployments as deployments_api
from app.api import inference as inference_api
from app.api import metrics as metrics_api
from app.api import registry as registry_api
from app.api import routing as routing_api
from app.config import Settings
from app.db.database import create_db_engine, create_session_factory, init_db
from app.errors import register_exception_handlers
from app.runtime.manager import ServingManager
from app.services.deployment_service import DeploymentService
from app.services.inference_service import InferenceService
from app.services.metrics_service import MetricsService
from app.services.registry_service import RegistryService
from app.services.routing_service import RoutingService

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    # Set up standard Python logging so logs have timestamps and levels
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    # Load default settings unless tests pass in custom temporary settings
    cfg = settings or Settings()
    configure_logging(cfg.log_level)

    # Runs once when the server starts and cleans everything up when the server stops
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Make sure the folder for uploaded model files exists and create DB tables
        cfg.artifact_storage_dir.mkdir(parents=True, exist_ok=True)
        engine = create_db_engine(cfg.database_url)
        init_db(engine)
        session_factory = create_session_factory(engine)

        # Create the core services that handle models, routing, inference, and metrics
        serving_manager = ServingManager(cfg)
        routing_service = RoutingService()
        inference_service = InferenceService(cfg, serving_manager, routing_service, session_factory)

        # Attach everything to app.state so all API routes can access the same instances
        app.state.settings = cfg
        app.state.engine = engine
        app.state.session_factory = session_factory
        app.state.serving_manager = serving_manager
        app.state.routing_service = routing_service
        app.state.inference_service = inference_service
        app.state.metrics_service = MetricsService(inference_service.drift_tracker)
        app.state.registry_service = RegistryService(cfg)
        app.state.deployment_service = DeploymentService(cfg, serving_manager)

        # Reload any models that were already running before the server restarted
        with session_factory() as db:
            app.state.deployment_service.warm_reload_running_deployments(db)

        logger.info("Model Serving Platform initialized (db=%s)", cfg.database_url)
        try:
            yield
        finally:
            # Shut down background threads, stop all model worker processes, and close DB
            inference_service.shutdown()
            serving_manager.shutdown_all()
            engine.dispose()
            logger.info("Model Serving Platform shut down cleanly")

    # Build the FastAPI app and plug in our custom JSON error handlers
    app = FastAPI(title=cfg.app_name, version=cfg.app_version, lifespan=lifespan)
    register_exception_handlers(app)

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next) -> Response:
        # Give every request a short ID (or keep the caller's X-Request-ID) for easy tracing
        req_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
        request.state.request_id = req_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response

    # Register all API endpoint groups
    app.include_router(registry_api.router)
    app.include_router(deployments_api.router)
    app.include_router(inference_api.router)
    app.include_router(routing_api.router)
    app.include_router(metrics_api.router)

    # Simple health endpoint so Docker and users can check if the server is alive
    @app.get("/health", tags=["health"])
    async def health(request: Request) -> dict[str, str]:
        return {
            "status": "healthy",
            "service": cfg.app_name,
            "version": cfg.app_version,
            "request_id": request.state.request_id,
        }

    return app


# Default app instance used by Uvicorn when starting the server
app = create_app()
