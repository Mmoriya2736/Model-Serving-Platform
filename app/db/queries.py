from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Model, ModelVersion
from app.errors import AppError


def require_model(db: Session, model_name: str) -> Model:
    # Look up the model by name, or stop right away with a 404 error
    model = db.get(Model, model_name)
    if model is None:
        raise AppError(code="NOT_FOUND", message=f"Model '{model_name}' not found", status_code=404)
    return model


def require_model_version(db: Session, model_name: str, version: str) -> ModelVersion:
    # Look up a specific model + version pair, or stop with a 404 error
    mv = db.scalar(
        select(ModelVersion).where(
            ModelVersion.model_name == model_name,
            ModelVersion.version == version,
        )
    )
    if mv is None:
        raise AppError(
            code="NOT_FOUND",
            message=f"Model '{model_name}' version '{version}' not found",
            status_code=404,
        )
    return mv
