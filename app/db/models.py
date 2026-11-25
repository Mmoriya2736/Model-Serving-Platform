from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def utcnow() -> datetime:
    # Always store timestamps in UTC so time windows and metrics stay consistent
    return datetime.now(timezone.utc)


# Top-level model record (e.g. "fraud-detector"), which can have many versions
class Model(Base):
    __tablename__ = "models"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )

    # Deleting a model also cleans up its versions and routing rules
    versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="model", cascade="all, delete-orphan"
    )
    routing_config: Mapped["RoutingConfig | None"] = relationship(
        back_populates="model", uselist=False, cascade="all, delete-orphan"
    )


# One uploaded version of a model (e.g. "v1" or "v2") with its file location and schemas
class ModelVersion(Base):
    __tablename__ = "model_versions"
    # Prevent two identical version names under the same model
    __table_args__ = (UniqueConstraint("model_name", "version", name="uq_model_version"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_name: Mapped[str] = mapped_column(
        String(64), ForeignKey("models.name", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    # Path on disk or remote URL where the model file lives
    artifact_location: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    framework: Mapped[str] = mapped_column(String(32), nullable=False)
    # JSON schemas used to validate inputs and describe expected outputs
    input_schema: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    output_schema: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="registered", nullable=False)
    # Optional summary stats for feature drift tracking
    baseline_stats: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    model: Mapped["Model"] = relationship(back_populates="versions")
    deployments: Mapped[list["Deployment"]] = relationship(
        back_populates="model_version", cascade="all, delete-orphan"
    )


# Tracks a live or past deployment of a model version, its replica count, and resources
class Deployment(Base):
    __tablename__ = "deployments"
    __table_args__ = (Index("ix_deployments_model_version", "model_name", "version"),)

    deployment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_version_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    replicas: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    resources: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    # Lifecycle states: deploying -> running -> stopped (or failed)
    status: Mapped[str] = mapped_column(String(32), default="deploying", nullable=False, index=True)
    endpoint: Mapped[str] = mapped_column(String(128), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )

    model_version: Mapped["ModelVersion"] = relationship(back_populates="deployments")


# Stores how traffic is split between versions, plus the last config for instant rollback
class RoutingConfig(Base):
    __tablename__ = "routing_configs"

    model_name: Mapped[str] = mapped_column(
        String(64), ForeignKey("models.name", ondelete="CASCADE"), primary_key=True
    )
    rules: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    previous_rules: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    # Active canary settings (target version, max error rate, min requests) if a rollout is running
    canary_config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )

    model: Mapped["Model"] = relationship(back_populates="routing_config")


# One log row per prediction so /metrics can compute latency, throughput, errors, and confidence
class InferenceLog(Base):
    __tablename__ = "inference_logs"
    # Composite index speeds up time-window queries like "last 1h for model X"
    __table_args__ = (Index("ix_inference_logs_model_ts", "model_name", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # Store only a SHA-256 hash of the input for debugging without saving sensitive user data
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


# Tracks background prediction jobs started by POST /predict/{model_name}/async
class AsyncJob(Base):
    __tablename__ = "async_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
