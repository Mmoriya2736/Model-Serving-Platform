from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import threading
import time
from typing import Any
import uuid

from jsonschema import Draft7Validator, ValidationError as JsonSchemaValidationError
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db.models import AsyncJob, InferenceLog, utcnow
from app.db.queries import require_model_version
from app.errors import AppError
from app.runtime.manager import ServingManager
from app.schemas.inference import (
    AsyncJobResponse,
    AsyncPredictSubmitResponse,
    BatchPredictItem,
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)
from app.services.routing_service import RoutingService


def compute_input_hash(payload: dict[str, Any]) -> str:
    # Sort keys and hash the input so identical inputs give the same hash without saving raw data
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _assert_finite(value: Any, path: str = "inputs") -> None:
    # Walk through nested dicts/lists and reject NaN or Infinity values right away
    if isinstance(value, float) and not math.isfinite(value):
        raise AppError(code="SCHEMA_VALIDATION_ERROR", message=f"Non-finite float value at '{path}'", status_code=422)
    if isinstance(value, dict):
        for k, v in value.items():
            _assert_finite(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _assert_finite(v, f"{path}[{i}]")


def _extract_confidence(outputs: dict[str, Any]) -> float | None:
    # Pull out a numeric "confidence" score if the model returned one
    conf = outputs.get("confidence")
    if isinstance(conf, (int, float)) and not isinstance(conf, bool) and math.isfinite(float(conf)):
        return float(conf)
    return None


# Tracks running averages of input features to spot when live data drifts away from the baseline
class DriftTracker:
    def __init__(self, baseline_window: int = 20) -> None:
        self.baseline_window = baseline_window
        self._lock = threading.Lock()
        self._stats: dict[str, dict[str, dict[str, Any]]] = {}

    def record(self, model_name: str, inputs: dict[str, Any]) -> None:
        # Only track requests that have a non-empty list of numbers under "features"
        features = inputs.get("features")
        if not isinstance(features, list) or not features:
            return
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in features):
            return
        vec = [float(x) for x in features]

        # Reset stats if this is the first time we see this model (or if feature count changed)
        with self._lock:
            entry = self._stats.setdefault(model_name, {})
            if "baseline" not in entry or len(entry["baseline"]["mean"]) != len(vec):
                entry["baseline"] = {"count": 0, "mean": [0.0] * len(vec), "m2": [0.0] * len(vec)}
                entry["recent"] = {"count": 0, "mean": [0.0] * len(vec), "m2": [0.0] * len(vec)}

            # First 20 requests build the baseline; everything after that goes into "recent"
            target = entry["baseline"] if entry["baseline"]["count"] < self.baseline_window else entry["recent"]
            self._update_welford(target, vec)

    @staticmethod
    def _update_welford(bucket: dict[str, Any], vec: list[float]) -> None:
        # Update running mean and variance in one pass without keeping raw input arrays in memory
        bucket["count"] += 1
        n = bucket["count"]
        mean, m2 = bucket["mean"], bucket["m2"]
        for i, x in enumerate(vec):
            delta = x - mean[i]
            mean[i] += delta / n
            m2[i] += delta * (x - mean[i])

    def summarize(self, model_name: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._stats.get(model_name)
            if not entry:
                return None
            base, rec = entry["baseline"], entry["recent"]
            # Need at least 2 baseline and 2 recent points before comparing distributions
            if base["count"] < 2 or rec["count"] < 2:
                return {
                    "status": "collecting_baseline",
                    "baseline_samples": base["count"],
                    "recent_samples": rec["count"],
                    "max_z_score": 0.0,
                    "drift_detected": False,
                }
            # Measure how many standard deviations each feature's average has shifted
            z_scores = []
            for i in range(len(base["mean"])):
                variance = base["m2"][i] / max(1, base["count"] - 1)
                std = math.sqrt(variance) if variance > 1e-12 else 1.0
                z_scores.append(abs(rec["mean"][i] - base["mean"][i]) / std)
            # Flag drift if any feature shifted by more than 3 standard deviations
            max_z = round(max(z_scores, default=0.0), 3)
            return {
                "status": "drift_detected" if max_z > 3.0 else "stable",
                "baseline_samples": base["count"],
                "recent_samples": rec["count"],
                "max_z_score": max_z,
                "drift_detected": max_z > 3.0,
            }


class InferenceService:
    def __init__(
        self,
        settings: Settings,
        serving_manager: ServingManager,
        routing_service: RoutingService,
        session_factory: sessionmaker[Session],
    ) -> None:
        self.settings = settings
        self.serving_manager = serving_manager
        self.routing_service = routing_service
        self.session_factory = session_factory
        self.drift_tracker = DriftTracker()
        # Small thread pool used to run /predict/{model}/async jobs in the background
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="async-infer")

    def _validate_inputs(self, schema: dict[str, Any], inputs: dict[str, Any], prefix: str = "inputs") -> None:
        # Reject NaN/Inf first, then check the input against the model's registered JSON Schema
        _assert_finite(inputs, prefix)
        try:
            Draft7Validator(schema).validate(inputs)
        except JsonSchemaValidationError as exc:
            raise AppError(
                code="SCHEMA_VALIDATION_ERROR",
                message=f"Input failed schema validation ({prefix}): {exc.message}",
                status_code=422,
                details={"path": list(exc.path)},
            ) from exc

    def _log_inference(
        self,
        db: Session,
        model_name: str,
        version: str,
        request_id: str,
        input_hash: str,
        latency_ms: float,
        confidence: float | None = None,
        error: str | None = None,
    ) -> None:
        # Save one lightweight log entry in SQLite for the /metrics endpoint
        db.add(
            InferenceLog(
                model_name=model_name,
                version=version,
                request_id=request_id,
                input_hash=input_hash,
                latency_ms=latency_ms,
                confidence=confidence,
                error=error,
            )
        )
        db.commit()

    def predict(
        self,
        db: Session,
        model_name: str,
        req: PredictRequest,
        request_id: str,
        session_id: str | None = None,
    ) -> PredictResponse:
        # Step 1: Pick which model version should handle this request and validate inputs
        version = self.routing_service.select_version(
            db, model_name, explicit_version=req.version, session_id=session_id
        )
        mv = require_model_version(db, model_name, version)
        self._validate_inputs(mv.input_schema, req.inputs)

        # Step 2: Send inputs to an isolated worker process and record any failure
        input_hash = compute_input_hash(req.inputs)
        start = time.perf_counter()
        try:
            outputs, latency_ms = self.serving_manager.predict(model_name, version, req.inputs)
        except AppError as exc:
            elapsed = round((time.perf_counter() - start) * 1000.0, 3)
            self._log_inference(db, model_name, version, request_id, input_hash, elapsed, error=exc.message)
            # Check if this error pushes a canary rollout over its allowed error limit
            self.routing_service.evaluate_canary_auto_rollback(db, model_name, version)
            raise

        # Step 3: Update drift stats, log success metrics, and return the prediction
        self.drift_tracker.record(model_name, req.inputs)
        self._log_inference(
            db, model_name, version, request_id, input_hash, latency_ms, confidence=_extract_confidence(outputs)
        )
        return PredictResponse(
            model=model_name, version=version, outputs=outputs, latency_ms=latency_ms, request_id=request_id
        )

    def predict_batch(
        self,
        db: Session,
        model_name: str,
        req: BatchPredictRequest,
        request_id: str,
        session_id: str | None = None,
    ) -> BatchPredictResponse:
        # Reject empty batches or batches bigger than our configured max size
        if not req.inputs:
            raise AppError(code="INVALID_BATCH", message="Batch 'inputs' must contain at least 1 item", status_code=422)
        if len(req.inputs) > self.settings.max_batch_size:
            raise AppError(
                code="INVALID_BATCH",
                message=f"Batch size {len(req.inputs)} exceeds max_batch_size={self.settings.max_batch_size}",
                status_code=422,
            )

        # Pick the target version and validate every item in the batch before running the model
        version = self.routing_service.select_version(
            db, model_name, explicit_version=req.version, session_id=session_id
        )
        mv = require_model_version(db, model_name, version)
        for idx, item in enumerate(req.inputs):
            self._validate_inputs(mv.input_schema, item, prefix=f"inputs[{idx}]")

        # Run all items in one batch call to the worker process
        start = time.perf_counter()
        try:
            outputs_list, total_latency_ms = self.serving_manager.predict_batch(model_name, version, req.inputs)
        except AppError as exc:
            elapsed = round((time.perf_counter() - start) * 1000.0, 3)
            self._log_inference(
                db, model_name, version, request_id, compute_input_hash({"batch_size": len(req.inputs)}), elapsed, error=exc.message
            )
            self.routing_service.evaluate_canary_auto_rollback(db, model_name, version)
            raise

        # Split total latency across items, log each item in one DB commit, and build response
        per_item_ms = round(total_latency_ms / len(outputs_list), 3)
        results: list[BatchPredictItem] = []
        for item_in, item_out in zip(req.inputs, outputs_list):
            self.drift_tracker.record(model_name, item_in)
            db.add(
                InferenceLog(
                    model_name=model_name,
                    version=version,
                    request_id=request_id,
                    input_hash=compute_input_hash(item_in),
                    latency_ms=per_item_ms,
                    confidence=_extract_confidence(item_out),
                )
            )
            results.append(BatchPredictItem(outputs=item_out, latency_ms=per_item_ms))
        db.commit()

        return BatchPredictResponse(
            model=model_name, version=version, results=results, total_latency_ms=total_latency_ms, request_id=request_id
        )

    def submit_async_predict(
        self,
        db: Session,
        model_name: str,
        req: PredictRequest,
        request_id: str,
        session_id: str | None = None,
    ) -> AsyncPredictSubmitResponse:
        # Check model availability and input schema right away so bad requests fail immediately
        version = self.routing_service.select_version(
            db, model_name, explicit_version=req.version, session_id=session_id
        )
        mv = require_model_version(db, model_name, version)
        self._validate_inputs(mv.input_schema, req.inputs)

        # Save a "pending" job row and hand the work off to our background thread pool
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        db.add(AsyncJob(job_id=job_id, model_name=model_name, version=version, status="pending"))
        db.commit()

        self._executor.submit(self._run_async_job, job_id, model_name, version, req.inputs, request_id)
        return AsyncPredictSubmitResponse(job_id=job_id, model=model_name, version=version, status="pending")

    def _run_async_job(
        self, job_id: str, model_name: str, version: str, inputs: dict[str, Any], request_id: str
    ) -> None:
        # Open a fresh DB session for this background thread and mark the job "running"
        with self.session_factory() as db:
            job = db.get(AsyncJob, job_id)
            if job is None:
                return
            job.status = "running"
            db.commit()

            # Run the normal predict() path and save either the result or the error message
            try:
                resp = self.predict(db, model_name, PredictRequest(inputs=inputs, version=version), request_id)
                job.status = "completed"
                job.result = resp.model_dump()
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
            job.completed_at = utcnow()
            db.commit()

    def get_job(self, db: Session, job_id: str) -> AsyncJobResponse:
        # Look up the background job by ID or return 404
        job = db.get(AsyncJob, job_id)
        if job is None:
            raise AppError(code="NOT_FOUND", message=f"Async job '{job_id}' not found", status_code=404)
        return AsyncJobResponse(
            job_id=job.job_id,
            model=job.model_name,
            version=job.version,
            status=job.status,
            result=job.result,
            error=job.error,
            created_at=job.created_at,
            completed_at=job.completed_at,
        )

    def shutdown(self) -> None:
        # Stop the background async thread pool when the server exits
        self._executor.shutdown(wait=False, cancel_futures=True)
