import bisect
from datetime import datetime
import hashlib
import logging
import random
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Deployment, InferenceLog, RoutingConfig, utcnow
from app.db.queries import require_model
from app.errors import AppError
from app.schemas.routing import RollbackRequest, RolloutRequest, RoutingResponse, RoutingRule, RoutingUpdateRequest

logger = logging.getLogger(__name__)


def normalize_weights(rules: list[dict[str, Any]], target_total: int = 100) -> list[dict[str, Any]]:
    # Scale all weights proportionally so they add up to target_total (usually 100%)
    total = sum(int(r["weight"]) for r in rules)
    if not rules or total <= 0:
        return []
    normalized: list[dict[str, Any]] = []
    allocated = 0
    for idx, r in enumerate(rules):
        # Give the last item the leftover remainder so rounding never leaves us at 99 or 101
        w = (target_total - allocated) if idx == len(rules) - 1 else round(int(r["weight"]) * target_total / total)
        allocated += w
        if w > 0:
            normalized.append({"version": r["version"], "weight": w})
    return normalized


class RoutingService:
    def _running_versions(self, db: Session, model_name: str) -> set[str]:
        # Get the set of version names that are actually running right now
        rows = db.scalars(
            select(Deployment.version).where(
                Deployment.model_name == model_name,
                Deployment.status == "running",
            )
        ).all()
        return set(rows)

    def get_routing(self, db: Session, model_name: str) -> RoutingResponse:
        # Return current routing rules, or an empty list if nothing has been deployed yet
        require_model(db, model_name)
        cfg = db.get(RoutingConfig, model_name)
        if cfg is None:
            return RoutingResponse(model=model_name, rules=[], previous_rules=None, canary=None, updated_at=utcnow())
        return self._to_response(cfg)

    def update_routing(
        self,
        db: Session,
        model_name: str,
        req: RoutingUpdateRequest,
        canary_config: dict[str, Any] | None = None,
    ) -> RoutingResponse:
        require_model(db, model_name)
        if not req.rules:
            raise AppError(code="INVALID_ROUTING_RULES", message="Routing rules must not be empty", status_code=422)

        # Check every rule: no duplicate versions, all weighted versions must be running
        seen: set[str] = set()
        total_weight = 0
        running = self._running_versions(db, model_name)
        for rule in req.rules:
            if rule.version in seen:
                raise AppError(
                    code="INVALID_ROUTING_RULES",
                    message=f"Duplicate version '{rule.version}' in routing rules",
                    status_code=422,
                )
            seen.add(rule.version)
            if rule.weight > 0 and rule.version not in running:
                raise AppError(
                    code="INVALID_ROUTING_RULES",
                    message=f"Version '{rule.version}' is not currently running and cannot receive traffic",
                    status_code=400,
                )
            total_weight += rule.weight

        # Make sure the percentages add up to exactly 100
        if total_weight != 100:
            raise AppError(
                code="INVALID_ROUTING_RULES",
                message=f"Routing weights must sum to 100 (got {total_weight})",
                status_code=422,
            )

        # Save the old rules into previous_rules first so we can roll back in one click
        serialized = [r.model_dump() for r in req.rules]
        cfg = db.get(RoutingConfig, model_name)
        if cfg is None:
            cfg = RoutingConfig(
                model_name=model_name,
                rules=serialized,
                previous_rules=None,
                canary_config=canary_config,
            )
            db.add(cfg)
        else:
            if cfg.rules:
                cfg.previous_rules = list(cfg.rules)
            cfg.rules = serialized
            cfg.canary_config = canary_config
            cfg.updated_at = utcnow()
        db.commit()
        db.refresh(cfg)
        return self._to_response(cfg)

    def start_or_advance_rollout(self, db: Session, model_name: str, req: RolloutRequest) -> RoutingResponse:
        # Confirm the canary version is already deployed and running
        running = self._running_versions(db, model_name)
        if req.target_version not in running:
            raise AppError(
                code="MODEL_NOT_DEPLOYED",
                message=f"Canary target version '{req.target_version}' is not running",
                status_code=400,
            )

        cfg = db.get(RoutingConfig, model_name)
        current_rules = cfg.rules if cfg else []
        others = [r for r in current_rules if r["version"] != req.target_version and r["version"] in running]

        # If we're moving to 100% (or there are no other versions), finish the rollout
        if req.weight == 100 or not others:
            return self.update_routing(
                db,
                model_name,
                RoutingUpdateRequest(rules=[RoutingRule(version=req.target_version, weight=100)]),
                canary_config=None,
            )

        # Give the canary its target % and split the rest across the existing stable versions
        remaining_rules = normalize_weights(others, target_total=100 - req.weight)
        new_rules = [RoutingRule(version=req.target_version, weight=req.weight)] + [
            RoutingRule.model_validate(r) for r in remaining_rules
        ]
        # Store the error-rate safety guard so bad canaries auto-rollback
        canary = {
            "target_version": req.target_version,
            "weight": req.weight,
            "max_error_rate": req.max_error_rate,
            "min_requests": req.min_requests,
            "started_at": utcnow().isoformat(),
        }
        return self.update_routing(db, model_name, RoutingUpdateRequest(rules=new_rules), canary_config=canary)

    def rollback(self, db: Session, model_name: str, req: RollbackRequest | None = None) -> RoutingResponse:
        require_model(db, model_name)
        running = self._running_versions(db, model_name)

        # If the caller named a specific version to roll back to, send 100% of traffic there
        if req and req.target_version:
            if req.target_version not in running:
                raise AppError(
                    code="MODEL_NOT_DEPLOYED",
                    message=f"Rollback target version '{req.target_version}' is not running",
                    status_code=400,
                )
            return self.update_routing(
                db,
                model_name,
                RoutingUpdateRequest(rules=[RoutingRule(version=req.target_version, weight=100)]),
                canary_config=None,
            )

        # Otherwise restore the last saved routing rules from previous_rules
        cfg = db.get(RoutingConfig, model_name)
        if cfg is None or not cfg.previous_rules:
            raise AppError(
                code="NO_ROLLBACK_TARGET",
                message=f"No previous routing configuration available to roll back '{model_name}'",
                status_code=409,
            )

        # Keep only versions from previous_rules that are still running and rescale to 100%
        valid_prev = [r for r in cfg.previous_rules if r["version"] in running and r["weight"] > 0]
        if not valid_prev:
            raise AppError(
                code="NO_ROLLBACK_TARGET",
                message="None of the versions in previous routing configuration are currently running",
                status_code=409,
            )

        restored = [RoutingRule.model_validate(r) for r in normalize_weights(valid_prev, target_total=100)]
        return self.update_routing(db, model_name, RoutingUpdateRequest(rules=restored), canary_config=None)

    def select_version(
        self,
        db: Session,
        model_name: str,
        explicit_version: str | None = None,
        session_id: str | None = None,
    ) -> str:
        running = self._running_versions(db, model_name)
        # If the caller asked for a specific version, check that it is running and use it
        if explicit_version is not None:
            if explicit_version not in running:
                raise AppError(
                    code="MODEL_NOT_DEPLOYED",
                    message=f"Model '{model_name}' version '{explicit_version}' is not deployed",
                    status_code=503,
                )
            return explicit_version

        if not running:
            raise AppError(
                code="MODEL_NOT_DEPLOYED",
                message=f"No running deployment found for model '{model_name}'",
                status_code=503,
            )

        # Only consider routing rules whose versions are actually running right now
        cfg = db.get(RoutingConfig, model_name)
        active_rules = [r for r in (cfg.rules if cfg else []) if r["version"] in running and r["weight"] > 0]
        if not active_rules:
            return sorted(running)[-1]

        # Turn weights like [90, 10] into running totals [90, 100] so we can pick a bucket fast
        cumulative: list[int] = []
        versions: list[str] = []
        total = 0
        for rule in active_rules:
            total += int(rule["weight"])
            cumulative.append(total)
            versions.append(rule["version"])

        # For sticky sessions, hash the session ID to always get the same bucket; otherwise pick randomly
        if session_id:
            digest = hashlib.sha256(f"{model_name}:{session_id}".encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % total
        else:
            bucket = random.randrange(total)

        idx = bisect.bisect_right(cumulative, bucket)
        return versions[min(idx, len(versions) - 1)]

    def evaluate_canary_auto_rollback(self, db: Session, model_name: str, version: str) -> bool:
        # Only check if there is an active canary rollout targeting this exact version
        cfg = db.get(RoutingConfig, model_name)
        if cfg is None or not cfg.canary_config:
            return False
        canary = cfg.canary_config
        if canary.get("target_version") != version:
            return False

        # Count how many requests this canary handled since the rollout began
        stmt = select(InferenceLog).where(
            InferenceLog.model_name == model_name,
            InferenceLog.version == version,
        )
        if canary.get("started_at"):
            stmt = stmt.where(InferenceLog.timestamp >= datetime.fromisoformat(canary["started_at"]))

        # Wait until we have enough samples (min_requests) before judging the error rate
        logs = db.scalars(stmt).all()
        if len(logs) < int(canary.get("min_requests", 5)):
            return False

        # If errors exceed the allowed threshold, automatically roll traffic back to the old version
        error_rate = sum(1 for row in logs if row.error is not None) / len(logs)
        if error_rate <= float(canary.get("max_error_rate", 0.1)):
            return False

        logger.warning(
            "Auto-rolling back canary %s:%s (error_rate=%.2f > %.2f)",
            model_name,
            version,
            error_rate,
            canary.get("max_error_rate", 0.1),
        )
        try:
            self.rollback(db, model_name)
            return True
        except AppError:
            return False

    @staticmethod
    def _to_response(cfg: RoutingConfig) -> RoutingResponse:
        return RoutingResponse(
            model=cfg.model_name,
            rules=[RoutingRule.model_validate(r) for r in (cfg.rules or [])],
            previous_rules=(
                [RoutingRule.model_validate(r) for r in cfg.previous_rules] if cfg.previous_rules else None
            ),
            canary=cfg.canary_config,
            updated_at=cfg.updated_at,
        )
