import asyncio
import json
import os
from pathlib import Path
import time

from fastapi.testclient import TestClient
import httpx
import pytest

from app.config import Settings
from app.main import create_app
from sample_models.generate_models import generate_all
from tests.helpers import SAMPLE_DIR, features_schema, upload_and_deploy_bytes, upload_bytes


def test_e2e_multi_format_and_multi_version_lifecycle(client: TestClient, tmp_path: Path) -> None:
    # Scenario 1: Run sklearn v1/v2, ONNX, and custom Python models all at the same time
    pytest.importorskip("onnxruntime")
    pytest.importorskip("skl2onnx")

    generated = generate_all(output_dir=tmp_path)
    upload_and_deploy_bytes(client, "fraud-multi", "v1", "v1.pkl", generated["fraud_classifier_v1"].read_bytes(), "sklearn", num_features=10)
    upload_and_deploy_bytes(client, "fraud-multi", "v2", "v2.pkl", generated["fraud_classifier_v2"].read_bytes(), "sklearn", num_features=10)
    upload_and_deploy_bytes(client, "fraud-onnx-e2e", "v1", "v1.onnx", generated["fraud_classifier_v1_onnx"].read_bytes(), "onnx", num_features=10)
    upload_and_deploy_bytes(client, "custom-e2e", "v1", "custom.py", (SAMPLE_DIR / "custom_model.py").read_bytes(), "custom", num_features=5)

    # Run both single and batch predictions on all 4 live deployments
    vec10 = {"features": [0.2, -0.1, 0.5, 0.3, -0.4, 0.8, 0.1, -0.2, 0.6, 0.0]}
    vec5 = {"features": [0.5, -0.2, 0.7, 0.1, 0.4]}
    for model, ver, payload in [
        ("fraud-multi", "v1", vec10),
        ("fraud-multi", "v2", vec10),
        ("fraud-onnx-e2e", "v1", vec10),
        ("custom-e2e", "v1", vec5),
    ]:
        r_single = client.post(f"/predict/{model}", json={"inputs": payload, "version": ver})
        assert r_single.status_code == 200
        assert r_single.json()["version"] == ver
        assert r_single.json()["outputs"]["prediction"] in (0, 1)

        r_batch = client.post(f"/predict/{model}/batch", json={"inputs": [payload, payload], "version": ver})
        assert r_batch.status_code == 200
        assert len(r_batch.json()["results"]) == 2

    # Confirm deleting a live model is blocked (409) and works cleanly once undeployed
    assert client.delete("/models/fraud-onnx-e2e/versions/v1").status_code == 409
    assert client.delete("/models/fraud-onnx-e2e/versions/v1/deploy").status_code == 200
    assert client.delete("/models/fraud-onnx-e2e/versions/v1").status_code == 200
    assert client.get("/models/fraud-onnx-e2e/versions/v1").status_code == 404


@pytest.mark.asyncio
async def test_high_concurrency_wal_and_backpressure_stress(tmp_path: Path) -> None:
    # Scenario 2: Send 60 concurrent predictions while another model runs slowly
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'stress.db').as_posix()}",
        artifact_storage_dir=tmp_path / "artifacts",
        max_queue_depth_per_deployment=64,
        log_level="WARNING",
    )
    app = create_app(settings)
    custom_py = (SAMPLE_DIR / "custom_model.py").read_bytes()
    pkl_v1 = (SAMPLE_DIR / "fraud_classifier_v1.pkl").read_bytes()

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=30.0) as ac:
            # Deploy a 3-replica fast classifier and a 1-replica slow custom model
            assert (
                await ac.post(
                    "/models/fast-clf/versions",
                    data={"metadata": json.dumps({"framework": "sklearn", "version": "v1", "input_schema": features_schema(10), "output_schema": {"type": "object"}})},
                    files={"artifact": ("clf.pkl", pkl_v1, "application/octet-stream")},
                )
            ).status_code == 201
            assert (await ac.post("/models/fast-clf/versions/v1/deploy", json={"replicas": 3, "resources": {"memory": "512Mi", "cpu": "0.5"}})).status_code == 200

            assert (
                await ac.post(
                    "/models/slow-other/versions",
                    data={"metadata": json.dumps({"framework": "custom", "version": "v1", "input_schema": {"type": "object"}, "output_schema": {"type": "object"}})},
                    files={"artifact": ("custom.py", custom_py, "text/x-python")},
                )
            ).status_code == 201
            assert (await ac.post("/models/slow-other/versions/v1/deploy", json={"replicas": 1, "resources": {"memory": "512Mi", "cpu": "0.5"}})).status_code == 200

            # Run 60 concurrent fast requests at the same time as one 500ms slow request
            fast_payload = {"inputs": {"features": [0.1] * 10}}
            slow_payload = {"inputs": {"features": [1.0, 0.2, 0.3, 0.4, 0.5], "delay_ms": 500}}
            slow_task = asyncio.create_task(ac.post("/predict/slow-other", json=slow_payload))
            fast_responses = await asyncio.gather(*[ac.post("/predict/fast-clf", json=fast_payload) for _ in range(60)])
            slow_response = await slow_task

            # Every single request should succeed with zero SQLite lock errors
            assert all(r.status_code == 200 for r in fast_responses)
            assert slow_response.status_code == 200

            # Confirm /metrics counted all 60 requests accurately
            metrics = (await ac.get("/models/fast-clf/metrics?period=1h")).json()
            assert metrics["requests_total"] == 60
            assert metrics["errors_total"] == 0
            assert metrics["latency"]["p50_ms"] <= metrics["latency"]["p95_ms"] <= metrics["latency"]["p99_ms"]

            # Shrink queue cap to 1 and confirm excess concurrent requests get 429
            app.state.settings.max_queue_depth_per_deployment = 1
            bp_statuses = [r.status_code for r in await asyncio.gather(*[ac.post("/predict/slow-other", json=slow_payload) for _ in range(5)])]
            assert 200 in bp_statuses and 429 in bp_statuses


def test_canary_rollout_and_auto_rollback_e2e(client: TestClient) -> None:
    # Scenario 3: Send normal unpinned traffic to a faulty canary and watch it auto-rollback
    faulty_py = b"class Model:\n    def predict(self, inputs):\n        raise RuntimeError('Injected canary fault')\n"
    upload_and_deploy_bytes(client, "payments-canary", "v1", "v1.py", (SAMPLE_DIR / "custom_model.py").read_bytes(), "custom")
    upload_and_deploy_bytes(client, "payments-canary", "v2", "v2.py", faulty_py, "custom")

    # Start a 50% rollout for v2 with a 15% error limit over at least 4 requests
    roll = client.post(
        "/models/payments-canary/rollout",
        json={"target_version": "v2", "weight": 50, "max_error_rate": 0.15, "min_requests": 4},
    )
    assert roll.status_code == 200

    # Keep sending unpinned traffic until auto-rollback turns the canary off
    payload = {"inputs": {"features": [0.5, 0.1, 0.8, 0.2, 0.4]}}
    for _ in range(30):
        client.post("/predict/payments-canary", json=payload)
        if client.get("/models/payments-canary/routing").json()["canary"] is None:
            break

    # Verify traffic is back at 100% v1 and new requests succeed cleanly
    final_routing = client.get("/models/payments-canary/routing").json()
    assert final_routing["rules"] == [{"version": "v1", "weight": 100}]
    assert final_routing["canary"] is None

    for _ in range(5):
        ok = client.post("/predict/payments-canary", json=payload)
        assert ok.status_code == 200 and ok.json()["version"] == "v1"


def test_strict_boundary_and_malformed_edge_cases(client: TestClient) -> None:
    # Scenario 4: Check bad weights, NaN/null/short features, duplicates, and missing models
    pkl_v1 = (SAMPLE_DIR / "fraud_classifier_v1.pkl").read_bytes()
    upload_and_deploy_bytes(client, "edge-model", "v1", "v1.pkl", pkl_v1, "sklearn", num_features=10)

    # Duplicate upload (409) and invalid routing weights (422 / 400)
    assert upload_bytes(client, "edge-model", "v1", "sklearn", "v1.pkl", pkl_v1, features_schema(10)).status_code == 409
    assert client.put("/models/edge-model/routing", json={"rules": [{"version": "v1", "weight": 80}]}).status_code == 422
    assert client.put("/models/edge-model/routing", json={"rules": [{"version": "v1", "weight": 120}]}).status_code == 422
    assert client.put("/models/edge-model/routing", json={"rules": [{"version": "v1", "weight": -10}]}).status_code == 422
    assert client.put("/models/edge-model/routing", json={"rules": [{"version": "v1", "weight": 50}, {"version": "v2", "weight": 50}]}).status_code == 400

    # Malformed prediction inputs (NaN, null, wrong length) and unknown models (404)
    nan_body = '{"inputs": {"features": [0.1, NaN, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}}'
    assert client.post("/predict/edge-model", content=nan_body, headers={"Content-Type": "application/json"}).status_code == 422
    assert client.post("/predict/edge-model", json={"inputs": {"features": [0.1, None, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}}).status_code == 422
    assert client.post("/predict/edge-model", json={"inputs": {"features": [0.1, 0.2, 0.3]}}).status_code == 422
    assert client.get("/models/does-not-exist/metrics?period=1h").status_code == 404
    assert client.get("/models/does-not-exist/versions/v1/health").status_code == 404


def test_clean_shutdown_leaves_no_orphan_worker_processes(tmp_path: Path) -> None:
    # Scenario 5: Make sure undeploy and app shutdown kill every child worker process cleanly
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'shutdown.db').as_posix()}",
        artifact_storage_dir=tmp_path / "artifacts",
        log_level="WARNING",
    )
    app = create_app(settings)
    custom_py = (SAMPLE_DIR / "custom_model.py").read_bytes()
    tracked_pids: list[int] = []

    with TestClient(app) as c:
        # Deploy two models with 2 replicas each and record all 4 child process IDs
        upload_and_deploy_bytes(c, "proc-check-a", "v1", "a.py", custom_py, "custom", replicas=2)
        upload_and_deploy_bytes(c, "proc-check-b", "v1", "b.py", custom_py, "custom", replicas=2)

        mgr = app.state.serving_manager
        pids_a = [w._process.pid for w in mgr._pools[("proc-check-a", "v1")]._workers if w._process is not None]
        pids_b = [w._process.pid for w in mgr._pools[("proc-check-b", "v1")]._workers if w._process is not None]
        assert len(pids_a) == 2 and len(pids_b) == 2
        tracked_pids.extend(pids_a + pids_b)

        # Undeploy model A and check that its 2 worker processes are completely gone
        assert c.delete("/models/proc-check-a/versions/v1/deploy").status_code == 200
        time.sleep(0.2)
        for pid in pids_a:
            assert not _pid_is_running(pid), f"Worker PID {pid} still alive after undeploy"

    # Once the app shuts down, model B's workers must also be completely gone (no zombies)
    time.sleep(0.2)
    for pid in tracked_pids:
        assert not _pid_is_running(pid), f"Orphan worker PID {pid} still alive after app shutdown"


def _pid_is_running(pid: int) -> bool:
    # Check /proc on Linux (catches zombie processes too), and fall back to os.kill(pid, 0)
    if os.name == "posix" and Path(f"/proc/{pid}/stat").exists():
        return True
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False
