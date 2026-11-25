from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from fastapi.testclient import TestClient
import pytest

from app.config import Settings
from app.main import create_app
from sample_models.generate_models import generate_all
from tests.helpers import upload_and_deploy_bytes, upload_and_deploy_sample


def test_onnx_upload_deploy_and_predict(client: TestClient, tmp_path: Path) -> None:
    # Skip only if onnxruntime isn't installed locally; runs in full inside Docker
    pytest.importorskip("onnxruntime")
    pytest.importorskip("skl2onnx")

    # Export a fresh ONNX classifier, upload it, and deploy it
    onnx_bytes = generate_all(output_dir=tmp_path)["fraud_classifier_v1_onnx"].read_bytes()
    upload_and_deploy_bytes(client, "fraud-onnx", "v1", "fraud_classifier_v1.onnx", onnx_bytes, "onnx", num_features=10)

    # Check both single and batch predictions on the ONNX worker
    pred = client.post("/predict/fraud-onnx", json={"inputs": {"features": [0.1] * 10}})
    assert pred.status_code == 200
    assert pred.json()["outputs"]["prediction"] in (0, 1)
    assert 0.0 <= pred.json()["outputs"]["confidence"] <= 1.0

    batch = client.post("/predict/fraud-onnx/batch", json={"inputs": [{"features": [0.1] * 10}, {"features": [-0.5] * 10}]})
    assert batch.status_code == 200
    assert len(batch.json()["results"]) == 2


def test_inference_timeout_returns_504_and_recovers(tmp_path: Path) -> None:
    # Configure a tight 350ms inference timeout for this test
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'timeout.db').as_posix()}",
        artifact_storage_dir=tmp_path / "artifacts",
        inference_timeout_seconds=0.35,
        log_level="WARNING",
    )
    with TestClient(create_app(settings)) as c:
        upload_and_deploy_sample(c, "slow-model", "v1", "custom_model.py", "custom")

        # Ask the custom model to sleep for 1000ms -> should time out with 504
        res = c.post("/predict/slow-model", json={"inputs": {"features": [1.0, 0.2, 0.3, 0.4, 0.5], "delay_ms": 1000}})
        assert res.status_code == 504
        assert res.json()["error"]["code"] == "INFERENCE_TIMEOUT"

        # Wait for the killed worker to respawn, then confirm fast requests work again
        deadline = time.time() + 10.0
        while time.time() < deadline and not c.get("/models/slow-model/versions/v1/health").json()["healthy"]:
            time.sleep(0.2)
        ok = c.post("/predict/slow-model", json={"inputs": {"features": [1.0, 0.2, 0.3, 0.4, 0.5], "delay_ms": 0}})
        assert ok.status_code == 200


def test_backpressure_queue_overflow_returns_429(tmp_path: Path) -> None:
    # Allow only 1 in-flight request at a time so concurrent calls trigger backpressure
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'bp.db').as_posix()}",
        artifact_storage_dir=tmp_path / "artifacts",
        max_queue_depth_per_deployment=1,
        log_level="WARNING",
    )
    with TestClient(create_app(settings)) as c:
        upload_and_deploy_sample(c, "bp-model", "v1", "custom_model.py", "custom")

        # Fire 3 slow requests at once: at least one succeeds (200) and overflow gets 429
        def _call() -> int:
            return c.post("/predict/bp-model", json={"inputs": {"features": [1.0, 0.2, 0.3, 0.4, 0.5], "delay_ms": 400}}).status_code

        with ThreadPoolExecutor(max_workers=3) as pool:
            statuses = list(pool.map(lambda _: _call(), range(3)))

        assert 200 in statuses
        assert 429 in statuses
