from fastapi.testclient import TestClient

from tests.helpers import SAMPLE_DIR, upload_and_deploy_bytes, upload_and_deploy_sample


def test_metrics_empty_state_and_invalid_period(client: TestClient) -> None:
    upload_and_deploy_sample(client, "empty-metrics", "v1", "fraud_classifier_v1.pkl", "sklearn")

    # With 0 predictions, /metrics should return zeros cleanly instead of crashing
    res = client.get("/models/empty-metrics/metrics?period=1h")
    assert res.status_code == 200
    body = res.json()
    assert body["model"] == "empty-metrics"
    assert body["period"] == "1h"
    assert body["requests_total"] == 0
    assert body["errors_total"] == 0
    assert body["error_rate"] == 0.0
    assert body["latency"] == {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    assert body["by_version"] == {}

    # Reject bad period strings (400) and unknown models (404)
    bad = client.get("/models/empty-metrics/metrics?period=yesterday")
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == "INVALID_PERIOD"
    assert client.get("/models/nonexistent/metrics?period=1h").status_code == 404


def test_metrics_aggregation_by_version_confidence_and_errors(client: TestClient) -> None:
    good_bytes = (SAMPLE_DIR / "custom_model.py").read_bytes()
    bad_bytes = b"""
class Model:
    def predict(self, inputs):
        raise ValueError("boom")
"""
    upload_and_deploy_bytes(client, "obs-model", "v1", "custom_model.py", good_bytes, "custom")
    upload_and_deploy_bytes(client, "obs-model", "v2", "bad.py", bad_bytes, "custom")

    # Send 8 passing requests to v1 and 2 failing requests to v2 (20% total error rate)
    payload = {"features": [1.0, 0.2, 0.8, 0.5, 0.9]}
    for _ in range(8):
        assert client.post("/predict/obs-model", json={"inputs": payload, "version": "v1"}).status_code == 200
    for _ in range(2):
        assert client.post("/predict/obs-model", json={"inputs": payload, "version": "v2"}).status_code == 500

    # Check overall totals, latency percentiles, per-version split, and confidence buckets
    m = client.get("/models/obs-model/metrics?period=1h").json()
    assert m["requests_total"] == 10
    assert m["errors_total"] == 2
    assert m["error_rate"] == 0.2
    assert m["throughput_rps"] > 0
    assert m["latency"]["p50_ms"] <= m["latency"]["p95_ms"] <= m["latency"]["p99_ms"]
    assert m["by_version"]["v1"]["requests"] == 8
    assert m["by_version"]["v1"]["errors"] == 0
    assert m["by_version"]["v2"]["requests"] == 2
    assert m["by_version"]["v2"]["errors"] == 2
    assert m["confidence_distribution"]["count"] == 8
    assert sum(m["confidence_distribution"]["buckets"].values()) == 8


def test_input_drift_detection_flags_distribution_shift(client: TestClient) -> None:
    upload_and_deploy_sample(client, "drift-model", "v1", "custom_model.py", "custom")

    # Send 20 normal baseline inputs around 0.1, then 6 huge shifted inputs at 50.0
    for i in range(20):
        delta = (i % 5) * 0.02
        assert client.post("/predict/drift-model", json={"inputs": {"features": [0.1 + delta, 0.2, 0.1, 0.3, 0.2]}}).status_code == 200
    for _ in range(6):
        assert client.post("/predict/drift-model", json={"inputs": {"features": [50.0, 50.0, 50.0, 50.0, 50.0]}}).status_code == 200

    # Confirm the drift tracker spots the large shift (z-score > 3)
    m = client.get("/models/drift-model/metrics?period=all").json()
    assert m["drift"] is not None
    assert m["drift"]["drift_detected"] is True
    assert m["drift"]["max_z_score"] > 3.0
