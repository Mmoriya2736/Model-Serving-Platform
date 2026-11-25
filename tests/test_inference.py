import time

from fastapi.testclient import TestClient

from app.db.models import InferenceLog
from tests.helpers import upload_and_deploy_sample


def test_single_inference_classifier_regressor_and_custom(client: TestClient) -> None:
    # Deploy all three sample models (classifier, regressor, and custom Python class)
    upload_and_deploy_sample(client, "fraud-detector", "v1", "fraud_classifier_v1.pkl", "sklearn", num_features=10)
    upload_and_deploy_sample(client, "house-price", "v1", "house_regressor_v1.pkl", "sklearn", num_features=5)
    upload_and_deploy_sample(client, "custom-scorer", "v1", "custom_model.py", "custom", num_features=5)

    # Call the classifier and check every field of the standard prediction response
    res = client.post(
        "/predict/fraud-detector",
        json={"inputs": {"features": [0.1, 0.5, 0.3, -0.2, 0.8, 0.1, 0.4, -0.7, 0.9, 0.0]}},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["model"] == "fraud-detector"
    assert body["version"] == "v1"
    assert body["outputs"]["prediction"] in (0, 1)
    assert 0.0 <= body["outputs"]["confidence"] <= 1.0
    assert body["latency_ms"] >= 0
    assert body["request_id"].startswith("req_")

    # Call the regressor and check we get a continuous float prediction
    reg = client.post("/predict/house-price", json={"inputs": {"features": [1.0, 2.0, 3.0, 4.0, 5.0]}, "version": "v1"})
    assert reg.status_code == 200
    assert isinstance(reg.json()["outputs"]["prediction"], float)

    # Call the custom Python model and verify its extra risk_score field is returned
    cust = client.post("/predict/custom-scorer", json={"inputs": {"features": [1.0, 0.2, 0.8, 0.5, 0.9]}})
    assert cust.status_code == 200
    assert cust.json()["outputs"]["prediction"] in (0, 1)
    assert "risk_score" in cust.json()["outputs"]

    # Check the DB log: it should store only a SHA-256 hash, never the raw input numbers
    with client.app.state.session_factory() as db:
        logs = db.query(InferenceLog).filter_by(model_name="fraud-detector").all()
        assert len(logs) == 1
        assert len(logs[0].input_hash) == 64
        assert "0.5" not in logs[0].input_hash


def test_batch_and_async_prediction(client: TestClient) -> None:
    upload_and_deploy_sample(client, "custom-batch", "v1", "custom_model.py", "custom", num_features=5)

    # Send 3 inputs in one batch and confirm we get 3 results plus total_latency_ms
    batch_res = client.post(
        "/predict/custom-batch/batch",
        json={
            "inputs": [
                {"features": [0.1, 0.2, 0.3, 0.4, 0.5]},
                {"features": [1.0, 0.0, 1.0, 0.5, 0.9]},
                {"features": [-1.0, -0.5, -0.2, -0.1, -0.3]},
            ]
        },
    )
    assert batch_res.status_code == 200
    body = batch_res.json()
    assert len(body["results"]) == 3
    assert body["total_latency_ms"] >= 0
    for item in body["results"]:
        assert item["outputs"]["prediction"] in (0, 1)
        assert item["latency_ms"] >= 0

    # Queue an async prediction with an 80ms artificial delay -> returns 202 and a job_id
    submit = client.post(
        "/predict/custom-batch/async",
        json={"inputs": {"features": [1.0, 0.2, 0.8, 0.5, 0.9], "delay_ms": 80}},
    )
    assert submit.status_code == 202
    job_id = submit.json()["job_id"]
    assert job_id.startswith("job_")

    # Poll /jobs/{job_id} until the background thread finishes
    deadline = time.time() + 10.0
    final_job = None
    while time.time() < deadline:
        poll = client.get(f"/jobs/{job_id}")
        assert poll.status_code == 200
        if poll.json()["status"] in ("completed", "failed"):
            final_job = poll.json()
            break
        time.sleep(0.05)

    assert final_job is not None
    assert final_job["status"] == "completed"
    assert final_job["result"]["outputs"]["prediction"] in (0, 1)


def test_inference_validation_and_error_cases(client: TestClient) -> None:
    # Predicting against a model that isn't deployed returns 503
    r_un = client.post("/predict/ghost-model", json={"inputs": {"features": [1, 2, 3]}})
    assert r_un.status_code == 503
    assert r_un.json()["error"]["code"] == "MODEL_NOT_DEPLOYED"

    upload_and_deploy_sample(client, "strict-clf", "v1", "fraud_classifier_v1.pkl", "sklearn", num_features=10)

    # Check schema validation (wrong count, strings instead of numbers, missing key) and empty batch
    assert client.post("/predict/strict-clf", json={"inputs": {"features": [1, 2, 3, 4, 5]}}).status_code == 422
    assert client.post("/predict/strict-clf", json={"inputs": {"features": ["a"] * 10}}).status_code == 422
    assert client.post("/predict/strict-clf", json={"inputs": {}}).status_code == 422
    assert client.post("/predict/strict-clf", json={"inputs": {"features": [0.1] * 10}, "version": "v99"}).status_code == 503
    assert client.post("/predict/strict-clf/batch", json={"inputs": []}).status_code == 422
    assert client.get("/jobs/job_missing").status_code == 404
