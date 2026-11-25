import time

from fastapi.testclient import TestClient

from tests.helpers import upload_bytes, upload_sample


def test_deploy_scale_health_undeploy_and_delete_lifecycle(client: TestClient) -> None:
    # Upload the sample classifier and deploy it with 1 worker process
    upload_sample(client, "fraud-detector", "v1", "fraud_classifier_v1.pkl", "sklearn")
    dep = client.post(
        "/models/fraud-detector/versions/v1/deploy",
        json={"replicas": 1, "resources": {"memory": "512Mi", "cpu": "0.5"}},
    )
    assert dep.status_code == 200
    body = dep.json()
    assert body["deployment_id"].startswith("dep_")
    assert body["status"] == "running"
    assert body["endpoint"] == "/predict/fraud-detector"
    dep_id = body["deployment_id"]

    # Deploying the exact same version a second time should fail with 409
    dup = client.post(
        "/models/fraud-detector/versions/v1/deploy",
        json={"replicas": 1, "resources": {"memory": "512Mi", "cpu": "0.5"}},
    )
    assert dup.status_code == 409

    # Check the health endpoint and active deployments list
    health = client.get("/models/fraud-detector/versions/v1/health").json()
    assert health["healthy"] is True
    assert health["replicas_total"] == 1
    assert health["replicas_healthy"] == 1

    deps = client.get("/deployments?status=running").json()
    assert len(deps) == 1
    assert deps[0]["deployment_id"] == dep_id

    # Scale up from 1 to 2 replicas, then scale back down to 1
    scaled = client.patch(f"/deployments/{dep_id}", json={"replicas": 2, "resources": {"memory": "1Gi", "cpu": "1.0"}})
    assert scaled.status_code == 200
    assert scaled.json()["replicas"] == 2
    assert scaled.json()["health"]["replicas_healthy"] == 2

    scaled_down = client.patch(f"/deployments/{dep_id}", json={"replicas": 1})
    assert scaled_down.status_code == 200
    assert scaled_down.json()["replicas"] == 1

    # Can't delete while running (409), but deleting works right after we undeploy
    assert client.delete("/models/fraud-detector/versions/v1").status_code == 409
    undep = client.delete("/models/fraud-detector/versions/v1/deploy")
    assert undep.status_code == 200
    assert undep.json()["status"] == "stopped"
    assert client.delete("/models/fraud-detector/versions/v1").status_code == 200


def test_corrupted_artifact_marks_deployment_failed_without_crashing_server(client: TestClient) -> None:
    # Upload garbage bytes as a .pkl file and try to deploy it
    assert upload_bytes(client, "broken-model", "v1", "sklearn", "corrupt.pkl", b"not-a-valid-pickle").status_code == 201

    dep = client.post(
        "/models/broken-model/versions/v1/deploy",
        json={"replicas": 1, "resources": {"memory": "512Mi", "cpu": "0.5"}},
    )
    assert dep.status_code == 422
    assert dep.json()["error"]["code"] == "MODEL_LOAD_FAILED"

    # The main server stays healthy and saves the failed deployment record with its error
    assert client.get("/health").status_code == 200
    deps = client.get("/deployments?model_name=broken-model").json()
    assert len(deps) == 1
    assert deps[0]["status"] == "failed"
    assert deps[0]["error_message"]


def test_invalid_deployment_resource_configs(client: TestClient) -> None:
    upload_sample(client, "bad-cfg", "v1", "custom_model.py", "custom")

    # Reject bad memory unit ("512MB"), zero CPU, and zero replicas
    assert client.post("/models/bad-cfg/versions/v1/deploy", json={"replicas": 1, "resources": {"memory": "512MB", "cpu": "0.5"}}).status_code == 422
    assert client.post("/models/bad-cfg/versions/v1/deploy", json={"replicas": 1, "resources": {"memory": "512Mi", "cpu": "0"}}).status_code == 422
    assert client.post("/models/bad-cfg/versions/v1/deploy", json={"replicas": 0, "resources": {"memory": "512Mi", "cpu": "0.5"}}).status_code == 422


def test_process_isolation_survives_worker_crash(client: TestClient) -> None:
    # Deploy a normal model alongside a custom model that kills its own process on command
    upload_sample(client, "stable-model", "v1", "custom_model.py", "custom")
    assert client.post("/models/stable-model/versions/v1/deploy", json={"replicas": 1}).status_code == 200

    crasher_code = b"""
import os
class Model:
    def predict(self, inputs):
        if inputs.get("crash"):
            os._exit(1)
        return {"prediction": 1, "confidence": 0.99}
"""
    assert upload_bytes(client, "crasher", "v1", "custom", "crasher.py", crasher_code).status_code == 201
    assert client.post("/models/crasher/versions/v1/deploy", json={"replicas": 1}).status_code == 200

    # Force the crasher worker process to exit abruptly and confirm we catch it cleanly
    mgr = client.app.state.serving_manager
    try:
        mgr.predict("crasher", "v1", {"crash": True})
        assert False, "Expected AppError from crashed worker"
    except Exception as exc:
        assert getattr(exc, "code", None) == "INFERENCE_ERROR"

    # The other model keeps working normally because it runs in its own process
    out, _ = mgr.predict("stable-model", "v1", {"features": [1.0, 0.2, 0.5, 0.1, 0.3]})
    assert out["prediction"] in (0, 1)

    # Wait briefly for the dead worker to automatically respawn, then predict again
    deadline = time.time() + 10.0
    while time.time() < deadline and not mgr.health("crasher", "v1")["healthy"]:
        time.sleep(0.2)
    assert mgr.health("crasher", "v1")["healthy"] is True
    out2, _ = mgr.predict("crasher", "v1", {"crash": False})
    assert out2["prediction"] == 1
