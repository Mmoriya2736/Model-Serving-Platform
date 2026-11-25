from collections import Counter

from fastapi.testclient import TestClient

from tests.helpers import SAMPLE_DIR, upload_and_deploy_bytes, upload_and_deploy_sample


def test_dark_launch_weighted_traffic_split_sticky_and_rollback(client: TestClient) -> None:
    # Deploy v1 first, then v2
    upload_and_deploy_sample(client, "fraud-ab", "v1", "fraud_classifier_v1.pkl", "sklearn")
    upload_and_deploy_sample(client, "fraud-ab", "v2", "fraud_classifier_v2.pkl", "sklearn")

    # Deploying v2 should NOT steal traffic from v1 yet (safe dark launch)
    routing = client.get("/models/fraud-ab/routing").json()
    assert routing["rules"] == [{"version": "v1", "weight": 100}]

    # Set an 80/20 traffic split and send 250 predictions to check the ratio
    put_res = client.put(
        "/models/fraud-ab/routing",
        json={"rules": [{"version": "v1", "weight": 80}, {"version": "v2", "weight": 20}]},
    )
    assert put_res.status_code == 200

    counts: Counter[str] = Counter()
    payload = {"inputs": {"features": [0.1] * 10}}
    for _ in range(250):
        resp = client.post("/predict/fraud-ab", json=payload)
        assert resp.status_code == 200
        counts[resp.json()["version"]] += 1

    # Check that observed counts land cleanly around 80% v1 / 20% v2
    assert 160 <= counts["v1"] <= 230
    assert 20 <= counts["v2"] <= 90

    # Sending the same X-Session-ID header 15 times should always hit the exact same version
    sticky_versions = {
        client.post("/predict/fraud-ab", json=payload, headers={"X-Session-ID": "sess-42"}).json()["version"]
        for _ in range(15)
    }
    assert len(sticky_versions) == 1

    # One-click rollback restores the previous 100% v1 routing
    rb = client.post("/models/fraud-ab/rollback")
    assert rb.status_code == 200
    assert rb.json()["rules"] == [{"version": "v1", "weight": 100}]


def test_routing_validation_errors(client: TestClient) -> None:
    upload_and_deploy_sample(client, "route-val", "v1", "fraud_classifier_v1.pkl", "sklearn")

    # Reject weights that don't add to 100, empty rules, undeployed versions, and empty rollback history
    assert client.put("/models/route-val/routing", json={"rules": [{"version": "v1", "weight": 50}]}).status_code == 422
    assert client.put("/models/route-val/routing", json={"rules": []}).status_code == 422
    assert client.put("/models/route-val/routing", json={"rules": [{"version": "v1", "weight": 50}, {"version": "v9", "weight": 50}]}).status_code == 400
    assert client.post("/models/route-val/rollback").status_code == 409


def test_canary_rollout_triggers_automatic_rollback_on_errors(client: TestClient) -> None:
    # Deploy a healthy v1 and a broken v2 that always throws an exception
    good_bytes = (SAMPLE_DIR / "custom_model.py").read_bytes()
    bad_bytes = b"""
class Model:
    def predict(self, inputs):
        raise RuntimeError("Simulated canary failure")
"""
    upload_and_deploy_bytes(client, "canary-svc", "v1", "custom_model.py", good_bytes, "custom")
    upload_and_deploy_bytes(client, "canary-svc", "v2", "bad_canary.py", bad_bytes, "custom")

    # Start a 50% canary rollout that auto-reverts if >20% of >=3 requests fail
    rollout = client.post(
        "/models/canary-svc/rollout",
        json={"target_version": "v2", "weight": 50, "max_error_rate": 0.2, "min_requests": 3},
    )
    assert rollout.status_code == 200
    assert rollout.json()["canary"]["target_version"] == "v2"

    # Send 3 requests to v2 (all fail) and confirm routing automatically jumps back to 100% v1
    for _ in range(3):
        err_resp = client.post(
            "/predict/canary-svc",
            json={"inputs": {"features": [1.0, 0.2, 0.3, 0.4, 0.5]}, "version": "v2"},
        )
        assert err_resp.status_code == 500

    after = client.get("/models/canary-svc/routing").json()
    assert after["rules"] == [{"version": "v1", "weight": 100}]
    assert after["canary"] is None
