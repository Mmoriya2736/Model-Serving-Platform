from pathlib import Path
import pickle

from fastapi.testclient import TestClient
import numpy as np
from sqlalchemy import inspect, text

from sample_models.custom_model import Model as CustomSampleModel
from sample_models.generate_models import generate_all


def test_health_endpoint_returns_healthy_and_request_id(client: TestClient) -> None:
    # Hit /health and check that we get 200 OK plus an auto-generated request ID
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["request_id"].startswith("req_")
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_health_preserves_caller_request_id(client: TestClient) -> None:
    # If the caller sends their own X-Request-ID, echo the exact same ID back
    response = client.get("/health", headers={"X-Request-ID": "req_custom_42"})
    assert response.status_code == 200
    assert response.json()["request_id"] == "req_custom_42"
    assert response.headers["X-Request-ID"] == "req_custom_42"


def test_database_tables_and_wal_mode_initialized(client: TestClient) -> None:
    # Confirm all 6 tables were created automatically on app startup
    engine = client.app.state.engine
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    expected = {
        "models",
        "model_versions",
        "deployments",
        "routing_configs",
        "inference_logs",
        "async_jobs",
    }
    assert expected.issubset(tables)

    # Confirm SQLite WAL mode and foreign key checks are both turned on
    with engine.connect() as conn:
        journal_mode = conn.execute(text("PRAGMA journal_mode;")).scalar()
        foreign_keys = conn.execute(text("PRAGMA foreign_keys;")).scalar()
    assert str(journal_mode).lower() == "wal"
    assert foreign_keys == 1


def test_sample_models_generate_and_predict(tmp_path: Path) -> None:
    # Run the sample model generator into a temp folder and verify all .pkl files exist
    generated = generate_all(output_dir=tmp_path)
    assert "fraud_classifier_v1" in generated
    assert "fraud_classifier_v2" in generated
    assert "house_regressor_v1" in generated

    # Load the 10-feature classifier and check it returns class probabilities
    with generated["fraud_classifier_v1"].open("rb") as f:
        clf = pickle.load(f)
    probs = clf.predict_proba(np.zeros((1, 10)))
    assert probs.shape == (1, 2)

    # Load the 5-feature regressor and check it returns a predicted value
    with generated["house_regressor_v1"].open("rb") as f:
        reg = pickle.load(f)
    pred = reg.predict(np.zeros((1, 5)))
    assert pred.shape == (1,)

    # Check the custom Python model class directly
    custom = CustomSampleModel()
    out = custom.predict({"features": [1.0, 0.2, 0.8, 0.5, 0.9]})
    assert out["prediction"] in (0, 1)
    assert 0.0 <= out["confidence"] <= 1.0
