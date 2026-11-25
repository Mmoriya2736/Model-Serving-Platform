import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.db.models import Deployment, ModelVersion
from app.main import create_app

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_models"

# Standard 10-feature input schema matching the assignment PDF example
CLASSIFIER_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "features": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 10,
            "maxItems": 10,
        }
    },
    "required": ["features"],
}

CLASSIFIER_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction": {"type": "number"},
        "confidence": {"type": "number"},
    },
}


def _metadata_json(**overrides) -> str:
    # Build a valid metadata JSON string and let each test override specific fields
    payload = {
        "framework": "sklearn",
        "input_schema": CLASSIFIER_INPUT_SCHEMA,
        "output_schema": CLASSIFIER_OUTPUT_SCHEMA,
        "description": "Fraud detection classifier",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_list_models_empty_state(client: TestClient) -> None:
    # Fresh registry should return an empty list, not an error
    res = client.get("/models")
    assert res.status_code == 200
    assert res.json() == []


def test_upload_list_download_and_delete_model_version(client: TestClient) -> None:
    artifact_bytes = (SAMPLE_DIR / "fraud_classifier_v1.pkl").read_bytes()

    # Upload without a version name and confirm auto-versioning picks "v1"
    res = client.post(
        "/models/fraud-detector/versions",
        data={"metadata": _metadata_json()},
        files={"artifact": ("fraud_classifier_v1.pkl", artifact_bytes, "application/octet-stream")},
    )
    assert res.status_code == 201
    body = res.json()
    assert body["model"] == "fraud-detector"
    assert body["version"] == "v1"
    assert body["status"] == "registered"
    assert body["artifact_size_bytes"] == len(artifact_bytes)
    assert "created_at" in body

    # Upload a second version with an explicit version name "v2"
    res2 = client.post(
        "/models/fraud-detector/versions",
        data={"metadata": _metadata_json(version="v2", description="v2 RF model")},
        files={"artifact": ("fraud_classifier_v2.pkl", artifact_bytes, "application/octet-stream")},
    )
    assert res2.status_code == 201
    assert res2.json()["version"] == "v2"

    # List all models and all versions under "fraud-detector"
    models = client.get("/models").json()
    assert len(models) == 1
    assert models[0]["name"] == "fraud-detector"
    assert models[0]["versions_count"] == 2
    assert models[0]["latest_version"] == "v2"

    versions = client.get("/models/fraud-detector/versions").json()
    assert [v["version"] for v in versions] == ["v1", "v2"]

    # Fetch v1 details and verify the saved artifact file exists on disk
    detail = client.get("/models/fraud-detector/versions/v1").json()
    assert detail["framework"] == "sklearn"
    assert detail["input_schema"] == CLASSIFIER_INPUT_SCHEMA
    stored_path = Path(detail["artifact_location"])
    assert stored_path.is_file()

    # Download v1 and confirm the bytes match the original file exactly
    dl = client.get("/models/fraud-detector/versions/v1/download")
    assert dl.status_code == 200
    assert dl.content == artifact_bytes

    # Delete undeployed v1 and confirm both the DB row and disk file are gone
    del_res = client.delete("/models/fraud-detector/versions/v1")
    assert del_res.status_code == 200
    assert del_res.json() == {"model": "fraud-detector", "version": "v1", "status": "deleted"}
    assert not stored_path.exists()
    assert client.get("/models/fraud-detector/versions/v1").status_code == 404


def test_upload_custom_model_and_reference_url(client: TestClient) -> None:
    # Upload a custom .py model file
    custom_bytes = (SAMPLE_DIR / "custom_model.py").read_bytes()
    res = client.post(
        "/models/custom-scorer/versions",
        data={"metadata": _metadata_json(framework="custom", version="v1")},
        files={"artifact": ("custom_model.py", custom_bytes, "text/x-python")},
    )
    assert res.status_code == 201
    assert res.json()["version"] == "v1"

    # Register a model by passing a local file path reference instead of uploading bytes
    ref_path = str((SAMPLE_DIR / "house_regressor_v1.pkl").resolve())
    res_ref = client.post(
        "/models/house-price/versions",
        data={"metadata": _metadata_json(framework="sklearn", version="v1", artifact_url=ref_path)},
    )
    assert res_ref.status_code == 201
    assert res_ref.json()["artifact_size_bytes"] > 0


def test_registry_validations_and_error_cases(client: TestClient, tmp_path: Path) -> None:
    artifact_bytes = (SAMPLE_DIR / "fraud_classifier_v1.pkl").read_bytes()

    # Uploading the same (model, version) twice should fail with 409 Conflict
    r1 = client.post(
        "/models/dup-model/versions",
        data={"metadata": _metadata_json(version="v1")},
        files={"artifact": ("m.pkl", artifact_bytes, "application/octet-stream")},
    )
    assert r1.status_code == 201
    r2 = client.post(
        "/models/dup-model/versions",
        data={"metadata": _metadata_json(version="v1")},
        files={"artifact": ("m.pkl", artifact_bytes, "application/octet-stream")},
    )
    assert r2.status_code == 409
    assert r2.json()["error"]["code"] == "VERSION_ALREADY_EXISTS"

    # Broken JSON metadata -> 422
    r_bad_json = client.post(
        "/models/bad-meta/versions",
        data={"metadata": "{not-json"},
        files={"artifact": ("m.pkl", artifact_bytes, "application/octet-stream")},
    )
    assert r_bad_json.status_code == 422
    assert r_bad_json.json()["error"]["code"] == "INVALID_METADATA"

    # Invalid JSON Schema definition -> 422
    r_bad_schema = client.post(
        "/models/bad-schema/versions",
        data={"metadata": _metadata_json(input_schema={"type": "not-a-valid-type"})},
        files={"artifact": ("m.pkl", artifact_bytes, "application/octet-stream")},
    )
    assert r_bad_schema.status_code == 422
    assert r_bad_schema.json()["error"]["code"] == "INVALID_METADATA"

    # Unsupported framework name -> 400
    r_bad_fw = client.post(
        "/models/bad-fw/versions",
        data={"metadata": _metadata_json(framework="tensorflow")},
        files={"artifact": ("m.pkl", artifact_bytes, "application/octet-stream")},
    )
    assert r_bad_fw.status_code == 400
    assert r_bad_fw.json()["error"]["code"] == "UNSUPPORTED_FRAMEWORK"

    # File extension doesn't match declared framework -> 400
    r_ext = client.post(
        "/models/bad-ext/versions",
        data={"metadata": _metadata_json(framework="onnx")},
        files={"artifact": ("m.pkl", artifact_bytes, "application/octet-stream")},
    )
    assert r_ext.status_code == 400
    assert r_ext.json()["error"]["code"] == "UNSUPPORTED_FRAMEWORK"

    # Neither file upload nor artifact_url provided -> 400
    r_missing = client.post(
        "/models/no-file/versions",
        data={"metadata": _metadata_json()},
    )
    assert r_missing.status_code == 400
    assert r_missing.json()["error"]["code"] == "MISSING_ARTIFACT"

    # Empty 0-byte upload -> 400
    r_empty = client.post(
        "/models/empty-file/versions",
        data={"metadata": _metadata_json()},
        files={"artifact": ("m.pkl", b"", "application/octet-stream")},
    )
    assert r_empty.status_code == 400
    assert r_empty.json()["error"]["code"] == "MISSING_ARTIFACT"

    # Path traversal in model name ("..") -> 400
    r_bad_id = client.post(
        "/models/bad..name/versions",
        data={"metadata": _metadata_json()},
        files={"artifact": ("m.pkl", artifact_bytes, "application/octet-stream")},
    )
    assert r_bad_id.status_code == 400
    assert r_bad_id.json()["error"]["code"] == "INVALID_IDENTIFIER"

    # Looking up a model that doesn't exist -> 404
    assert client.get("/models/does-not-exist/versions").status_code == 404

    # File larger than max_upload_size_mb -> 413 Payload Too Large
    tiny_settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'tiny.db').as_posix()}",
        artifact_storage_dir=tmp_path / "tiny_artifacts",
        max_upload_size_mb=0,
        log_level="WARNING",
    )
    with TestClient(create_app(tiny_settings)) as tiny_client:
        r_big = tiny_client.post(
            "/models/big/versions",
            data={"metadata": _metadata_json()},
            files={"artifact": ("m.pkl", b"1234", "application/octet-stream")},
        )
        assert r_big.status_code == 413
        assert r_big.json()["error"]["code"] == "ARTIFACT_TOO_LARGE"


def test_cannot_delete_deployed_model_version(client: TestClient) -> None:
    artifact_bytes = (SAMPLE_DIR / "fraud_classifier_v1.pkl").read_bytes()
    res = client.post(
        "/models/live-model/versions",
        data={"metadata": _metadata_json(version="v1")},
        files={"artifact": ("m.pkl", artifact_bytes, "application/octet-stream")},
    )
    assert res.status_code == 201

    # Insert an active "running" deployment directly in SQLite to simulate a live model
    with client.app.state.session_factory() as db:
        mv = db.query(ModelVersion).filter_by(model_name="live-model", version="v1").one()
        mv.status = "deployed"
        db.add(
            Deployment(
                deployment_id="dep_test_1",
                model_version_id=mv.id,
                model_name="live-model",
                version="v1",
                replicas=1,
                resources={"memory": "512Mi", "cpu": "0.5"},
                status="running",
                endpoint="/predict/live-model",
            )
        )
        db.commit()

    # Trying to delete a live version must be blocked with 409
    del_res = client.delete("/models/live-model/versions/v1")
    assert del_res.status_code == 409
    assert del_res.json()["error"]["code"] == "VERSION_DEPLOYED"
