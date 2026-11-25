import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import httpx

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_models"


def features_schema(num_features: int | None = None) -> dict[str, Any]:
    # Build a simple JSON Schema expecting a "features" array of numbers
    features_prop: dict[str, Any] = {"type": "array", "items": {"type": "number"}}
    if num_features is not None:
        # Lock the array to an exact length when requested
        features_prop["minItems"] = num_features
        features_prop["maxItems"] = num_features
    return {
        "type": "object",
        "properties": {"features": features_prop},
        "required": ["features"],
    }


def upload_bytes(
    client: TestClient,
    model_name: str,
    version: str,
    framework: str,
    filename: str,
    raw: bytes,
    input_schema: dict[str, Any] | None = None,
) -> httpx.Response:
    # Upload raw model file bytes plus JSON metadata to the registry endpoint
    meta = {
        "framework": framework,
        "version": version,
        "input_schema": input_schema or {"type": "object"},
        "output_schema": {"type": "object"},
    }
    return client.post(
        f"/models/{model_name}/versions",
        data={"metadata": json.dumps(meta)},
        files={"artifact": (filename, raw, "application/octet-stream")},
    )


def upload_sample(
    client: TestClient,
    model_name: str,
    version: str,
    filename: str,
    framework: str,
    num_features: int | None = None,
) -> None:
    # Read one of the sample_models files and register it, asserting 201 Created
    raw = (SAMPLE_DIR / filename).read_bytes()
    schema = features_schema(num_features) if num_features is not None else {"type": "object"}
    res = upload_bytes(client, model_name, version, framework, filename, raw, input_schema=schema)
    assert res.status_code == 201


def upload_and_deploy_bytes(
    client: TestClient,
    model_name: str,
    version: str,
    filename: str,
    raw: bytes,
    framework: str,
    num_features: int | None = None,
    replicas: int = 1,
) -> None:
    # Upload raw model bytes and immediately deploy them with default memory/CPU limits
    schema = features_schema(num_features) if num_features is not None else {"type": "object"}
    assert upload_bytes(client, model_name, version, framework, filename, raw, input_schema=schema).status_code == 201
    dep = client.post(
        f"/models/{model_name}/versions/{version}/deploy",
        json={"replicas": replicas, "resources": {"memory": "512Mi", "cpu": "0.5"}},
    )
    assert dep.status_code == 200


def upload_and_deploy_sample(
    client: TestClient,
    model_name: str,
    version: str,
    filename: str,
    framework: str,
    num_features: int | None = None,
    replicas: int = 1,
) -> None:
    # Shortcut for uploading and deploying one of the built-in sample models
    raw = (SAMPLE_DIR / filename).read_bytes()
    upload_and_deploy_bytes(client, model_name, version, filename, raw, framework, num_features, replicas)
