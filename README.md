# Model Serving Platform

A production-oriented ML model serving platform built with **FastAPI**, **SQLAlchemy (SQLite WAL)**, and **process-isolated worker replicas**. It supports versioned model registration, multi-format model loading, schema-validated synchronous/batch/async inference, weighted A/B traffic splitting, canary rollouts with automatic rollback, and latency/error/confidence/drift observability.

---

## 1. How to Run the Project

### Option A: Docker Compose (Recommended — includes full `.pkl`, `.onnx`, and `.py` support)

```bash
docker compose up --build -d
```

The API will be available at `http://localhost:8000` (interactive Swagger docs at `http://localhost:8000/docs`).

To verify health:
```bash
curl http://localhost:8000/health
```

To stop:
```bash
docker compose down
```

### Option B: Local Python (3.11+)

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements-dev.txt
python sample_models/generate_models.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> **Note on Python 3.14:** `onnxruntime` and `skl2onnx` are conditionally installed for `python_version < "3.14"` in `requirements.txt`. On Python 3.14 locally, `.pkl` and custom `.py` models run natively; use the Docker image (`python:3.12-slim`) for full ONNX runtime execution.

---

## 2. How to Run Tests

### Locally
```bash
pytest -v
```

### Inside Docker (runs the full suite including ONNX inference tests)
```bash
docker compose run --rm --entrypoint sh model-serving-platform -c "pip install --no-cache-dir -q pytest httpx pytest-asyncio && python -m pytest -v"
```

The test suite covers all required areas:
- `tests/test_health.py` — Health endpoint, request ID propagation, SQLite WAL initialization, sample model generation
- `tests/test_registry.py` — Upload, auto/explicit versioning, listing, byte-exact download, schema validation, safe deletion blocking on deployed models
- `tests/test_deployment.py` — Deploy, readiness health checks, replica scaling, corrupted artifact failure isolation, subprocess crash isolation (`os._exit(1)`) & worker respawn
- `tests/test_inference.py` — Classifier, regressor, custom model inference, batch prediction, async job polling, strict JSON Schema validation, SHA-256 `input_hash` privacy
- `tests/test_routing.py` — Dark launch, weighted traffic splitting, sticky sessions (`X-Session-ID`), instant rollback, canary rollout + automatic rollback on error-rate breach
- `tests/test_metrics.py` — Empty metrics, `p50`/`p95`/`p99` latencies, throughput RPS, error rate, per-version breakdown, confidence distribution buckets, input drift detection
- `tests/test_onnx_and_resilience.py` — ONNX end-to-end serving, inference timeout (`504`) + recovery, backpressure queue overflow (`429`)
- `tests/test_advanced_e2e.py` — End-to-end multi-format lifecycle, 60-request concurrency & backpressure stress, canary auto-rollback, boundary validation, and clean worker process shutdown

---

## 3. Supported Model Formats

| Format | Extension | Framework Key | Why Supported |
| :--- | :--- | :--- | :--- |
| **Scikit-learn (Pickle/Joblib)** | `.pkl`, `.joblib` | `sklearn` | Ubiquitous baseline for tabular ML classifiers and regressors; fast vectorized batch execution via NumPy. |
| **ONNX** | `.onnx` | `onnx` | Framework-agnostic, portable graph format exported from PyTorch, TensorFlow, LightGBM, XGBoost, or scikit-learn without pickle deserialization risk. |
| **Custom Python Class** | `.py` | `custom` | Escape hatch for custom pre/post-processing, business rules, or ensemble logic implementing `predict(self, inputs: dict) -> dict`. |

Sample models are included in `sample_models/` (`generate_models.py` produces `fraud_classifier_v1.pkl`, `fraud_classifier_v2.pkl`, `house_regressor_v1.pkl`, and `fraud_classifier_v1.onnx`, alongside `custom_model.py`).

---

## 4. How to Deploy a Model (Step by Step)

### Step 1: Upload Model v1
```bash
curl -X POST http://localhost:8000/models/fraud-detector/versions \
  -F "artifact=@sample_models/fraud_classifier_v1.pkl" \
  -F 'metadata={
    "framework": "sklearn",
    "version": "v1",
    "description": "LogisticRegression fraud detector v1",
    "input_schema": {
      "type": "object",
      "properties": {
        "features": {
          "type": "array",
          "items": {"type": "number"},
          "minItems": 10,
          "maxItems": 10
        }
      },
      "required": ["features"]
    },
    "output_schema": {
      "type": "object",
      "properties": {
        "prediction": {"type": "number"},
        "confidence": {"type": "number"}
      }
    }
  }'
```

### Step 2: Deploy Model v1 (with 2 isolated replicas)
```bash
curl -X POST http://localhost:8000/models/fraud-detector/versions/v1/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "replicas": 2,
    "resources": {
      "memory": "512Mi",
      "cpu": "0.5"
    }
  }'
```

*(The first deployed version automatically receives 100% of traffic. Subsequent versions dark-launch at 0% until explicitly routed.)*

### Step 3: Upload & Deploy v2, then Split Traffic (A/B Test)
```bash
curl -X POST http://localhost:8000/models/fraud-detector/versions \
  -F "artifact=@sample_models/fraud_classifier_v2.pkl" \
  -F 'metadata={"framework":"sklearn","version":"v2","input_schema":{"type":"object","properties":{"features":{"type":"array","items":{"type":"number"},"minItems":10,"maxItems":10}},"required":["features"]},"output_schema":{"type":"object"}}'

curl -X POST http://localhost:8000/models/fraud-detector/versions/v2/deploy \
  -H "Content-Type: application/json" \
  -d '{"replicas": 1, "resources": {"memory": "512Mi", "cpu": "0.5"}}'

curl -X PUT http://localhost:8000/models/fraud-detector/routing \
  -H "Content-Type: application/json" \
  -d '{
    "rules": [
      {"version": "v1", "weight": 90},
      {"version": "v2", "weight": 10}
    ]
  }'
```

---

## 5. Example Inference & Observability Calls

### Single Prediction (routed via traffic split)
```bash
curl -X POST http://localhost:8000/predict/fraud-detector \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "features": [0.1, 0.5, 0.3, -0.2, 0.8, 0.1, 0.4, -0.7, 0.9, 0.0]
    }
  }'
```

### Single Prediction (pinned version + sticky session header)
```bash
curl -X POST http://localhost:8000/predict/fraud-detector \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: user-42" \
  -d '{
    "inputs": {"features": [0.1, 0.5, 0.3, -0.2, 0.8, 0.1, 0.4, -0.7, 0.9, 0.0]},
    "version": "v2"
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/fraud-detector/batch \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {"features": [0.1, 0.5, 0.3, -0.2, 0.8, 0.1, 0.4, -0.7, 0.9, 0.0]},
      {"features": [-0.4, 0.2, 0.9, 0.1, -0.3, 0.7, 0.0, 0.5, -0.1, 0.2]}
    ]
  }'
```

### Async Prediction & Job Polling
```bash
curl -X POST http://localhost:8000/predict/fraud-detector/async \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"features": [0.1, 0.5, 0.3, -0.2, 0.8, 0.1, 0.4, -0.7, 0.9, 0.0]}}'

curl http://localhost:8000/jobs/<job_id>
```

### Gradual Canary Rollout & Instant Rollback
```bash
# Send 20% traffic to v2; auto-rollback to v1 if error rate > 5% after 10 requests
curl -X POST http://localhost:8000/models/fraud-detector/rollout \
  -H "Content-Type: application/json" \
  -d '{"target_version": "v2", "weight": 20, "max_error_rate": 0.05, "min_requests": 10}'

# Manual instant rollback to previous routing configuration
curl -X POST http://localhost:8000/models/fraud-detector/rollback
```

### Metrics & Drift Summary
```bash
curl "http://localhost:8000/models/fraud-detector/metrics?period=1h"
```
