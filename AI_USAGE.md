# AI Tool Usage

## Tools I Used
- **Antigravity (Gemini-based coding assistant)** for specification extraction, architecture planning, boilerplate generation, and test authoring.

## What Helped Most
1. **Milestone-driven decomposition and API contract verification:** Breaking the 15-page specification into 7 testable milestones allowed verifying each subsystem (registry, isolated runtime, inference, routing, metrics) independently before wiring the next layer.
2. **Rapid scaffolding of FastAPI routes, SQLAlchemy 2.x models, and thorough edge-case tests:** Generating Pydantic v2 schemas, structured error envelopes, and boundary tests (path traversal, empty batches, duplicate versions, crash isolation, drift detection) saved significant repetitive typing.

## What I Had to Fix
1. **PDF API specification discrepancy (Page 5 vs. Page 7):** The assignment showed two slightly different inference response shapes (`prediction` + nested `metadata` on Page 5 vs. `outputs` + top-level `latency_ms` on Page 7). I explicitly reconciled this before coding by aligning with Page 7's `POST /predict/{model_name}` contract while retaining top-level `request_id` for `InferenceLog` traceability.
2. **Windows vs. Linux runtime nuances:** On Windows with Python 3.14, `onnxruntime`/`skl2onnx` wheels are not yet available, `resource` does not exist, and `fork()` multiprocessing is unavailable. I pinned Docker to `python:3.12-slim`, used `multiprocessing.get_context("spawn")` everywhere, and replaced an initial virtual-memory limit (`RLIMIT_DATA`) with a physical RAM check (`ru_maxrss`) after Docker tests showed that Linux virtual-memory caps block shared libraries like OpenBLAS and ONNXRuntime from loading.
3. **Docker test coverage & CLI documentation gaps:** Because local Windows runs skipped ONNX tests, I enforced full test execution inside the `python:3.12-slim` container, fixed the `README.md` Docker test command for Windows PowerShell (`--entrypoint ""` argument stripping -> `--entrypoint sh` + `python -m pytest`), and added end-to-end stress tests covering high concurrency, canary auto-rollback, and orphan worker cleanup.

## What AI Struggled With
- **Environment-specific operational edge cases:** Distinguishing between an application bug and a Windows sandbox loopback restriction when `TestClient` hung during startup required inspecting thread stack traces (`faulthandler`) rather than guessing at FastAPI lifespan code.
- **Right-sizing architecture scope:** AI tends to over-engineer with external brokers (Redis/Celery/Kubernetes) unless explicitly constrained to a self-contained, reviewable architecture that still demonstrates real process isolation, backpressure, and canary auto-rollback.
