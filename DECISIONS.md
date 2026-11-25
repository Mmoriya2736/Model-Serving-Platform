# Design Decisions

## 1. Model Loading Strategy

**Approach chosen:** Hybrid (Eager on deploy + Startup warm-reload + LRU capacity cap)

**Why:**
- Pure lazy loading shifts deserialization and runtime initialization latency onto the first production inference request, causing unpredictable p99 spikes and discovering broken artifacts only after traffic arrives.
- Pure eager loading of every registered version wastes memory on archived or experimental versions.
- Loading eagerly during `POST /models/{model_name}/versions/{version}/deploy` (and warm-reloading only `status="running"` deployments on server restart) guarantees that every running model has passed a readiness probe before any user request hits it, while keeping unregistered/stopped versions off memory.

**Memory management:**
- `ServingManager` stores active `ReplicaPool` instances in a thread-safe `OrderedDict[(model_name, version), ReplicaPool]` bounded by `MAX_LOADED_MODELS` (default `10`).
- Every inference call updates recency via `move_to_end()`. When a new deployment exceeds capacity, the least-recently-used pool is gracefully stopped and evicted before spawning new workers.
- If a model fails to load (corrupted pickle/ONNX, syntax error in custom code, or failed health check), partial worker processes are immediately terminated, the `Deployment` row is marked `status="failed"` with `error_message`, and all previously running models remain unaffected.

---

## 2. Request Routing

**Algorithm:** Weighted random selection via precomputed cumulative weights ($O(\log K)$ via `bisect_right`), with optional deterministic sticky sessions when an `X-Session-ID` header is provided (`sha256(f"{model_name}:{session_id}") % total_weight`).

**Implementation:**
- `RoutingConfig` persists active `rules` (`[{"version": "v1", "weight": 90}, ...]`), `previous_rules` (for O(1) instant rollback), and optional `canary_config` in SQLite.
- `RoutingService.select_version()` builds a cumulative weight table over the intersection of configured rules and currently `running` deployments, ensuring stale or failed versions never receive traffic.
- First deployment of a model automatically initializes `100%` traffic to that version; deploying a subsequent version (`v2`) performs a safe **dark launch** (preserving `v1` at `100%`) until routing or rollout is explicitly updated.

**Edge cases:**
- **Undeployed versions in routing rules:** Rejected up front on `PUT /models/{model_name}/routing` (`400`) and defensively filtered out at inference time if a deployment stops unexpectedly.
- **Undeploying a routed version:** Automatically removes that version from `RoutingConfig` and re-normalizes remaining running weights to sum to `100` (saving `previous_rules` first).
- **Invalid weights / duplicates:** Weights must be non-negative integers summing to exactly `100` with unique version names (`422`).
- **Response schema consistency:** Where the assignment PDF differed slightly between Page 5 (`prediction` + nested `metadata`) and Page 7 (`outputs` + top-level `latency_ms`), the platform follows Page 7's explicit `POST /predict/{model_name}` contract and adds top-level `request_id` for end-to-end traceability with `InferenceLog`.

---

## 3. Model Isolation

**Approach chosen:** OS Process isolation (`multiprocessing.get_context("spawn")`) with one child process per replica communicating over a duplex `Pipe`.

**Failure handling:**
- Because each model replica runs in its own OS process rather than a Python thread, a crashing model (unhandled exception, C-extension fault, or even `os._exit(1)`) or hung inference cannot crash the FastAPI server or impact other deployed models.
- `ReplicaWorker` detects `EOFError` / `BrokenPipeError` or `TimeoutError` (`INFERENCE_TIMEOUT_SECONDS`), force-terminates the dead/hung child process, returns `500` or `504` for that request, logs the failure to `InferenceLog`, and spawns a replacement worker asynchronously so subsequent requests recover automatically.
- Using `spawn` (rather than `fork`) guarantees cross-platform safety on both Linux/Docker and Windows and avoids mutex deadlocks with SQLite, OpenMP, and ONNXRuntime thread pools.

**Resource limits & Security Boundary:**
- Deployments validate `resources.memory` (`^\d+(Mi|Gi)$`) and `resources.cpu` (`0 < cpu <= 16`).
- On POSIX/Docker hosts, each worker enforces the deployment's `resources.memory` budget by checking actual physical RAM (`ru_maxrss`) after model load instead of OS virtual-memory limits (`RLIMIT_DATA`), which on Linux block shared ML libraries (`OpenBLAS`/`ONNXRuntime`) from loading. On Windows dev hosts, the POSIX check is cleanly guarded and skipped.
- **Scaling & Backpressure:** Each deployment runs a `ReplicaPool` of `N` worker processes with least-in-flight + round-robin dispatch and a bounded pending request counter (`MAX_QUEUE_DEPTH_PER_DEPLOYMENT`). Excess requests fail fast with `429 Too Many Requests` instead of causing unbounded memory growth.
- **Pickle / Custom `.py` Trust Boundary:** Deserializing `.pkl` files and executing custom `.py` modules can run arbitrary Python code inside the worker process. In production, this should be restricted to signed artifacts from trusted CI pipelines, gated by `ENABLE_CUSTOM_PYTHON_MODELS`, and isolated further using per-model containers/microVMs (see Section 5).

---

## 4. Deployment Safety

**Health checks:**
- Every new deployment starts in `status="deploying"`, spawns its worker processes, loads the model, and runs a `health_check()` round-trip over IPC before transitioning to `status="running"`.
- A `GET /models/{model_name}/versions/{version}/health` endpoint exposes live replica health (`replicas_total`, `replicas_healthy`, `queue_depth`).

**Rollout strategy:**
- New versions dark-launch at `0%` traffic.
- `POST /models/{model_name}/rollout` shifts a specified percentage (e.g., `10%` -> `50%` -> `100%`) to `target_version`, proportionally scaling down existing versions, and attaches a canary guard (`max_error_rate`, `min_requests`).

**Rollback:**
- **Manual instant rollback:** `POST /models/{model_name}/rollback` immediately restores `previous_rules` (or switches `100%` to an explicit `target_version`).
- **Automatic rollback on errors:** Whenever a canary version logs an inference error, `RoutingService.evaluate_canary_auto_rollback()` checks the canary's error rate since rollout start; once `requests >= min_requests` and `error_rate > max_error_rate`, traffic automatically reverts to the previous stable routing configuration.

---

## 5. One Thing I Would Do Differently With More Time

With more time and a multi-node production target, I would make four concrete upgrades that were intentionally simplified here to keep the single-container submission clean and maintainable:

1. **Container-per-Model Isolation & True CPU/Memory Cgroups:** Replace in-container subprocess workers with Kubernetes Pods / Ray Serve / Triton deployments so each model version gets kernel-enforced cgroup CPU/memory quotas, independent Python dependency environments, and zero blast radius for untrusted custom code.
2. **Externalized Storage & Metrics Pipeline:** Swap SQLite + local disk for PostgreSQL + S3-compatible object storage (MinIO/S3) and emit metrics to Prometheus/OpenTelemetry rather than querying `InferenceLog` rows directly—enabling horizontal scaling of stateless API replicas.
3. **Adaptive Autoscaling & Dynamic Micro-Batching:** Add an autoscaler loop driven by queue depth / p95 latency and server-side dynamic micro-batching (coalescing concurrent single `/predict` calls into vectorized batches for GPU/ONNX throughput).
4. **Comparison Dashboard & Statistical A/B Testing:** Add a lightweight web UI for side-by-side version comparison plus two-proportion z-test / Mann-Whitney U significance testing on A/B traffic.
