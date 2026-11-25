from collections import OrderedDict
import logging
import multiprocessing
from multiprocessing.connection import Connection
import re
import threading
import time
from typing import Any

from app.config import Settings
from app.errors import AppError
from app.runtime.worker_process import worker_main

logger = logging.getLogger(__name__)

# Accepts memory strings like "512Mi" or "2Gi"
MEMORY_RE = re.compile(r"^(\d+)(Mi|Gi)$")


def parse_memory_bytes(memory_spec: str) -> int:
    # Check that the memory string has a valid number + Mi/Gi unit
    match = MEMORY_RE.match(memory_spec.strip())
    if not match:
        raise AppError(
            code="INVALID_DEPLOYMENT_CONFIG",
            message=f"Invalid memory resource '{memory_spec}'. Expected format like '512Mi' or '2Gi'.",
            status_code=422,
        )
    # Reject zero and convert Mi or Gi into bytes
    amount = int(match.group(1))
    if amount <= 0:
        raise AppError(code="INVALID_DEPLOYMENT_CONFIG", message="Memory limit must be positive", status_code=422)
    multiplier = 1024 * 1024 if match.group(2) == "Mi" else 1024 * 1024 * 1024
    return amount * multiplier


def parse_cpu_cores(cpu_spec: str | float | int) -> float:
    # Make sure CPU is a valid number between 0 and 16
    try:
        cores = float(cpu_spec)
    except (TypeError, ValueError) as exc:
        raise AppError(
            code="INVALID_DEPLOYMENT_CONFIG",
            message=f"Invalid cpu resource '{cpu_spec}'. Must be a positive number.",
            status_code=422,
        ) from exc
    if cores <= 0 or cores > 16:
        raise AppError(code="INVALID_DEPLOYMENT_CONFIG", message="CPU limit must be > 0 and <= 16", status_code=422)
    return cores


# Runs one model copy in a separate process and talks to it through a pipe
class ReplicaWorker:
    def __init__(self, framework: str, artifact_path: str, memory_bytes: int | None, startup_timeout: float) -> None:
        self.framework = framework
        self.artifact_path = artifact_path
        self.memory_bytes = memory_bytes
        self.startup_timeout = startup_timeout
        # Use "spawn" (fresh process) so it works on both Windows and Linux without deadlocks
        self._ctx = multiprocessing.get_context("spawn")
        self._lock = threading.Lock()
        self._process: Any = None
        self._conn: Connection | None = None
        self.in_flight = 0

    def start(self) -> None:
        # Open a two-way pipe and launch the child process running worker_main
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        proc = self._ctx.Process(
            target=worker_main,
            args=(self.framework, self.artifact_path, self.memory_bytes, child_conn),
            daemon=True,
        )
        proc.start()
        # Close our handle to the child's end of the pipe
        child_conn.close()

        # Wait until the worker tells us it finished loading the model (or time out)
        if not parent_conn.poll(self.startup_timeout):
            parent_conn.close()
            proc.terminate()
            proc.join(timeout=2.0)
            raise RuntimeError(f"Worker startup timed out after {self.startup_timeout}s")

        # Read the startup message; if the pipe closed early, the worker crashed on load
        try:
            msg = parent_conn.recv()
        except EOFError as exc:
            parent_conn.close()
            proc.join(timeout=2.0)
            raise RuntimeError("Worker crashed during model loading") from exc

        # If the worker reported an error, clean it up and surface the message
        if msg.get("status") != "ready":
            parent_conn.close()
            proc.join(timeout=2.0)
            raise RuntimeError(msg.get("error", "Unknown model load error"))

        self._process = proc
        self._conn = parent_conn

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def _call(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        # Lock the pipe so two requests don't talk to the same worker at the same time
        with self._lock:
            if not self.is_alive() or self._conn is None:
                raise ConnectionError("Model worker process is not alive")
            try:
                self._conn.send(payload)
                # If the model takes too long, kill this stuck worker and raise a timeout
                if not self._conn.poll(timeout):
                    self._force_kill_locked()
                    raise TimeoutError(f"Inference timed out after {timeout}s")
                return self._conn.recv()
            except TimeoutError:
                # Re-raise TimeoutError first since it is a subclass of OSError in Python
                raise
            except (EOFError, BrokenPipeError, OSError) as exc:
                # Pipe broke because the worker process died unexpectedly
                self._force_kill_locked()
                raise ConnectionError("Model worker process crashed during inference") from exc

    def health_check(self, timeout: float = 5.0) -> bool:
        # Ping the worker over the pipe to confirm it is still responsive
        try:
            return bool(self._call({"cmd": "health"}, timeout=timeout).get("ok"))
        except Exception:
            return False

    def predict(self, inputs: dict[str, Any], timeout: float) -> dict[str, Any]:
        resp = self._call({"cmd": "predict", "inputs": inputs}, timeout=timeout)
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "Model prediction failed"))
        return resp["outputs"]

    def predict_batch(self, inputs_list: list[dict[str, Any]], timeout: float) -> list[dict[str, Any]]:
        resp = self._call({"cmd": "predict_batch", "inputs_list": inputs_list}, timeout=timeout)
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "Batch model prediction failed"))
        return resp["outputs_list"]

    def restart(self) -> None:
        # Kill the old dead/stuck process and start a fresh one
        with self._lock:
            self._force_kill_locked()
        self.start()

    def _force_kill_locked(self) -> None:
        # Close the pipe and forcefully terminate the child process so no zombie is left
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
            self._process = None

    def stop(self) -> None:
        # Ask the worker to shut down politely first, then force-kill if it doesn't exit
        with self._lock:
            if self._conn is not None and self.is_alive():
                try:
                    self._conn.send({"cmd": "shutdown"})
                    self._conn.poll(1.5)
                except Exception:
                    pass
            self._force_kill_locked()


# Manages a group of worker copies (replicas) for one deployed model version
class ReplicaPool:
    def __init__(
        self,
        model_name: str,
        version: str,
        framework: str,
        artifact_path: str,
        replicas: int,
        memory_bytes: int | None,
        settings: Settings,
    ) -> None:
        self.model_name = model_name
        self.version = version
        self.framework = framework
        self.artifact_path = artifact_path
        self.memory_bytes = memory_bytes
        self.settings = settings
        self._pool_lock = threading.Lock()
        self._workers: list[ReplicaWorker] = []
        self._pending = 0
        self._rr_index = 0
        # Start all requested worker replicas right away
        self._start_workers(replicas)

    def _start_workers(self, count: int) -> None:
        # Start N new workers; if any one fails, stop the ones we already started
        created: list[ReplicaWorker] = []
        try:
            for _ in range(count):
                worker = ReplicaWorker(
                    framework=self.framework,
                    artifact_path=self.artifact_path,
                    memory_bytes=self.memory_bytes,
                    startup_timeout=self.settings.worker_startup_timeout_seconds,
                )
                worker.start()
                created.append(worker)
        except Exception:
            for w in created:
                w.stop()
            raise
        with self._pool_lock:
            self._workers.extend(created)

    def scale(self, new_replicas: int, memory_bytes: int | None = None) -> None:
        # Either spin up extra workers or gracefully stop extra ones to match new_replicas
        if memory_bytes is not None:
            self.memory_bytes = memory_bytes
        with self._pool_lock:
            current = len(self._workers)
        if new_replicas > current:
            self._start_workers(new_replicas - current)
            return
        if new_replicas < current:
            to_stop: list[ReplicaWorker] = []
            with self._pool_lock:
                while len(self._workers) > new_replicas:
                    to_stop.append(self._workers.pop())
            for w in to_stop:
                w.stop()

    def _acquire_slot(self) -> ReplicaWorker:
        with self._pool_lock:
            if not self._workers:
                raise AppError(
                    code="MODEL_NOT_DEPLOYED",
                    message=f"No active worker replicas for '{self.model_name}' version '{self.version}'",
                    status_code=503,
                )
            # Reject right away with 429 if too many requests are already waiting
            if self._pending >= self.settings.max_queue_depth_per_deployment:
                raise AppError(
                    code="QUEUE_OVERFLOW",
                    message=f"Inference queue full for '{self.model_name}' version '{self.version}'",
                    status_code=429,
                )
            self._pending += 1
            # Pick the worker with the fewest active requests (round-robin on ties)
            n = len(self._workers)
            best = min(range(n), key=lambda i: (self._workers[(self._rr_index + i) % n].in_flight, i))
            idx = (self._rr_index + best) % n
            self._rr_index = (idx + 1) % n
            worker = self._workers[idx]
            worker.in_flight += 1
            return worker

    def _release_slot(self, worker: ReplicaWorker) -> None:
        # Decrement active counters once the request finishes or errors out
        with self._pool_lock:
            self._pending = max(0, self._pending - 1)
            worker.in_flight = max(0, worker.in_flight - 1)

    def _respawn_async(self, worker: ReplicaWorker) -> None:
        # Restart a crashed or timed-out worker in the background so future requests succeed
        def _do_restart() -> None:
            try:
                worker.restart()
                logger.info("Respawned crashed worker for %s:%s", self.model_name, self.version)
            except Exception as exc:
                logger.error("Failed to respawn worker for %s:%s: %s", self.model_name, self.version, exc)

        threading.Thread(target=_do_restart, daemon=True).start()

    def _dispatch(self, fn_name: str, payload: Any) -> Any:
        # Reserve a worker, run the call, and handle timeouts or crashes cleanly
        worker = self._acquire_slot()
        try:
            fn = getattr(worker, fn_name)
            return fn(payload, timeout=self.settings.inference_timeout_seconds)
        except TimeoutError as exc:
            self._respawn_async(worker)
            raise AppError(code="INFERENCE_TIMEOUT", message=str(exc), status_code=504) from exc
        except ConnectionError as exc:
            self._respawn_async(worker)
            raise AppError(code="INFERENCE_ERROR", message=str(exc), status_code=500) from exc
        except RuntimeError as exc:
            raise AppError(code="INFERENCE_ERROR", message=str(exc), status_code=500) from exc
        finally:
            self._release_slot(worker)

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return self._dispatch("predict", inputs)

    def predict_batch(self, inputs_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._dispatch("predict_batch", inputs_list)

    def health_summary(self) -> dict[str, Any]:
        # Ping every replica in this pool and report how many are healthy plus queue size
        with self._pool_lock:
            workers = list(self._workers)
            pending = self._pending
        healthy = sum(1 for w in workers if w.health_check())
        return {
            "replicas_total": len(workers),
            "replicas_healthy": healthy,
            "queue_depth": pending,
            "healthy": len(workers) > 0 and healthy > 0,
        }

    def stop_all(self) -> None:
        # Stop every worker in this pool and clear the list
        with self._pool_lock:
            workers = list(self._workers)
            self._workers.clear()
        for w in workers:
            w.stop()


# Keeps track of all currently loaded model pools and caps memory with an LRU limit
class ServingManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        # OrderedDict keeps models sorted from oldest-used (front) to most-recently-used (back)
        self._pools: OrderedDict[tuple[str, str], ReplicaPool] = OrderedDict()

    def deploy(
        self,
        model_name: str,
        version: str,
        framework: str,
        artifact_path: str,
        replicas: int,
        memory_bytes: int | None,
    ) -> dict[str, Any]:
        key = (model_name, version)
        # If we reached the max number of loaded models, stop the oldest unused one first
        evicted: ReplicaPool | None = None
        with self._lock:
            if key not in self._pools and len(self._pools) >= self.settings.max_loaded_models:
                _, evicted = self._pools.popitem(last=False)
        if evicted is not None:
            logger.info("Evicting LRU model %s:%s due to capacity cap", evicted.model_name, evicted.version)
            evicted.stop_all()

        # Start the worker pool and run a health check before allowing traffic
        try:
            pool = ReplicaPool(
                model_name=model_name,
                version=version,
                framework=framework,
                artifact_path=artifact_path,
                replicas=replicas,
                memory_bytes=memory_bytes,
                settings=self.settings,
            )
            health = pool.health_summary()
            if not health["healthy"]:
                pool.stop_all()
                raise RuntimeError("Deployed model failed readiness health check")
        except Exception as exc:
            raise AppError(
                code="MODEL_LOAD_FAILED",
                message=f"Failed to load model '{model_name}' version '{version}': {exc}",
                status_code=422,
            ) from exc

        # Register the new pool (and stop any old pool if we were replacing it)
        with self._lock:
            old_pool = self._pools.pop(key, None)
            self._pools[key] = pool
        if old_pool is not None:
            old_pool.stop_all()
        return health

    def undeploy(self, model_name: str, version: str) -> None:
        # Remove the model pool from memory and shut down its worker processes
        with self._lock:
            pool = self._pools.pop((model_name, version), None)
        if pool is not None:
            pool.stop_all()

    def scale(self, model_name: str, version: str, replicas: int, memory_bytes: int | None = None) -> None:
        # Find the running pool and adjust how many worker processes it has
        with self._lock:
            pool = self._pools.get((model_name, version))
        if pool is None:
            raise AppError(
                code="MODEL_NOT_DEPLOYED",
                message=f"Model '{model_name}' version '{version}' is not currently running",
                status_code=404,
            )
        pool.scale(replicas, memory_bytes=memory_bytes)

    def health(self, model_name: str, version: str) -> dict[str, Any]:
        with self._lock:
            pool = self._pools.get((model_name, version))
        if pool is None:
            return {"replicas_total": 0, "replicas_healthy": 0, "queue_depth": 0, "healthy": False}
        return pool.health_summary()

    def _get_pool_for_inference(self, model_name: str, version: str) -> ReplicaPool:
        # Mark this model as recently used so it stays in memory
        with self._lock:
            pool = self._pools.get((model_name, version))
            if pool is not None:
                self._pools.move_to_end((model_name, version))
        if pool is None:
            raise AppError(
                code="MODEL_NOT_DEPLOYED",
                message=f"Model '{model_name}' version '{version}' is not deployed",
                status_code=503,
            )
        return pool

    def predict(self, model_name: str, version: str, inputs: dict[str, Any]) -> tuple[dict[str, Any], float]:
        # Run a single prediction and measure how many milliseconds it took
        pool = self._get_pool_for_inference(model_name, version)
        start = time.perf_counter()
        outputs = pool.predict(inputs)
        return outputs, round((time.perf_counter() - start) * 1000.0, 3)

    def predict_batch(
        self, model_name: str, version: str, inputs_list: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], float]:
        # Run a batch prediction and measure total time in milliseconds
        pool = self._get_pool_for_inference(model_name, version)
        start = time.perf_counter()
        outputs_list = pool.predict_batch(inputs_list)
        return outputs_list, round((time.perf_counter() - start) * 1000.0, 3)

    def shutdown_all(self) -> None:
        # Called when the server shuts down so every child process exits cleanly
        with self._lock:
            pools = list(self._pools.values())
            self._pools.clear()
        for pool in pools:
            pool.stop_all()
