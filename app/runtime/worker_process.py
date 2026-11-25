from multiprocessing.connection import Connection
import sys
import traceback
from typing import Any

from app.runtime.loaders import create_runner


def _check_memory_budget(memory_bytes: int | None) -> None:
    # Skip memory checks on Windows (the resource module is Linux/macOS only)
    if memory_bytes is None or sys.platform == "win32":
        return
    try:
        import resource

        # Check actual RAM used after loading (instead of RLIMIT_DATA, which breaks OpenBLAS on Linux)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = rss if sys.platform == "darwin" else rss * 1024
        if rss_bytes > memory_bytes:
            raise MemoryError(
                f"Worker RSS ({rss_bytes // (1024 * 1024)}Mi) exceeded configured limit "
                f"({memory_bytes // (1024 * 1024)}Mi)"
            )
    except MemoryError:
        raise
    except Exception:
        pass


def worker_main(
    framework: str,
    artifact_path: str,
    memory_bytes: int | None,
    conn: Connection,
) -> None:
    # Step 1: Load the model inside this separate process, check memory, and tell the parent we're ready
    try:
        runner = create_runner(framework, artifact_path)
        _check_memory_budget(memory_bytes)
        if not runner.health_check():
            raise RuntimeError("Model runner failed post-load health check")
        conn.send({"status": "ready"})
    except Exception as exc:
        # If loading fails, send the error message back to the parent and exit cleanly
        try:
            conn.send({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
        conn.close()
        return

    # Step 2: Stay alive in a loop waiting for commands from the main API process
    try:
        while True:
            try:
                msg: dict[str, Any] = conn.recv()
            except EOFError:
                # Parent closed the pipe, so exit the loop
                break

            cmd = msg.get("cmd")
            # Stop the worker cleanly when asked to shut down
            if cmd == "shutdown":
                conn.send({"ok": True})
                break
            # Answer health probes from the parent process
            if cmd == "health":
                conn.send({"ok": bool(runner.health_check())})
                continue
            # Run a single prediction and send back either the result or the error
            if cmd == "predict":
                try:
                    conn.send({"ok": True, "outputs": runner.predict(msg["inputs"])})
                except Exception as exc:
                    conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=3)})
                continue
            # Run a batch prediction in one call and send back all results
            if cmd == "predict_batch":
                try:
                    conn.send({"ok": True, "outputs_list": runner.predict_batch(msg["inputs_list"])})
                except Exception as exc:
                    conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=3)})
                continue
            conn.send({"ok": False, "error": f"Unknown command: {cmd}"})
    finally:
        # Always close our end of the pipe before exiting
        conn.close()
