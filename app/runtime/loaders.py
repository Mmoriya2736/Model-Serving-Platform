from abc import ABC, abstractmethod
import importlib.util
import inspect
import math
from pathlib import Path
import pickle
from typing import Any

import joblib
import numpy as np


def to_native(value: Any) -> Any:
    # Turn NumPy values into plain Python types so FastAPI can return them as JSON
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    # Also catch NaN or Infinity here because JSON doesn't support them
    if isinstance(value, (np.floating, float)):
        fval = float(value)
        if not math.isfinite(fval):
            raise ValueError("Model produced non-finite float output")
        return fval
    # Recursively clean up arrays, dicts, and lists
    if isinstance(value, np.ndarray):
        return [to_native(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(k): to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(v) for v in value]
    return value


# Shared interface that every model format (sklearn, ONNX, custom Python) implements
class ModelRunner(ABC):
    @abstractmethod
    def load(self, artifact_path: Path) -> None:
        ...

    @abstractmethod
    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        ...

    def predict_batch(self, inputs_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Default batch behavior: run one item at a time unless the subclass optimizes it
        return [self.predict(item) for item in inputs_list]

    def health_check(self) -> bool:
        return True


class SklearnRunner(ModelRunner):
    def __init__(self) -> None:
        self.estimator: Any = None

    def load(self, artifact_path: Path) -> None:
        # Try loading with regular pickle first, and fall back to joblib if needed
        try:
            with artifact_path.open("rb") as f:
                self.estimator = pickle.load(f)
        except Exception:
            self.estimator = joblib.load(artifact_path)

        # Make sure the loaded object actually looks like a scikit-learn model
        if not hasattr(self.estimator, "predict"):
            raise ValueError("Loaded sklearn artifact does not implement a predict() method")

    def _to_matrix(self, inputs_list: list[dict[str, Any]]) -> np.ndarray:
        # Pull the "features" list out of each request and stack them into a 2D NumPy array
        rows = []
        for item in inputs_list:
            if "features" not in item:
                raise ValueError("Expected 'features' array in inputs for sklearn model")
            rows.append(item["features"])
        return np.asarray(rows, dtype=np.float64)

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        # Reuse the batch path for a single item so there is only one code path
        return self.predict_batch([inputs])[0]

    def predict_batch(self, inputs_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Run the whole batch in one NumPy/sklearn call for better speed
        matrix = self._to_matrix(inputs_list)
        preds = self.estimator.predict(matrix)
        # If this is a classifier with predict_proba, also grab its confidence scores
        probas = self.estimator.predict_proba(matrix) if hasattr(self.estimator, "predict_proba") else None

        # Package each row's prediction (and confidence if available) into a plain dict
        results: list[dict[str, Any]] = []
        for idx, pred in enumerate(preds):
            out: dict[str, Any] = {"prediction": to_native(pred)}
            if probas is not None:
                out["confidence"] = to_native(float(np.max(probas[idx])))
            results.append(out)
        return results

    def health_check(self) -> bool:
        return self.estimator is not None and hasattr(self.estimator, "predict")


class OnnxRunner(ModelRunner):
    def __init__(self) -> None:
        self.session: Any = None
        self.input_names: list[str] = []

    def load(self, artifact_path: Path) -> None:
        # Import onnxruntime only when needed so local Python 3.14 still works without it
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed in this Python environment (use Docker python:3.12-slim for ONNX support)"
            ) from exc

        # Start a CPU inference session and remember the model's expected input tensor names
        self.session = ort.InferenceSession(str(artifact_path), providers=["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in self.session.get_inputs()]

    def _build_feed(self, inputs_list: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        # If the ONNX model has one input, accept either that exact name or "features"
        if len(self.input_names) == 1:
            name = self.input_names[0]
            rows = [item[name] if name in item else item["features"] for item in inputs_list]
            return {name: np.asarray(rows, dtype=np.float32)}
        # For multi-input ONNX models, map each input tensor by its exact name
        return {
            name: np.asarray([item[name] for item in inputs_list], dtype=np.float32)
            for name in self.input_names
        }

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return self.predict_batch([inputs])[0]

    def predict_batch(self, inputs_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Run the ONNX graph and grab the main prediction (plus probabilities if present)
        raw_outputs = self.session.run(None, self._build_feed(inputs_list))
        primary = raw_outputs[0]
        probs = raw_outputs[1] if len(raw_outputs) > 1 else None

        results: list[dict[str, Any]] = []
        for idx in range(len(inputs_list)):
            # Unwrap 1-element NumPy arrays into a plain number
            pred_val = primary[idx]
            if isinstance(pred_val, np.ndarray) and pred_val.size == 1:
                pred_val = pred_val.item()
            out: dict[str, Any] = {"prediction": to_native(pred_val)}
            # Pull the highest class probability whether ONNX returns a dict or an array
            if probs is not None:
                row_prob = probs[idx]
                if isinstance(row_prob, dict) and row_prob:
                    out["confidence"] = to_native(max(float(v) for v in row_prob.values()))
                elif isinstance(row_prob, (np.ndarray, list, tuple)):
                    out["confidence"] = to_native(float(np.max(row_prob)))
            results.append(out)
        return results

    def health_check(self) -> bool:
        return self.session is not None


class CustomPythonRunner(ModelRunner):
    def __init__(self) -> None:
        self.instance: Any = None

    def load(self, artifact_path: Path) -> None:
        # Load the uploaded .py file directly from disk as a Python module
        spec = importlib.util.spec_from_file_location(f"custom_model_{artifact_path.stem}", artifact_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load custom Python module from {artifact_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Look for a class named "Model" first; otherwise use any class in that file with predict()
        model_cls = getattr(module, "Model", None)
        if model_cls is None:
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == module.__name__ and callable(getattr(obj, "predict", None)):
                    model_cls = obj
                    break
        if model_cls is None:
            raise ValueError("Custom Python artifact must define a class with a predict(self, inputs) method")

        # Create an instance of the user's class and confirm predict() is callable
        self.instance = model_cls()
        if not callable(getattr(self.instance, "predict", None)):
            raise ValueError("Custom model class does not expose a callable predict() method")

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        # Call the custom model and make sure the result is a JSON-safe dict
        result = self.instance.predict(inputs)
        if not isinstance(result, dict):
            return {"prediction": to_native(result)}
        return to_native(result)

    def health_check(self) -> bool:
        return self.instance is not None and callable(getattr(self.instance, "predict", None))


def create_runner(framework: str, artifact_path: str) -> ModelRunner:
    # Make sure the model file exists on disk before trying to load it
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"Artifact file not found: {artifact_path}")

    # Pick the right loader class based on the framework name
    runners: dict[str, type[ModelRunner]] = {
        "sklearn": SklearnRunner,
        "onnx": OnnxRunner,
        "custom": CustomPythonRunner,
    }
    runner_cls = runners.get(framework)
    if runner_cls is None:
        raise ValueError(f"Unsupported framework: {framework}")

    # Create the runner, load the model file, and hand it back ready to serve
    runner = runner_cls()
    runner.load(path)
    return runner
