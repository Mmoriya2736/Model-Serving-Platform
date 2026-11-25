"""Sample custom Python model demonstrating the custom model contract."""

import math
import time
from typing import Any


class Model:
    """Weighted risk scorer implementing the platform's predict(inputs) contract."""

    def __init__(self) -> None:
        self.weights = [0.25, -0.15, 0.4, 0.1, 0.35]
        self.bias = -0.3

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        features = inputs["features"]
        delay_ms = float(inputs.get("delay_ms", 0))
        if delay_ms > 0:
            time.sleep(min(delay_ms, 5000.0) / 1000.0)

        score = sum(w * float(x) for w, x in zip(self.weights, features)) + self.bias
        probability = 1.0 / (1.0 + math.exp(-score))
        label = 1 if probability >= 0.5 else 0
        confidence = probability if label == 1 else (1.0 - probability)
        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "risk_score": round(probability, 4),
        }
