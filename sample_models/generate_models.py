"""Deterministic generator for sample ML artifacts used in testing and demos."""

from pathlib import Path
import logging
import pickle

from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path(__file__).resolve().parent


def generate_all(output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}

    # 1. Simple Classifier v1 (10 features -> class + confidence)
    x_clf, y_clf = make_classification(
        n_samples=400,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )
    clf_v1 = LogisticRegression(max_iter=500, random_state=42)
    clf_v1.fit(x_clf, y_clf)
    clf_v1_path = output_dir / "fraud_classifier_v1.pkl"
    with clf_v1_path.open("wb") as f:
        pickle.dump(clf_v1, f)
    generated["fraud_classifier_v1"] = clf_v1_path

    # 2. Simple Classifier v2 (10 features -> alternative model for A/B testing & rollout)
    clf_v2 = RandomForestClassifier(n_estimators=25, max_depth=6, random_state=99)
    clf_v2.fit(x_clf, y_clf)
    clf_v2_path = output_dir / "fraud_classifier_v2.pkl"
    with clf_v2_path.open("wb") as f:
        pickle.dump(clf_v2, f)
    generated["fraud_classifier_v2"] = clf_v2_path

    # 3. Regression Model (5 features -> continuous predicted value)
    x_reg, y_reg = make_regression(
        n_samples=300,
        n_features=5,
        n_informative=4,
        noise=2.5,
        random_state=42,
    )
    reg_v1 = Ridge(alpha=1.0, random_state=42)
    reg_v1.fit(x_reg, y_reg)
    reg_v1_path = output_dir / "house_regressor_v1.pkl"
    with reg_v1_path.open("wb") as f:
        pickle.dump(reg_v1, f)
    generated["house_regressor_v1"] = reg_v1_path

    # 4. ONNX export of classifier v1 (gracefully skipped if skl2onnx is unavailable locally)
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [("features", FloatTensorType([None, 10]))]
        onnx_model = convert_sklearn(
            clf_v1,
            initial_types=initial_type,
            options={id(clf_v1): {"zipmap": False}},
        )
        onnx_path = output_dir / "fraud_classifier_v1.onnx"
        onnx_path.write_bytes(onnx_model.SerializeToString())
        generated["fraud_classifier_v1_onnx"] = onnx_path
    except ImportError:
        logger.warning("skl2onnx not installed in this Python environment; skipping .onnx export")

    return generated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    files = generate_all()
    for name, path in files.items():
        logger.info("Generated %s -> %s (%d bytes)", name, path.name, path.stat().st_size)
