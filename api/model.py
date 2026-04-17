import joblib
import json
import shap
from pathlib import Path

_ML_DIR = Path(__file__).parent.parent / "ml"
MODEL_PATH = _ML_DIR / "model.pkl"
FEATURES_PATH = _ML_DIR / "feature_names.json"
VERSION_PATH = _ML_DIR / "VERSION"

_pipeline = None
_explainer = None
_feature_names: list[str] = []
_model_version: str = "unknown"


def load_model() -> None:
    global _pipeline, _explainer, _feature_names, _model_version
    _pipeline = joblib.load(MODEL_PATH)
    _feature_names = json.loads(FEATURES_PATH.read_text())
    _model_version = VERSION_PATH.read_text().strip()
    classifier = _pipeline.named_steps["classifier"]
    _explainer = shap.TreeExplainer(classifier)
    print(f"[model] Loaded {_model_version} — {len(_feature_names)} features")


def get_pipeline():
    return _pipeline


def get_explainer():
    return _explainer


def get_feature_names() -> list[str]:
    return _feature_names


def get_model_version() -> str:
    return _model_version
