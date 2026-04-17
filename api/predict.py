import numpy as np
import pandas as pd
from model import get_pipeline, get_explainer, get_feature_names

CATEGORICAL_FEATURES = [
    "BusinessTravel", "Department", "JobRole", "MaritalStatus", "OverTime"
]
NUMERICAL_FEATURES = [
    "Age", "DistanceFromHome", "EnvironmentSatisfaction",
    "JobSatisfaction", "MonthlyIncome", "NumCompaniesWorked",
    "TotalWorkingYears", "WorkLifeBalance", "YearsAtCompany",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def validate_input(data: dict) -> None:
    missing = [f for f in ALL_FEATURES if f not in data]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")


def risk_level(probability: float) -> str:
    if probability >= 0.60:
        return "HIGH"
    if probability >= 0.30:
        return "MED"
    return "LOW"


def _clean_feature_name(raw: str) -> str:
    """'cat__OverTime_Yes' -> 'OverTime',  'num__Age' -> 'Age'"""
    name = raw.split("__", 1)[-1]
    if raw.startswith("cat__"):
        name = name.rsplit("_", 1)[0]
    return name


def get_top_shap(shap_values: np.ndarray, feature_names: list[str], n: int = 5) -> list[dict]:
    top_idx = np.argsort(np.abs(shap_values))[::-1][:n]
    return [
        {"feature": _clean_feature_name(feature_names[i]), "value": round(float(shap_values[i]), 4)}
        for i in top_idx
    ]


def predict(data: dict) -> dict:
    validate_input(data)
    pipeline = get_pipeline()
    explainer = get_explainer()
    feature_names = get_feature_names()

    df = pd.DataFrame([data])[ALL_FEATURES]

    probability = float(pipeline.predict_proba(df)[0][1])
    prediction = probability >= 0.5

    transformed = pipeline.named_steps["preprocessor"].transform(df)
    raw_shap = explainer.shap_values(transformed)
    # RandomForest returns list [class0, class1]; handle both shapes
    if isinstance(raw_shap, list):
        sv = raw_shap[1][0]
    else:
        sv = raw_shap[0]

    return {
        "prediction": bool(prediction),
        "probability": round(probability, 4),
        "risk_level": risk_level(probability),
        "shap_values": get_top_shap(sv, feature_names),
    }
