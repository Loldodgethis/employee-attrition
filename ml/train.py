import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

DATA_PATH = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
MODEL_PATH = Path("model.pkl")
FEATURES_PATH = Path("feature_names.json")
VERSION_PATH = Path("VERSION")

CATEGORICAL_FEATURES = [
    "BusinessTravel", "Department", "JobRole", "MaritalStatus", "OverTime"
]
NUMERICAL_FEATURES = [
    "Age", "DistanceFromHome", "EnvironmentSatisfaction",
    "JobSatisfaction", "MonthlyIncome", "NumCompaniesWorked",
    "TotalWorkingYears", "WorkLifeBalance", "YearsAtCompany",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET = "Attrition"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df[ALL_FEATURES + [TARGET]]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERICAL_FEATURES),
    ])


def train(df: pd.DataFrame):
    X, y = df[ALL_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )),
    ])
    lr_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        )),
    ])

    rf_pipeline.fit(X_train, y_train)
    lr_pipeline.fit(X_train, y_train)

    rf_auc = roc_auc_score(y_test, rf_pipeline.predict_proba(X_test)[:, 1])
    lr_auc = roc_auc_score(y_test, lr_pipeline.predict_proba(X_test)[:, 1])

    best = rf_pipeline if rf_auc >= lr_auc else lr_pipeline
    best_auc = max(rf_auc, lr_auc)
    best_acc = accuracy_score(y_test, best.predict(X_test))
    best_name = "RandomForest" if rf_auc >= lr_auc else "LogisticRegression"

    print(f"RandomForest  ROC-AUC: {rf_auc:.4f}")
    print(f"LogisticRegression ROC-AUC: {lr_auc:.4f}")
    print(f"Selected: {best_name}  accuracy={best_acc:.4f}  auc={best_auc:.4f}")

    feature_names = list(best.named_steps["preprocessor"].get_feature_names_out())
    return best, feature_names


if __name__ == "__main__":
    df = load_data()
    pipeline, feature_names = train(df)

    joblib.dump(pipeline, MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps(feature_names))

    print(f"Saved {MODEL_PATH}  ({len(feature_names)} features)")
    print(f"Saved {FEATURES_PATH}")
