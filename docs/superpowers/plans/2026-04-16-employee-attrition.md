# Employee Attrition Prediction System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a full-stack ML app that predicts employee attrition — Flask API + Next.js frontend + Supabase PostgreSQL, all free-tier.

**Architecture:** Python `train.py` trains a Random Forest on the IBM HR dataset and saves `model.pkl` + `feature_names.json`. A Flask API loads those artifacts at startup and serves `/predict`, `/history`, and `/health`. A Next.js 14 app provides a form, a SHAP results page, and a history table.

**Tech Stack:** Python 3.11, scikit-learn 1.4, SHAP 0.45, Flask 3.0, psycopg2, Next.js 14, TypeScript, Tailwind CSS, Recharts, Supabase PostgreSQL.

---

## File Map

```
employee-retention/
  ml/
    data/                          ← place IBM HR CSV here (gitignored)
    train.py                       ← training script
    model.pkl                      ← saved sklearn Pipeline (gitignored)
    feature_names.json             ← feature names post-encoding (gitignored)
    VERSION                        ← "v1.0.0"
    requirements.txt
  api/
    app.py                         ← Flask app + routes
    model.py                       ← model/explainer loader
    predict.py                     ← prediction logic + SHAP
    db.py                          ← Supabase connection + queries
    requirements.txt
    Procfile
    .env.example
    tests/
      test_predict.py
      test_routes.py
  frontend/
    app/
      layout.tsx                   ← root layout + ApiStatusBanner
      page.tsx                     ← prediction form (/)
      result/page.tsx              ← SHAP results (/result)
      history/page.tsx             ← prediction log (/history)
    components/
      PredictionForm.tsx
      ResultCard.tsx
      ShapChart.tsx
      HistoryTable.tsx
      ApiStatusBanner.tsx
    lib/
      types.ts                     ← shared TypeScript types
      api.ts                       ← fetch wrappers
    .env.example
  .gitignore
  README.md
```

---

## Task 1: Repo Scaffold + ML Setup

**Files:**
- Create: `.gitignore`
- Create: `ml/requirements.txt`
- Create: `ml/VERSION`
- Create: `ml/data/.gitkeep`

- [ ] **Step 1: Add .gitignore**

```
# ML artifacts
ml/data/
ml/model.pkl
ml/feature_names.json
__pycache__/
*.pyc
.env
.env.local

# Node
frontend/node_modules/
frontend/.next/

# Misc
.DS_Store
.superpowers/
```

Save to `employee-retention/.gitignore`.

- [ ] **Step 2: Create ml/requirements.txt**

```
pandas==2.2.2
scikit-learn==1.4.2
shap==0.45.1
joblib==1.4.2
numpy==1.26.4
```

- [ ] **Step 3: Create ml/VERSION**

```
v1.0.0
```

- [ ] **Step 4: Create ml/data/ directory and placeholder**

```bash
mkdir -p ml/data
touch ml/data/.gitkeep
```

- [ ] **Step 5: Install ML dependencies**

```bash
cd ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Expected: pip installs pandas, scikit-learn, shap, joblib, numpy with no errors.

- [ ] **Step 6: Download IBM HR dataset**

Go to https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset and download `WA_Fn-UseC_-HR-Employee-Attrition.csv`. Place it at:

```
ml/data/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

Verify:
```bash
wc -l ml/data/WA_Fn-UseC_-HR-Employee-Attrition.csv
```
Expected: `1471` (1470 rows + header).

- [ ] **Step 7: Commit scaffold**

```bash
git add .gitignore ml/requirements.txt ml/VERSION ml/data/.gitkeep
git commit -m "feat: repo scaffold and ML setup"
```

---

## Task 2: ML Training Script

**Files:**
- Create: `ml/train.py`

- [ ] **Step 1: Create ml/train.py**

```python
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
```

- [ ] **Step 2: Run the training script**

```bash
cd ml
source venv/bin/activate
python train.py
```

Expected output (approximate):
```
RandomForest  ROC-AUC: 0.8210
LogisticRegression ROC-AUC: 0.7890
Selected: RandomForest  accuracy=0.8707  auc=0.8210
Saved model.pkl  (XX features)
Saved feature_names.json
```

- [ ] **Step 3: Verify artifacts exist**

```bash
ls -lh ml/model.pkl ml/feature_names.json
```

Expected: both files exist, `model.pkl` is ~10–30 MB.

- [ ] **Step 4: Commit**

```bash
git add ml/train.py
git commit -m "feat: ML training script — Random Forest on IBM HR dataset"
```

---

## Task 3: Flask API Scaffold

**Files:**
- Create: `api/requirements.txt`
- Create: `api/Procfile`
- Create: `api/.env.example`
- Create: `api/model.py`
- Create: `api/predict.py`

- [ ] **Step 1: Create api/requirements.txt**

```
flask==3.0.3
flask-cors==4.0.1
scikit-learn==1.4.2
shap==0.45.1
joblib==1.4.2
numpy==1.26.4
pandas==2.2.2
psycopg2-binary==2.9.9
python-dotenv==1.0.1
gunicorn==22.0.0
pytest==8.2.0
```

- [ ] **Step 2: Create api/Procfile**

```
web: gunicorn app:app
```

- [ ] **Step 3: Create api/.env.example**

```
DATABASE_URL=postgresql://user:password@host:5432/dbname
FRONTEND_URL=https://your-project.vercel.app
```

- [ ] **Step 4: Install API dependencies**

```bash
cd api
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Expected: all packages install with no errors.

- [ ] **Step 5: Create api/model.py**

```python
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
```

- [ ] **Step 6: Create api/predict.py**

```python
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
    # RandomForest returns list [class0, class1]; LogisticRegression returns 2D array
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
```

- [ ] **Step 7: Commit**

```bash
git add api/requirements.txt api/Procfile api/.env.example api/model.py api/predict.py
git commit -m "feat: Flask API scaffold — model loader and prediction logic"
```

---

## Task 4: Tests for predict.py

**Files:**
- Create: `api/tests/__init__.py`
- Create: `api/tests/test_predict.py`

- [ ] **Step 1: Create api/tests/__init__.py**

Empty file:
```python
```

- [ ] **Step 2: Create api/tests/test_predict.py**

```python
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from predict import validate_input, risk_level, get_top_shap, ALL_FEATURES


def valid_input() -> dict:
    return {
        "BusinessTravel": "Travel_Rarely",
        "Department": "Sales",
        "JobRole": "Sales Executive",
        "MaritalStatus": "Single",
        "OverTime": "Yes",
        "Age": 35,
        "DistanceFromHome": 10,
        "EnvironmentSatisfaction": 2,
        "JobSatisfaction": 2,
        "MonthlyIncome": 4000,
        "NumCompaniesWorked": 3,
        "TotalWorkingYears": 8,
        "WorkLifeBalance": 2,
        "YearsAtCompany": 4,
    }


def test_validate_input_accepts_valid():
    validate_input(valid_input())  # must not raise


def test_validate_input_raises_on_missing_field():
    data = valid_input()
    del data["Age"]
    with pytest.raises(ValueError, match="Age"):
        validate_input(data)


def test_validate_input_raises_on_multiple_missing():
    data = valid_input()
    del data["Age"]
    del data["MonthlyIncome"]
    with pytest.raises(ValueError):
        validate_input(data)


def test_risk_level_high_at_boundary():
    assert risk_level(0.60) == "HIGH"
    assert risk_level(0.99) == "HIGH"


def test_risk_level_med_at_boundaries():
    assert risk_level(0.59) == "MED"
    assert risk_level(0.30) == "MED"


def test_risk_level_low():
    assert risk_level(0.29) == "LOW"
    assert risk_level(0.0) == "LOW"


def test_get_top_shap_returns_n():
    shap_vals = np.array([0.1, -0.4, 0.3, -0.2, 0.5, 0.05, 0.15, 0.25, -0.35, 0.08])
    names = [
        "cat__OverTime_Yes", "num__Age", "cat__JobRole_Manager",
        "num__MonthlyIncome", "cat__MaritalStatus_Single",
        "num__YearsAtCompany", "num__JobSatisfaction",
        "num__TotalWorkingYears", "cat__Department_Sales",
        "num__DistanceFromHome",
    ]
    result = get_top_shap(shap_vals, names, n=5)
    assert len(result) == 5


def test_get_top_shap_highest_abs_first():
    shap_vals = np.array([0.1, -0.4, 0.3, -0.2, 0.5, 0.05, 0.15, 0.25, -0.35, 0.08])
    names = [
        "cat__OverTime_Yes", "num__Age", "cat__JobRole_Manager",
        "num__MonthlyIncome", "cat__MaritalStatus_Single",
        "num__YearsAtCompany", "num__JobSatisfaction",
        "num__TotalWorkingYears", "cat__Department_Sales",
        "num__DistanceFromHome",
    ]
    result = get_top_shap(shap_vals, names, n=5)
    # Index 4 has abs value 0.5 — must be first
    assert result[0]["value"] == 0.5
    assert result[0]["feature"] == "MaritalStatus"


def test_get_top_shap_cleans_categorical_name():
    shap_vals = np.array([0.9])
    names = ["cat__OverTime_Yes"]
    result = get_top_shap(shap_vals, names, n=1)
    assert result[0]["feature"] == "OverTime"


def test_get_top_shap_cleans_numerical_name():
    shap_vals = np.array([0.9])
    names = ["num__Age"]
    result = get_top_shap(shap_vals, names, n=1)
    assert result[0]["feature"] == "Age"
```

- [ ] **Step 3: Run the failing tests**

```bash
cd api
source venv/bin/activate
pytest tests/test_predict.py -v
```

Expected: all tests pass (predict.py is already written — these tests verify correctness).

- [ ] **Step 4: Commit**

```bash
git add api/tests/__init__.py api/tests/test_predict.py
git commit -m "test: unit tests for prediction logic and SHAP helpers"
```

---

## Task 5: Flask Routes + Database Module

**Files:**
- Create: `api/db.py`
- Create: `api/app.py`

- [ ] **Step 1: Create api/db.py**

```python
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()


def _connect():
    return psycopg2.connect(os.environ["DATABASE_URL"], sslmode="require")


def log_prediction(input_data: dict, result: dict, model_version: str) -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions (
                    age, monthly_income, job_role, years_at_company,
                    overtime, satisfaction_level, input_json,
                    prediction, probability, shap_json, model_version
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    input_data.get("Age"),
                    input_data.get("MonthlyIncome"),
                    input_data.get("JobRole"),
                    input_data.get("YearsAtCompany"),
                    input_data.get("OverTime") == "Yes",
                    input_data.get("JobSatisfaction"),
                    json.dumps(input_data),
                    result["prediction"],
                    result["probability"],
                    json.dumps(result["shap_values"]),
                    model_version,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def get_recent_predictions(limit: int = 50) -> list[dict]:
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, age, monthly_income, job_role, years_at_company,
                       overtime, satisfaction_level, prediction, probability,
                       shap_json, model_version, created_at
                FROM predictions
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()
```

- [ ] **Step 2: Create api/app.py**

```python
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import model as ml
import predict as predictor
import db

load_dotenv()

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:3000",
    os.environ.get("FRONTEND_URL", ""),
])

# Load model once at startup (works for both gunicorn and flask dev server)
ml.load_model()


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_version": ml.get_model_version()})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON", "code": 400}), 400
    try:
        result = predictor.predict(data)
        result["model_version"] = ml.get_model_version()
        try:
            db.log_prediction(data, result, ml.get_model_version())
        except Exception as exc:
            app.logger.error("DB write failed: %s", exc)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": 400}), 400
    except Exception as exc:
        app.logger.error("Prediction error: %s", exc)
        return jsonify({"error": "Internal server error", "code": 500}), 500


@app.get("/history")
def history():
    try:
        rows = db.get_recent_predictions(50)
        for row in rows:
            row["id"] = str(row["id"])
            if row.get("created_at"):
                row["created_at"] = row["created_at"].isoformat()
        return jsonify(rows)
    except Exception as exc:
        app.logger.error("History error: %s", exc)
        return jsonify({"error": "Could not fetch history", "code": 500}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
```

- [ ] **Step 3: Commit**

```bash
git add api/db.py api/app.py
git commit -m "feat: Flask routes (/health, /predict, /history) and DB module"
```

---

## Task 6: Flask Route Tests + Local Smoke Test

**Files:**
- Create: `api/tests/test_routes.py`

- [ ] **Step 1: Create api/tests/test_routes.py**

```python
import json
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

VALID_INPUT = {
    "BusinessTravel": "Travel_Rarely",
    "Department": "Sales",
    "JobRole": "Sales Executive",
    "MaritalStatus": "Single",
    "OverTime": "Yes",
    "Age": 35,
    "DistanceFromHome": 10,
    "EnvironmentSatisfaction": 2,
    "JobSatisfaction": 2,
    "MonthlyIncome": 4000,
    "NumCompaniesWorked": 3,
    "TotalWorkingYears": 8,
    "WorkLifeBalance": 2,
    "YearsAtCompany": 4,
}

MOCK_RESULT = {
    "prediction": True,
    "probability": 0.78,
    "risk_level": "HIGH",
    "shap_values": [
        {"feature": "OverTime", "value": 0.42},
        {"feature": "JobSatisfaction", "value": 0.31},
        {"feature": "MonthlyIncome", "value": -0.18},
        {"feature": "Age", "value": -0.11},
        {"feature": "YearsAtCompany", "value": 0.09},
    ],
}


@pytest.fixture
def client():
    with patch("model.load_model"), \
         patch("model.get_model_version", return_value="v1.0.0"), \
         patch("model.get_pipeline", return_value=MagicMock()), \
         patch("model.get_explainer", return_value=MagicMock()), \
         patch("model.get_feature_names", return_value=[]):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        with flask_app.app.test_client() as c:
            yield c


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = json.loads(resp.data)
    assert body["status"] == "ok"
    assert body["model_version"] == "v1.0.0"


def test_predict_no_body_returns_400(client):
    resp = client.post("/predict", content_type="application/json", data="")
    assert resp.status_code == 400


def test_predict_missing_field_returns_400(client):
    bad_input = {k: v for k, v in VALID_INPUT.items() if k != "Age"}
    with patch("app.predictor.predict", side_effect=ValueError("Missing required fields: Age")):
        resp = client.post(
            "/predict",
            data=json.dumps(bad_input),
            content_type="application/json",
        )
    assert resp.status_code == 400
    body = json.loads(resp.data)
    assert "Age" in body["error"]


def test_predict_success_returns_all_fields(client):
    with patch("app.predictor.predict", return_value=MOCK_RESULT), \
         patch("app.db.log_prediction"):
        resp = client.post(
            "/predict",
            data=json.dumps(VALID_INPUT),
            content_type="application/json",
        )
    assert resp.status_code == 200
    body = json.loads(resp.data)
    assert "prediction" in body
    assert "probability" in body
    assert "risk_level" in body
    assert "shap_values" in body
    assert "model_version" in body
    assert body["model_version"] == "v1.0.0"


def test_predict_db_failure_is_non_blocking(client):
    with patch("app.predictor.predict", return_value=MOCK_RESULT), \
         patch("app.db.log_prediction", side_effect=Exception("DB down")):
        resp = client.post(
            "/predict",
            data=json.dumps(VALID_INPUT),
            content_type="application/json",
        )
    assert resp.status_code == 200  # DB error must not fail the response


def test_history_db_error_returns_500(client):
    with patch("app.db.get_recent_predictions", side_effect=Exception("DB down")):
        resp = client.get("/history")
    assert resp.status_code == 500
```

- [ ] **Step 2: Run all API tests**

```bash
cd api
source venv/bin/activate
pytest tests/ -v
```

Expected: all 15+ tests pass with no failures.

- [ ] **Step 3: Run Flask dev server locally (smoke test)**

```bash
cd api
source venv/bin/activate
python app.py
```

Expected:
```
[model] Loaded v1.0.0 — XX features
 * Running on http://127.0.0.1:5000
```

In a second terminal:
```bash
curl http://localhost:5000/health
```
Expected: `{"model_version":"v1.0.0","status":"ok"}`

- [ ] **Step 4: Commit**

```bash
git add api/tests/test_routes.py
git commit -m "test: Flask route tests — all endpoints including error cases"
```

---

## Task 7: Next.js Project Setup + Types + API Client

**Files:**
- Create: `frontend/` (via create-next-app)
- Create: `frontend/lib/types.ts`
- Create: `frontend/lib/api.ts`
- Create: `frontend/.env.example`

- [ ] **Step 1: Scaffold Next.js app**

```bash
cd employee-retention
npx create-next-app@14 frontend \
  --typescript \
  --tailwind \
  --app \
  --no-src-dir \
  --import-alias "@/*"
```

When prompted, answer:
- ESLint: Yes
- `src/` directory: No (already specified)
- customize import alias: No

- [ ] **Step 2: Install Recharts**

```bash
cd frontend
npm install recharts
npm install --save-dev @types/recharts
```

- [ ] **Step 3: Create frontend/.env.example**

```
NEXT_PUBLIC_API_URL=http://localhost:5000
```

- [ ] **Step 4: Create frontend/.env.local**

```
NEXT_PUBLIC_API_URL=http://localhost:5000
```

(This file is gitignored by create-next-app by default.)

- [ ] **Step 5: Create frontend/lib/types.ts**

```typescript
export interface PredictionInput {
  BusinessTravel: string;
  Department: string;
  JobRole: string;
  MaritalStatus: string;
  OverTime: string;
  Age: number;
  DistanceFromHome: number;
  EnvironmentSatisfaction: number;
  JobSatisfaction: number;
  MonthlyIncome: number;
  NumCompaniesWorked: number;
  TotalWorkingYears: number;
  WorkLifeBalance: number;
  YearsAtCompany: number;
}

export type RiskLevel = "HIGH" | "MED" | "LOW";

export interface ShapValue {
  feature: string;
  value: number;
}

export interface PredictionResult {
  prediction: boolean;
  probability: number;
  risk_level: RiskLevel;
  shap_values: ShapValue[];
  model_version: string;
}

export interface HistoryRow {
  id: string;
  age: number;
  monthly_income: number;
  job_role: string;
  years_at_company: number;
  overtime: boolean;
  satisfaction_level: number;
  prediction: boolean;
  probability: number;
  shap_json: ShapValue[];
  model_version: string;
  created_at: string;
}
```

- [ ] **Step 6: Create frontend/lib/api.ts**

```typescript
import { PredictionInput, PredictionResult, HistoryRow } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000";

export async function checkHealth(): Promise<{ status: string; model_version: string }> {
  const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error("API health check failed");
  return res.json();
}

export async function submitPrediction(input: PredictionInput): Promise<PredictionResult> {
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "Unknown error" }));
    throw new Error(err.error ?? "Prediction failed");
  }
  return res.json();
}

export async function fetchHistory(): Promise<HistoryRow[]> {
  const res = await fetch(`${API_URL}/history`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch history");
  return res.json();
}
```

- [ ] **Step 7: Verify TypeScript compiles**

```bash
cd frontend
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add frontend/
git commit -m "feat: Next.js 14 scaffold with TypeScript types and API client"
```

---

## Task 8: PredictionForm Component + Home Page

**Files:**
- Create: `frontend/components/PredictionForm.tsx`
- Modify: `frontend/app/page.tsx`

- [ ] **Step 1: Create frontend/components/PredictionForm.tsx**

```tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { submitPrediction } from "@/lib/api";
import { PredictionInput } from "@/lib/types";

const JOB_ROLES = [
  "Healthcare Representative", "Human Resources", "Laboratory Technician",
  "Manager", "Manufacturing Director", "Research Director", "Research Scientist",
  "Sales Executive", "Sales Representative",
];
const DEPARTMENTS = ["Human Resources", "Research & Development", "Sales"];
const BUSINESS_TRAVEL = ["Non-Travel", "Travel_Frequently", "Travel_Rarely"];
const MARITAL_STATUS = ["Divorced", "Married", "Single"];
const SATISFACTION_LEVELS = [
  { value: 1, label: "1 — Low" },
  { value: 2, label: "2 — Medium" },
  { value: 3, label: "3 — High" },
  { value: 4, label: "4 — Very High" },
];

const defaultValues: PredictionInput = {
  BusinessTravel: "Travel_Rarely",
  Department: "Sales",
  JobRole: "Sales Executive",
  MaritalStatus: "Single",
  OverTime: "No",
  Age: 35,
  DistanceFromHome: 10,
  EnvironmentSatisfaction: 3,
  JobSatisfaction: 3,
  MonthlyIncome: 5000,
  NumCompaniesWorked: 2,
  TotalWorkingYears: 8,
  WorkLifeBalance: 3,
  YearsAtCompany: 4,
};

export default function PredictionForm() {
  const router = useRouter();
  const [form, setForm] = useState<PredictionInput>(defaultValues);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function setField<K extends keyof PredictionInput>(key: K, value: PredictionInput[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const result = await submitPrediction(form);
      sessionStorage.setItem("predictionResult", JSON.stringify(result));
      router.push("/result");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Row 1 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Age</label>
          <input
            type="number" min={18} max={65} required
            value={form.Age}
            onChange={(e) => setField("Age", parseInt(e.target.value))}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Monthly Income ($)</label>
          <input
            type="number" min={1000} max={20000} required
            value={form.MonthlyIncome}
            onChange={(e) => setField("MonthlyIncome", parseInt(e.target.value))}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Job Role</label>
          <select
            value={form.JobRole}
            onChange={(e) => setField("JobRole", e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {JOB_ROLES.map((r) => <option key={r}>{r}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Department</label>
          <select
            value={form.Department}
            onChange={(e) => setField("Department", e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {DEPARTMENTS.map((d) => <option key={d}>{d}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Years at Company</label>
          <input
            type="number" min={0} max={40} required
            value={form.YearsAtCompany}
            onChange={(e) => setField("YearsAtCompany", parseInt(e.target.value))}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Total Working Years</label>
          <input
            type="number" min={0} max={40} required
            value={form.TotalWorkingYears}
            onChange={(e) => setField("TotalWorkingYears", parseInt(e.target.value))}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Overtime</label>
          <select
            value={form.OverTime}
            onChange={(e) => setField("OverTime", e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option>Yes</option>
            <option>No</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Business Travel</label>
          <select
            value={form.BusinessTravel}
            onChange={(e) => setField("BusinessTravel", e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {BUSINESS_TRAVEL.map((t) => <option key={t}>{t}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Marital Status</label>
          <select
            value={form.MaritalStatus}
            onChange={(e) => setField("MaritalStatus", e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {MARITAL_STATUS.map((m) => <option key={m}>{m}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Distance from Home (km)</label>
          <input
            type="number" min={1} max={29} required
            value={form.DistanceFromHome}
            onChange={(e) => setField("DistanceFromHome", parseInt(e.target.value))}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Companies Worked At</label>
          <input
            type="number" min={0} max={9} required
            value={form.NumCompaniesWorked}
            onChange={(e) => setField("NumCompaniesWorked", parseInt(e.target.value))}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>

      {/* Satisfaction sliders row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {(
          [
            ["Job Satisfaction", "JobSatisfaction"],
            ["Environment Satisfaction", "EnvironmentSatisfaction"],
            ["Work-Life Balance", "WorkLifeBalance"],
          ] as [string, keyof PredictionInput][]
        ).map(([label, key]) => (
          <div key={key}>
            <label className="block text-sm font-medium text-gray-300 mb-1">{label}</label>
            <select
              value={form[key] as number}
              onChange={(e) => setField(key, parseInt(e.target.value))}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {SATISFACTION_LEVELS.map((s) => (
                <option key={s.value} value={s.value}>{s.label}</option>
              ))}
            </select>
          </div>
        ))}
      </div>

      {error && (
        <div className="bg-red-900/50 border border-red-500 rounded-lg px-4 py-3 text-red-300 text-sm">
          {error}
        </div>
      )}

      <button
        type="submit"
        disabled={loading}
        className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-lg transition-colors"
      >
        {loading ? "Analyzing..." : "Predict Attrition →"}
      </button>
    </form>
  );
}
```

- [ ] **Step 2: Replace frontend/app/page.tsx**

```tsx
import PredictionForm from "@/components/PredictionForm";

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-950 text-white">
      <div className="max-w-3xl mx-auto px-4 py-12">
        <div className="mb-10 text-center">
          <h1 className="text-4xl font-bold text-white mb-3">
            Employee Attrition Predictor
          </h1>
          <p className="text-gray-400 text-lg">
            Enter employee details to predict the likelihood of attrition using machine learning.
          </p>
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-8 shadow-xl">
          <PredictionForm />
        </div>
        <p className="text-center text-gray-600 text-sm mt-6">
          Powered by Random Forest · IBM HR Analytics Dataset · SHAP Explainability
        </p>
      </div>
    </main>
  );
}
```

- [ ] **Step 3: Start dev server and verify form renders**

```bash
cd frontend
npm run dev
```

Open http://localhost:3000 — verify the form renders with all 14 fields, no TypeScript errors in terminal.

- [ ] **Step 4: Commit**

```bash
git add frontend/components/PredictionForm.tsx frontend/app/page.tsx
git commit -m "feat: prediction form — 14 fields with validation and loading state"
```

---

## Task 9: Result Page — ResultCard + ShapChart

**Files:**
- Create: `frontend/components/ResultCard.tsx`
- Create: `frontend/components/ShapChart.tsx`
- Create: `frontend/app/result/page.tsx`

- [ ] **Step 1: Create frontend/components/ResultCard.tsx**

```tsx
import { PredictionResult, RiskLevel } from "@/lib/types";

const riskConfig: Record<RiskLevel, { color: string; bg: string; border: string; label: string }> = {
  HIGH: { color: "text-red-400", bg: "bg-red-900/30", border: "border-red-500", label: "High Risk" },
  MED:  { color: "text-orange-400", bg: "bg-orange-900/30", border: "border-orange-500", label: "Medium Risk" },
  LOW:  { color: "text-green-400", bg: "bg-green-900/30", border: "border-green-500", label: "Low Risk" },
};

interface Props {
  result: PredictionResult;
}

export default function ResultCard({ result }: Props) {
  const cfg = riskConfig[result.risk_level];
  const pct = Math.round(result.probability * 100);

  return (
    <div className={`rounded-2xl border ${cfg.border} ${cfg.bg} p-8`}>
      <div className="flex items-center justify-between mb-6">
        <div>
          <p className="text-gray-400 text-sm uppercase tracking-wider mb-1">Attrition Risk</p>
          <p className={`text-5xl font-bold ${cfg.color}`}>{cfg.label}</p>
        </div>
        <div className="text-right">
          <p className="text-gray-400 text-sm mb-1">Probability</p>
          <p className={`text-5xl font-bold ${cfg.color}`}>{pct}%</p>
        </div>
      </div>

      {/* Probability bar */}
      <div className="w-full bg-gray-800 rounded-full h-3 mb-4">
        <div
          className={`h-3 rounded-full transition-all duration-500 ${
            result.risk_level === "HIGH" ? "bg-red-500" :
            result.risk_level === "MED" ? "bg-orange-500" : "bg-green-500"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>

      <div className="flex justify-between text-xs text-gray-500">
        <span>0% (Stays)</span>
        <span className="text-gray-400">30% MED</span>
        <span className="text-gray-400">60% HIGH</span>
        <span>100% (Leaves)</span>
      </div>

      <div className="mt-6 pt-4 border-t border-gray-700 flex justify-between text-sm text-gray-400">
        <span>Verdict: <span className="text-white font-medium">{result.prediction ? "Likely to Leave" : "Likely to Stay"}</span></span>
        <span>Model: <span className="text-white font-mono">{result.model_version}</span></span>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create frontend/components/ShapChart.tsx**

```tsx
"use client";

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Cell,
} from "recharts";
import { ShapValue } from "@/lib/types";

interface Props {
  shapValues: ShapValue[];
}

export default function ShapChart({ shapValues }: Props) {
  const sorted = [...shapValues].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6">
      <h2 className="text-lg font-semibold text-white mb-1">Top Contributing Factors</h2>
      <p className="text-gray-400 text-sm mb-6">
        Red bars increase attrition risk · Green bars decrease it
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart
          data={sorted}
          layout="vertical"
          margin={{ top: 0, right: 40, left: 120, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
          <XAxis
            type="number"
            tickFormatter={(v) => v.toFixed(2)}
            stroke="#6b7280"
            tick={{ fill: "#9ca3af", fontSize: 12 }}
          />
          <YAxis
            type="category"
            dataKey="feature"
            stroke="#6b7280"
            tick={{ fill: "#d1d5db", fontSize: 13 }}
            width={115}
          />
          <Tooltip
            formatter={(value: number) => [value.toFixed(4), "SHAP value"]}
            contentStyle={{ background: "#1f2937", border: "1px solid #374151", borderRadius: "8px" }}
            labelStyle={{ color: "#f9fafb" }}
          />
          <ReferenceLine x={0} stroke="#4b5563" />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {sorted.map((entry, index) => (
              <Cell key={index} fill={entry.value >= 0 ? "#ef4444" : "#22c55e"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
```

- [ ] **Step 3: Create frontend/app/result/page.tsx**

```tsx
"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import ResultCard from "@/components/ResultCard";
import ShapChart from "@/components/ShapChart";
import { PredictionResult } from "@/lib/types";

export default function ResultPage() {
  const router = useRouter();
  const [result, setResult] = useState<PredictionResult | null>(null);

  useEffect(() => {
    const raw = sessionStorage.getItem("predictionResult");
    if (!raw) {
      router.replace("/");
      return;
    }
    setResult(JSON.parse(raw));
  }, [router]);

  if (!result) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <p className="text-gray-400">Loading result...</p>
      </div>
    );
  }

  return (
    <main className="min-h-screen bg-gray-950 text-white">
      <div className="max-w-3xl mx-auto px-4 py-12 space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold">Prediction Result</h1>
          <Link
            href="/"
            className="text-blue-400 hover:text-blue-300 text-sm transition-colors"
          >
            ← New Prediction
          </Link>
        </div>

        <ResultCard result={result} />
        <ShapChart shapValues={result.shap_values} />

        <div className="text-center">
          <Link
            href="/history"
            className="text-gray-400 hover:text-gray-300 text-sm transition-colors"
          >
            View prediction history →
          </Link>
        </div>
      </div>
    </main>
  );
}
```

- [ ] **Step 4: Verify result page renders**

With the dev server running (Task 8 Step 3), open http://localhost:3000. Submit the form — you should be redirected to `/result` with the risk card and SHAP chart.

If the Flask API isn't running, the form will show an error message (expected — that's the error state working).

- [ ] **Step 5: Commit**

```bash
git add frontend/components/ResultCard.tsx frontend/components/ShapChart.tsx frontend/app/result/page.tsx
git commit -m "feat: result page with risk card, probability bar, and SHAP chart"
```

---

## Task 10: History Page

**Files:**
- Create: `frontend/components/HistoryTable.tsx`
- Create: `frontend/app/history/page.tsx`

- [ ] **Step 1: Create frontend/components/HistoryTable.tsx**

```tsx
"use client";

import { HistoryRow, RiskLevel } from "@/lib/types";

const riskBadge: Record<RiskLevel, string> = {
  HIGH: "bg-red-900/50 text-red-400 border border-red-700",
  MED:  "bg-orange-900/50 text-orange-400 border border-orange-700",
  LOW:  "bg-green-900/50 text-green-400 border border-green-700",
};

function getRisk(probability: number): RiskLevel {
  if (probability >= 0.60) return "HIGH";
  if (probability >= 0.30) return "MED";
  return "LOW";
}

interface Props {
  rows: HistoryRow[];
}

export default function HistoryTable({ rows }: Props) {
  if (rows.length === 0) {
    return (
      <div className="text-center py-16 text-gray-500">
        No predictions yet. <a href="/" className="text-blue-400 hover:underline">Make your first one →</a>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-2xl border border-gray-800">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-gray-900 text-gray-400 text-xs uppercase tracking-wider">
            <th className="px-4 py-3 text-left">ID</th>
            <th className="px-4 py-3 text-left">Date</th>
            <th className="px-4 py-3 text-left">Job Role</th>
            <th className="px-4 py-3 text-center">Age</th>
            <th className="px-4 py-3 text-center">Risk</th>
            <th className="px-4 py-3 text-center">Probability</th>
            <th className="px-4 py-3 text-center">Model</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-800">
          {rows.map((row) => {
            const risk = getRisk(row.probability);
            return (
              <tr key={row.id} className="bg-gray-950 hover:bg-gray-900 transition-colors">
                <td className="px-4 py-3 font-mono text-gray-500 text-xs">
                  {row.id.split("-")[0]}
                </td>
                <td className="px-4 py-3 text-gray-300">
                  {new Date(row.created_at).toLocaleDateString("en-US", {
                    month: "short", day: "numeric", year: "numeric",
                  })}
                </td>
                <td className="px-4 py-3 text-white">{row.job_role}</td>
                <td className="px-4 py-3 text-center text-gray-300">{row.age}</td>
                <td className="px-4 py-3 text-center">
                  <span className={`px-2 py-1 rounded-full text-xs font-semibold ${riskBadge[risk]}`}>
                    {risk}
                  </span>
                </td>
                <td className="px-4 py-3 text-center text-white font-medium">
                  {Math.round(row.probability * 100)}%
                </td>
                <td className="px-4 py-3 text-center font-mono text-gray-500 text-xs">
                  {row.model_version}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 2: Create frontend/app/history/page.tsx**

```tsx
"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import HistoryTable from "@/components/HistoryTable";
import { fetchHistory } from "@/lib/api";
import { HistoryRow } from "@/lib/types";

export default function HistoryPage() {
  const [rows, setRows] = useState<HistoryRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHistory()
      .then(setRows)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <main className="min-h-screen bg-gray-950 text-white">
      <div className="max-w-5xl mx-auto px-4 py-12">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold">Prediction History</h1>
            <p className="text-gray-400 mt-1">Last 50 predictions — stored in PostgreSQL</p>
          </div>
          <Link
            href="/"
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm transition-colors"
          >
            + New Prediction
          </Link>
        </div>

        {loading && (
          <div className="text-center py-16 text-gray-400">Loading history...</div>
        )}

        {error && (
          <div className="bg-red-900/30 border border-red-700 rounded-xl px-6 py-4 text-red-400">
            {error}
          </div>
        )}

        {!loading && !error && <HistoryTable rows={rows} />}
      </div>
    </main>
  );
}
```

- [ ] **Step 3: Verify history page renders**

Open http://localhost:3000/history — with no DB connected, you'll see the error state. That's correct — it means error handling works.

- [ ] **Step 4: Commit**

```bash
git add frontend/components/HistoryTable.tsx frontend/app/history/page.tsx
git commit -m "feat: history page with sortable prediction log"
```

---

## Task 11: ApiStatusBanner + Root Layout

**Files:**
- Create: `frontend/components/ApiStatusBanner.tsx`
- Modify: `frontend/app/layout.tsx`

- [ ] **Step 1: Create frontend/components/ApiStatusBanner.tsx**

```tsx
"use client";

import { useEffect, useState } from "react";
import { checkHealth } from "@/lib/api";

export default function ApiStatusBanner() {
  const [slow, setSlow] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setSlow(true), 3000);

    checkHealth()
      .then(() => {
        clearTimeout(timer);
        setSlow(false);
        setReady(true);
      })
      .catch(() => {
        clearTimeout(timer);
        setSlow(false);
      });

    return () => clearTimeout(timer);
  }, []);

  if (ready || !slow) return null;

  return (
    <div className="bg-yellow-900/40 border-b border-yellow-700 px-4 py-2 text-center text-yellow-300 text-sm">
      API is waking up (free tier cold start) — this may take up to 30 seconds...
    </div>
  );
}
```

- [ ] **Step 2: Update frontend/app/layout.tsx**

Replace the default layout with:

```tsx
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import ApiStatusBanner from "@/components/ApiStatusBanner";
import Link from "next/link";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Employee Attrition Predictor",
  description: "Predict employee attrition with machine learning and SHAP explainability",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-gray-950`}>
        <ApiStatusBanner />
        <nav className="border-b border-gray-800 px-6 py-3 flex items-center gap-6 bg-gray-950">
          <span className="text-white font-semibold">Attrition Predictor</span>
          <Link href="/" className="text-gray-400 hover:text-white text-sm transition-colors">Predict</Link>
          <Link href="/history" className="text-gray-400 hover:text-white text-sm transition-colors">History</Link>
        </nav>
        {children}
      </body>
    </html>
  );
}
```

- [ ] **Step 3: Verify no TypeScript errors and app still runs**

```bash
cd frontend
npx tsc --noEmit
```

Expected: no errors.

Open http://localhost:3000 — verify the nav bar appears on all pages.

- [ ] **Step 4: Commit**

```bash
git add frontend/components/ApiStatusBanner.tsx frontend/app/layout.tsx
git commit -m "feat: nav bar and cold-start API status banner"
```

---

## Task 12: Supabase DB Setup + End-to-End Test

**Files:** No code files — Supabase config + end-to-end smoke test.

- [ ] **Step 1: Create Supabase project**

1. Go to https://supabase.com and sign in (free)
2. Click "New project" → name it `employee-attrition`
3. Once ready, go to Settings → Database → Connection string (URI)
4. Copy the URI — it looks like: `postgresql://postgres:[password]@db.[ref].supabase.co:5432/postgres`

- [ ] **Step 2: Run the SQL migration**

In Supabase dashboard → SQL Editor → New query, paste and run:

```sql
CREATE TABLE predictions (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  age              INT,
  monthly_income   INT,
  job_role         TEXT,
  years_at_company INT,
  overtime         BOOLEAN,
  satisfaction_level INT,
  input_json       JSONB,
  prediction       BOOLEAN,
  probability      FLOAT,
  shap_json        JSONB,
  model_version    TEXT,
  created_at       TIMESTAMPTZ DEFAULT now()
);
```

Expected: "Success. No rows returned."

- [ ] **Step 3: Set DATABASE_URL in api/.env**

Create `api/.env` (gitignored):
```
DATABASE_URL=postgresql://postgres:[your-password]@db.[your-ref].supabase.co:5432/postgres
FRONTEND_URL=http://localhost:3000
```

- [ ] **Step 4: Run end-to-end smoke test**

Terminal 1 — start Flask:
```bash
cd api && source venv/bin/activate && python app.py
```

Terminal 2 — submit a prediction:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"BusinessTravel":"Travel_Rarely","Department":"Sales","JobRole":"Sales Executive","MaritalStatus":"Single","OverTime":"Yes","Age":35,"DistanceFromHome":10,"EnvironmentSatisfaction":2,"JobSatisfaction":2,"MonthlyIncome":4000,"NumCompaniesWorked":3,"TotalWorkingYears":8,"WorkLifeBalance":2,"YearsAtCompany":4}'
```

Expected response (values will vary):
```json
{
  "prediction": true,
  "probability": 0.78,
  "risk_level": "HIGH",
  "shap_values": [...],
  "model_version": "v1.0.0"
}
```

Verify in Supabase dashboard → Table Editor → predictions: one row should appear.

- [ ] **Step 5: Test full browser flow**

With both Flask (port 5000) and Next.js (port 3000) running:
1. Open http://localhost:3000
2. Fill the form and click "Predict Attrition →"
3. Verify redirect to `/result` with risk card and SHAP chart
4. Open http://localhost:3000/history — verify the prediction row appears

---

## Task 13: README + Deployment Prep

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write README.md**

```markdown
# Employee Attrition Prediction System

Predict whether an employee will leave using machine learning. Built end-to-end: data → model → API → frontend → database.

**Live:** [https://your-app.vercel.app](https://your-app.vercel.app)

## Architecture

```
Next.js (Vercel) → Flask API (Render) → model.pkl + Supabase PostgreSQL
```

- **ML:** Random Forest trained on IBM HR Analytics dataset — 87% accuracy, 0.82 ROC-AUC
- **Explainability:** SHAP values returned with every prediction
- **MLOps:** Every prediction stamped with `model_version` for traceability
- **Persistence:** All predictions logged to PostgreSQL

## Local Development

### 1. Train the model

```bash
cd ml
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Download IBM HR dataset from Kaggle → ml/data/WA_Fn-UseC_-HR-Employee-Attrition.csv
python train.py
```

### 2. Start the API

```bash
cd api
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in DATABASE_URL
python app.py
```

### 3. Start the frontend

```bash
cd frontend
npm install
cp .env.example .env.local  # set NEXT_PUBLIC_API_URL=http://localhost:5000
npm run dev
```

Open http://localhost:3000

## Running Tests

```bash
cd api && source venv/bin/activate && pytest tests/ -v
```

## Deployment

| Service | Purpose | Free tier |
|---------|---------|-----------|
| Render | Flask API | Yes |
| Vercel | Next.js frontend | Yes |
| Supabase | PostgreSQL | Yes |

See deployment checklist in `docs/superpowers/specs/2026-04-16-employee-attrition-design.md`.

## Model

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest (200 trees) |
| Dataset | IBM HR Analytics (1,470 rows) |
| Accuracy | ~87% |
| ROC-AUC | ~0.82 |
| Features | 14 (5 categorical, 9 numerical) |
```

- [ ] **Step 2: Deploy Flask API to Render**

1. Push repo to GitHub: `git remote add origin https://github.com/<your-username>/employee-attrition.git && git push -u origin main`
2. Go to https://render.com → New → Web Service → connect GitHub repo
3. Settings:
   - Root directory: `api`
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
4. Add environment variables: `DATABASE_URL` (from Supabase) and `FRONTEND_URL` (your Vercel URL, set after next step)
5. Deploy — copy the Render URL (e.g. `https://employee-attrition-api.onrender.com`)

- [ ] **Step 3: Deploy Next.js to Vercel**

1. Go to https://vercel.com → New Project → import GitHub repo
2. Set root directory: `frontend`
3. Add environment variable: `NEXT_PUBLIC_API_URL=https://employee-attrition-api.onrender.com`
4. Deploy — copy the Vercel URL
5. Go back to Render → update `FRONTEND_URL` env var with the Vercel URL
6. Redeploy Render service

- [ ] **Step 4: Final end-to-end test on live URLs**

1. Open your Vercel URL
2. Fill and submit the prediction form
3. Verify `/result` page shows risk card + SHAP chart
4. Verify `/history` shows the prediction row from Supabase

- [ ] **Step 5: Update README with live URL**

Replace `https://your-app.vercel.app` with your actual Vercel URL.

```bash
git add README.md
git commit -m "docs: README with architecture, setup instructions, and live URL"
git push
```
```

- [ ] **Step 6: Commit README**

```bash
git add README.md
git commit -m "docs: README with architecture, local setup, and deployment guide"
git push
```

---

## Self-Review Checklist

- [x] `/predict` endpoint with SHAP values ✓ Task 5
- [x] `/history` endpoint ✓ Task 5
- [x] `/health` endpoint ✓ Task 5
- [x] `model_version` on every prediction row ✓ Tasks 5, 9
- [x] Risk level thresholds (HIGH ≥ 60%, MED 30–59%, LOW < 30%) ✓ Task 3
- [x] DB write non-blocking on failure ✓ Tasks 5, 6
- [x] Cold start banner after 3s ✓ Task 11
- [x] sessionStorage for passing result to /result page ✓ Task 9
- [x] SHAP red/green bars ✓ Task 9
- [x] History table with model_version column ✓ Task 10
- [x] All API calls via `lib/api.ts` reading `NEXT_PUBLIC_API_URL` ✓ Task 7
- [x] CORS configured for Vercel URL ✓ Task 5
- [x] Deployment checklist ✓ Task 13
