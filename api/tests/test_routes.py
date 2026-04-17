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
        import importlib
        importlib.reload(flask_app)
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
