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
