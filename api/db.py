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
