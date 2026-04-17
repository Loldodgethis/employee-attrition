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
