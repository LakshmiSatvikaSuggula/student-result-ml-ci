import logging
import sys
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g
from flasgger import Swagger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("student-result-api")

app = Flask(__name__)
swagger = Swagger(app, template={
    "info": {
        "title": "Customer Churn Prediction API",
        "description": "API for predicting customer churn based on ML model.",
        "version": "1.0.0"
    }
})

MODEL_PATH = Path("student_result_model.pkl")

NUMERICAL_FEATURES = ["age", "tenure_months", "monthly_charges", "total_charges", "num_support_calls", "satisfaction_score"]
CATEGORICAL_FEATURES = ["gender", "contract_type", "payment_method", "internet_service", "tech_support"]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

try:
    model = joblib.load(MODEL_PATH)
    logger.info("event=model_loaded model_file=%s", MODEL_PATH)
except Exception:
    logger.exception("event=model_load_failed model_file=%s", MODEL_PATH)
    raise

@app.before_request
def _start_timer():
    g.start_time = time.perf_counter()
    g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

@app.after_request
def _log_request(response):
    if request.path.startswith('/flasgger'):
        return response
    duration_ms = (time.perf_counter() - g.start_time) * 1000
    level = logging.WARNING if response.status_code >= 400 else logging.INFO
    logger.log(level, "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f", g.request_id, request.method, request.path, response.status_code, duration_ms)
    response.headers["X-Request-ID"] = g.request_id
    return response

@app.get("/")
@app.get("/health")
def health_check():
    """Health Check Endpoint
    ---
    responses:
      200:
        description: Returns the API status
    """
    return jsonify({"status": "ok", "service": "customer-churn-prediction"})

@app.post("/predict")
def predict():
    """Churn Prediction Endpoint
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            age: {type: integer, example: 30}
            tenure_months: {type: integer, example: 2}
            monthly_charges: {type: number, example: 95.0}
            total_charges: {type: number, example: 200.0}
            num_support_calls: {type: integer, example: 8}
            satisfaction_score: {type: integer, example: 2}
            gender: {type: string, example: "Male"}
            contract_type: {type: string, example: "Month-to-month"}
            payment_method: {type: string, example: "Electronic check"}
            internet_service: {type: string, example: "Fiber optic"}
            tech_support: {type: string, example: "No"}
    responses:
      200: {description: Successful prediction}
      400: {description: Missing fields}
    """
    data = request.get_json(silent=True)
    if not data:
        logger.warning("event=validation_error reason=no_json path=/predict")
        return jsonify({"error": "JSON request body is required"}), 400

    missing_fields = [f for f in ALL_FEATURES if f not in data]
    if missing_fields:
        logger.warning("event=validation_error path=/predict missing_fields=%s", missing_fields)
        return jsonify({"error": "Missing required fields", "missing_fields": missing_fields}), 400

    try:
        sample = pd.DataFrame([{f: data[f] for f in ALL_FEATURES}])
        prediction_code = int(model.predict(sample)[0])
        prediction = "CHURN" if prediction_code == 1 else "NO CHURN"
        logger.info("event=prediction_success prediction=%s prediction_code=%d", prediction, prediction_code)
        return jsonify({"prediction": prediction, "prediction_code": prediction_code})
    except Exception:
        logger.exception("event=prediction_failed")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
