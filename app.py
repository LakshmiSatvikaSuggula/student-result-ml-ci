from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_PATH = Path("student_result_model.pkl")

NUMERICAL_FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    "satisfaction_score",
]

CATEGORICAL_FEATURES = [
    "gender",
    "contract_type",
    "payment_method",
    "internet_service",
    "tech_support",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "student_result_model.pkl was not found. "
            "Run the training pipeline first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return jsonify({
    "status": "ok",
    "service": "customer-churn-prediction",
    "version": "v2"
})

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON request body is required"}), 400

    missing_fields = [f for f in ALL_FEATURES if f not in data]
    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    sample = pd.DataFrame([{f: data[f] for f in ALL_FEATURES}])
    model = load_model()
    prediction_code = int(model.predict(sample)[0])
    prediction = "CHURN" if prediction_code == 1 else "NO CHURN"

    return jsonify({
        "prediction": prediction,
        "prediction_code": prediction_code
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)