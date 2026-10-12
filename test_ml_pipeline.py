import unittest
import os
import json
import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_created(self):
        self.assertTrue(os.path.exists("student_results.csv"))

    def test_model_created(self):
        self.assertTrue(os.path.exists("student_result_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)
        accuracy = metrics["accuracy"]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):
        model = joblib.load("student_result_model.pkl")
        sample = pd.DataFrame([{
            "age": 45,
            "tenure_months": 12,
            "monthly_charges": 70.0,
            "total_charges": 840.0,
            "num_support_calls": 3,
            "satisfaction_score": 5,
            "gender": "Male",
            "contract_type": "Month-to-month",
            "payment_method": "Electronic check",
            "internet_service": "Fiber optic",
            "tech_support": "No",
        }])
        prediction = model.predict(sample)[0]
        self.assertIn(int(prediction), [0, 1])

    def test_high_risk_customer(self):
        model = joblib.load("student_result_model.pkl")
        sample = pd.DataFrame([{
            "age": 25,
            "tenure_months": 2,
            "monthly_charges": 100.0,
            "total_charges": 200.0,
            "num_support_calls": 8,
            "satisfaction_score": 1,
            "gender": "Male",
            "contract_type": "Month-to-month",
            "payment_method": "Electronic check",
            "internet_service": "Fiber optic",
            "tech_support": "No",
        }])
        prediction = model.predict(sample)[0]
        self.assertIn(int(prediction), [0, 1])

    def test_low_risk_customer(self):
        model = joblib.load("student_result_model.pkl")
        sample = pd.DataFrame([{
            "age": 50,
            "tenure_months": 60,
            "monthly_charges": 40.0,
            "total_charges": 2400.0,
            "num_support_calls": 0,
            "satisfaction_score": 9,
            "gender": "Female",
            "contract_type": "Two year",
            "payment_method": "Bank transfer",
            "internet_service": "DSL",
            "tech_support": "Yes",
        }])
        prediction = model.predict(sample)[0]
        self.assertIn(int(prediction), [0, 1])


if __name__ == "__main__":
    unittest.main()