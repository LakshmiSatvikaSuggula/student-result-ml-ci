import unittest
from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_high_risk_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "age": 30,
                "tenure_months": 2,
                "monthly_charges": 95.0,
                "total_charges": 200.0,
                "num_support_calls": 8,
                "satisfaction_score": 2,
                "gender": "Male",
                "contract_type": "Month-to-month",
                "payment_method": "Electronic check",
                "internet_service": "Fiber optic",
                "tech_support": "No",
            }
        )
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertIn(result["prediction"], ["CHURN", "NO CHURN"])
        self.assertIn(result["prediction_code"], [0, 1])

    def test_low_risk_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "age": 55,
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
            }
        )
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertIn(result["prediction"], ["CHURN", "NO CHURN"])

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={"age": 30, "gender": "Male"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())


if __name__ == "__main__":
    unittest.main()