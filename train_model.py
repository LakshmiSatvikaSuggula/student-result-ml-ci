import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_dataset():
    data = pd.read_csv("ml_dataset_700.csv")
    print("Dataset loaded:", len(data), "records")

    # Drop ID column — not a feature
    data = data.drop(columns=["customer_id"])

    # Drop rows with missing values (safety)
    data = data.dropna()
    print("Records after cleaning:", len(data))
    return data


def train_model():
    data = load_dataset()
    data.to_csv("student_results.csv", index=False)   # keep this filename for compatibility
    print("Dataset snapshot saved as student_results.csv")

    target = "churn"
    numerical_features = [
        "age",
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "num_support_calls",
        "satisfaction_score",
    ]
    categorical_features = [
        "gender",
        "contract_type",
        "payment_method",
        "internet_service",
        "tech_support",
    ]

    X = data[numerical_features + categorical_features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    print("Training model...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\nModel Evaluation")
    print("Accuracy:", round(accuracy, 4))
    print("\nConfusion Matrix:")
    print(matrix)

    joblib.dump(model, "student_result_model.pkl")
    print("\nModel saved as student_result_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test),
    }
    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)
    print("Metrics saved as metrics.json")

    return accuracy


if __name__ == "__main__":
    train_model()
