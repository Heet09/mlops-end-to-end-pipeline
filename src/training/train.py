import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
import mlflow
import mlflow.sklearn

def load_data():
    data = {
        "tenure": [1, 12, 24, 6, 36, 48],
        "monthly_charges": [70, 80, 65, 90, 60, 75],
        "churn": [1, 0, 0, 1, 0, 0],
    }
    return pd.DataFrame(data)

def train():
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run():
        df = load_data()
        X = df[["tenure", "monthly_charges"]]
        y = df["churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(model, "artifacts/churn_model.joblib")

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print(f"Training completed | Accuracy: {acc:.2f}")

if __name__ == "__main__":
    train()
