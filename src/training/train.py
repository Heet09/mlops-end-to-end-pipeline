import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

def load_data():
    # Simple dummy dataset for CI & demo
    data = {
        "tenure": [1, 12, 24, 6, 36, 48],
        "monthly_charges": [70, 80, 65, 90, 60, 75],
        "churn": [1, 0, 0, 1, 0, 0],
    }
    return pd.DataFrame(data)

def train():
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

    print(f"Training completed | Accuracy: {acc:.2f}")

if __name__ == "__main__":
    train()
