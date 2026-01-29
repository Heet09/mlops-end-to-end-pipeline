import os
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load real dataset
X, y = load_iris(return_X_y=True)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))

# Versioning
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
MODEL_DIR = os.path.join("models", MODEL_VERSION)
os.makedirs(MODEL_DIR, exist_ok=True)

# Save model
model_path = os.path.join(MODEL_DIR, "model.pkl")
joblib.dump(model, model_path)

print(f"Training completed | Accuracy: {accuracy:.2f}")
print(f"Model saved at: {model_path}")
print(f"Model version: {MODEL_VERSION}")
print(f"Model directory: {MODEL_DIR}")
print(f"Model training and versioning complete.")