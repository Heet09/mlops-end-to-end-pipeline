import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Dummy dataset
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])

# Train model
model = LogisticRegression()
model.fit(X, y)

# Evaluate
accuracy = accuracy_score(y, model.predict(X))

# Model versioning
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
MODEL_DIR = os.path.join("models", MODEL_VERSION)
os.makedirs(MODEL_DIR, exist_ok=True)

# Save model
model_path = os.path.join(MODEL_DIR, "model.pkl")
joblib.dump(model, model_path)

print(f"Training completed | Accuracy: {accuracy:.2f}")
print(f"Model saved at: {model_path}")
print(f"Model version: {MODEL_VERSION}")
print(f"Model directory contents: {os.listdir(MODEL_DIR)}") 