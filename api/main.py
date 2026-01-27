from fastapi import FastAPI, HTTPException
import joblib
import os

app = FastAPI(title="ML Inference API with Model Versioning")

def load_model(version: str):
    model_path = os.path.join("models", version, "model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model version '{version}' not found")

    return joblib.load(model_path)

@app.get("/predict")
def predict(feature: float, version: str = "latest"):
    try:
        model = load_model(version)
        prediction = model.predict([[feature]])
        return {
            "model_version": version,
            "prediction": prediction.tolist()
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
