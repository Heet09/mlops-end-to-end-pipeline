from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Churn Prediction API")

MODEL_PATH = "artifacts/churn_model.joblib"

class ChurnRequest(BaseModel):
    tenure: int
    monthly_charges: float

class ChurnResponse(BaseModel):
    churn_probability: float

@app.get("/")
def root():
    return {"message": "Churn Prediction API", "version": "1.0.0", "endpoints": ["/health", "/predict"]}

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = None

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    if model is None:
        return {"churn_probability": -1.0}

    prediction = model.predict_proba(
        [[request.tenure, request.monthly_charges]]
    )[0][1]

    return {"churn_probability": float(prediction)}
