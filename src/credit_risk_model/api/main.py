import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException

from credit_risk_model.core.settings import settings
from .pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI(title="Credit Risk Prediction API")


# ---- Load Model from MLflow Registry ----
MODEL_NAME = "credit_risk_model"
MODEL_STAGE = "Production"

try:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.sklearn.load_model(model_uri)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        X = np.array(request.features).reshape(1, -1)
        prob = model.predict_proba(X)[0][1]
        return PredictionResponse(risk_probability=float(prob))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
