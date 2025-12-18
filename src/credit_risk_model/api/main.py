import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException

# Import your custom utilities
from credit_risk_model.api.pydantic_models import PredictionRequest, PredictionResponse
from credit_risk_model.utils.project_root import get_project_root
from credit_risk_model.api.utils import get_logger

# 1. Setup
logger = get_logger("credit_risk_api")
root = get_project_root()
app = FastAPI(title="Credit Risk Prediction API")

# 2. MLflow Configuration
tracking_uri = f"file:///{root / 'mlruns'}"
mlflow.set_tracking_uri(tracking_uri)
MODEL_URI = "models:/credit_risk_model/Production"

# 3. Load Model
model = None
try:
    logger.info(f"Attempting to load model from: {MODEL_URI}")
    model = mlflow.pyfunc.load_model(MODEL_URI)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Critical Error: Could not load model. {str(e)}")
    logger.warning(
        "Ensure the model is registered in MLflow with the tag 'Production'.")


@app.get("/health")
def health():
    return {
        "status": "online",
        "model_status": "ready" if model else "not_loaded",
        "tracking_uri": tracking_uri
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model unavailable.")

    try:
        # Transform Pydantic to DataFrame
        input_data = pd.DataFrame([request.dict()])

        # Inference
        prediction = model.predict(input_data)

        # Get Probability
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_data)[:, 1][0]
        else:
            # For some mlflow wrappers, the predict output itself might be the prob
            prob = float(prediction[0])

        return PredictionResponse(
            risk_probability=float(prob),
            risk_score=int((1 - prob) * 850),
            prediction=int(prediction[0]),
            status="Denied" if int(prediction[0]) == 1 else "Approved"
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Inference Error")
