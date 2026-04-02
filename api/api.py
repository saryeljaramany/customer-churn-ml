import os
import tempfile
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import logging
import traceback as tb

from src.customer_churn_ml.config import PATHS
from src.customer_churn_ml.constants import CONTRACT_ORDER, INTERNET_ORDER
from src.customer_churn_ml.predict import predict_churn

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn probability",
    version="1.0.0",
)

CORS_ALLOW_ORIGINS = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501",
)
allow_origins = [origin.strip() for origin in CORS_ALLOW_ORIGINS.split(",") if origin.strip()]
if not allow_origins:
    allow_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerInput(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionRequest(BaseModel):
    customers: list[CustomerInput]


class PredictionResponse(BaseModel):
    results: list[dict]


@app.get("/meta")
async def get_meta():
    return {
        "service": "customer-churn-prediction",
        "version": "1.0.0",
        "fields": {
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["No", "Yes"],
            "Dependents": ["No", "Yes"],
            "PhoneService": ["No", "Yes"],
            "MultipleLines": ["No", "Yes", "No phone service"],
            "InternetService": list(INTERNET_ORDER),
            "OnlineSecurity": ["No", "Yes", "No internet service"],
            "OnlineBackup": ["No", "Yes", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No internet service"],
            "TechSupport": ["No", "Yes", "No internet service"],
            "StreamingTV": ["No", "Yes", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No internet service"],
            "Contract": list(CONTRACT_ORDER),
            "PaperlessBilling": ["No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        },
    }


@app.post("/predict", response_model=PredictionResponse, status_code=200)
async def predict_customer(request: PredictionRequest):

    if not request.customers:
        raise HTTPException(status_code=400, detail="At least one customer record is required.")

    data = [customer.model_dump() for customer in request.customers]
    input_df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        input_df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        try:
            results_df = predict_churn(
                input_path=tmp_path,
                output_path=None,
                model_dir=PATHS.model_dir,
            )
            results_list = results_df.to_dict(orient="records")

        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail="Prediction service is not ready. Model files could not be found.",
            ) from exc

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        except Exception as exc:
            logger.error("Prediction failed:\n%s", tb.format_exc())
            raise HTTPException(
                status_code=500,
                detail="We could not process this request right now. Please try again.",
            ) from exc

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return PredictionResponse(results=results_list)


@app.get("/")
async def root():
    return {
        "message": "Customer Churn Prediction API is running",
        "version": "1.0.0",
        "endpoints": {"predict": "/predict", "docs": "/docs", "openapi": "/openapi.json"},
    }


@app.get("/health")
async def health_check():
    model_dir = PATHS.model_dir
    model_files = ["churn_model.pkl", "feature_names.pkl", "preprocessor.pkl", "scaler.pkl"]
    missing_files = [name for name in model_files if not (model_dir / name).exists()]

    return {
        "status": "healthy",
        "service": "customer-churn-prediction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_ready": len(missing_files) == 0,
        "missing_model_files": missing_files,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", os.getenv("API_PORT", "8000"))),
    )