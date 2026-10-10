"""API REST para servir el modelo de churn.

Uso:  uvicorn src.api:app --port 8000   (docs interactivas en /docs)
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", ROOT / "models" / "metadata.json"))
THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.5"))

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """Un cliente. Pydantic valida tipos y valores permitidos antes de predecir."""
    gender: Literal["Female", "Male"]
    SeniorCitizen: YesNo
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Meses como cliente")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {"json_schema_extra": {"examples": [{
        "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 70.7, "TotalCharges": 151.65,
    }]}}


class Prediction(BaseModel):
    churn_probability: float
    churn_prediction: bool
    threshold: float
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # El modelo se carga una sola vez al arrancar, no en cada petición.
    if not MODEL_PATH.exists():
        raise RuntimeError(f"No existe {MODEL_PATH}. Ejecuta primero `make train`.")
    app.state.model = joblib.load(MODEL_PATH)
    meta = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    app.state.version = str(meta.get("registry_version", meta.get("model", "local")))
    yield


app = FastAPI(title="Telco Churn API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": app.state.version}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])
    df["SeniorCitizen"] = df["SeniorCitizen"].map({"No": 0, "Yes": 1})
    try:
        proba = float(app.state.model.predict_proba(df)[0, 1])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {exc}")
    return Prediction(
        churn_probability=round(proba, 4),
        churn_prediction=proba >= THRESHOLD,
        threshold=THRESHOLD,
        model_version=app.state.version,
    )
