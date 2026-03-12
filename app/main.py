import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "salary_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

app = FastAPI(title="Experience-Salary Predictor", version="1.0.0")


class PredictionRequest(BaseModel):
    experience_months: float = Field(
        ...,
        ge=0,
        description="Experience in months (must be >= 0).",
    )


class PredictionResponse(BaseModel):
    predicted_salary_thousands: float
    predicted_salary: float
    unit: str



def load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError("Model artifacts are missing. Run scripts/train.py first.")

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


try:
    model, metadata = load_artifacts()
except Exception:
    model, metadata = None, None


@app.get("/")
def root() -> dict:
    return {
        "message": "Salary prediction API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    ready = model is not None and metadata is not None
    return {
        "status": "ok" if ready else "not_ready",
        "model_loaded": ready,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Run training first and restart API.",
        )

    features = pd.DataFrame(
        [{"experience_months": payload.experience_months}],
        columns=["experience_months"],
    )
    pred_thousands = float(model.predict(features)[0])

    return PredictionResponse(
        predicted_salary_thousands=pred_thousands,
        predicted_salary=pred_thousands * 1000,
        unit="predicted_salary is in full currency units, predicted_salary_thousands is in thousands",
    )
