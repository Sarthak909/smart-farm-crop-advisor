from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model.joblib"
METADATA_PATH = ROOT / "model_metadata.json"

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model artifact not found at {MODEL_PATH}. Run the training notebook first.")

model = joblib.load(MODEL_PATH)
metadata: dict[str, Any] = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
FEATURES = metadata["features"]

app = FastAPI(
    title="Smart Farm Crop & Resource Advisor",
    description="Recommend a suitable crop from soil, weather, and farm-management conditions.",
    version="1.0.0",
)


class FarmConditions(BaseModel):
    """One farm sample; fields match the SF24 Kaggle dataset."""

    model_config = ConfigDict(extra="forbid")

    N: float = Field(description="Soil nitrogen indicator")
    P: float = Field(description="Soil phosphorus indicator")
    K: float = Field(description="Soil potassium indicator")
    temperature: float
    humidity: float
    ph: float = Field(ge=0, le=14)
    rainfall: float = Field(ge=0)
    soil_moisture: float = Field(ge=0)
    soil_type: int = Field(ge=1, le=3)
    sunlight_exposure: float = Field(ge=0)
    wind_speed: float = Field(ge=0)
    co2_concentration: float = Field(ge=0)
    organic_matter: float = Field(ge=0)
    irrigation_frequency: int = Field(ge=0)
    crop_density: float = Field(ge=0)
    pest_pressure: float = Field(ge=0)
    fertilizer_usage: float = Field(ge=0)
    growth_stage: int = Field(ge=0)
    urban_area_proximity: float = Field(ge=0)
    water_source_type: int = Field(ge=1, le=3)
    frost_risk: float = Field(ge=0)
    water_usage_efficiency: float = Field(ge=0)


class Prediction(BaseModel):
    crop: str
    confidence: float
    alternatives: list[dict[str, float | str]]


@app.get("/")
def root() -> dict[str, str]:
    return {"service": app.title, "docs": "/docs", "health": "/health"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": metadata["model_name"]}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "model_name": metadata["model_name"],
        "dataset": metadata["dataset"],
        "test_accuracy": metadata["test_accuracy"],
        "test_macro_f1": metadata["test_macro_f1"],
        "classes": metadata["classes"],
        "features": FEATURES,
    }


@app.post("/predict", response_model=Prediction)
def predict(conditions: FarmConditions) -> Prediction:
    try:
        row = pd.DataFrame([conditions.model_dump()], columns=FEATURES)
        probabilities = model.predict_proba(row)[0]
        ranking = probabilities.argsort()[::-1][:3]
        classes = model.classes_
        return Prediction(
            crop=str(classes[ranking[0]]),
            confidence=round(float(probabilities[ranking[0]]), 6),
            alternatives=[
                {"crop": str(classes[i]), "confidence": round(float(probabilities[i]), 6)}
                for i in ranking
            ],
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not score farm conditions: {exc}") from exc
