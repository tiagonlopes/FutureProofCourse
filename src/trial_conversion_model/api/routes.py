import json
import logging
import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from fastapi import APIRouter

from trial_conversion_model.api.schemas import PredictionRequest, PredictionResponse
from trial_conversion_model.predict import load_model, predict_proba

logger = logging.getLogger(__name__)

router = APIRouter()

PREDICTION_LOG = Path("logs/predictions.jsonl")

# Load the model once, when the service starts, not on every request.
model = load_model()


def to_band(probability: float) -> str:
    """Turn a raw probability into a label a human can act on."""
    if probability < 0.33:
        return "low"
    if probability < 0.66:
        return "medium"
    return "high"


def log_prediction(request: PredictionRequest, probability: float, band: str) -> None:
    """Append one JSON object per prediction, for later analysis."""
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        **request.model_dump(),
        "probability": probability,
        "band": band,
    }
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTION_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": version("trial-conversion-model"),
        "commit": os.getenv("GIT_SHA", "unknown"),
    }


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict conversion probability for a single live trial."""
    aggregates = pd.DataFrame([request.model_dump()])
    probability = round(float(predict_proba(model, aggregates).iloc[0]), 4)
    band = to_band(probability)
    logger.info(
        "prediction | %s -> probability=%.4f band=%s",
        request.model_dump(),
        probability,
        band,
    )
    log_prediction(request, probability, band)
    return PredictionResponse(probability=probability, band=band)
