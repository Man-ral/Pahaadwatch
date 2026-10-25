
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.feature_service import get_terrain_features
from src.rainfall_service import get_dynamic_rainfall
from fastapi.middleware.cors import CORSMiddleware
from src.feature_service import get_location_features
from src.predict import predict_susceptibility as predict_susceptibility_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "landslide_3day_model_imerg.joblib"

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
features = artifact["features"]
threshold = float(artifact["threshold"])

app = FastAPI(
    title="Himalayan Landslide Risk API",
    description="Experimental three-day landslide-risk prototype",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

class LocationInput(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH.name,
        "horizon": "next_3_days",
    }


@app.post("/predict")
def predict(data: LocationInput):
    try:
        # 1. Get the rainfall features.
        rainfall = get_dynamic_rainfall(
            data.latitude, data.longitude
        )

        # 2. Get static terrain and land-cover features.
        terrain = get_terrain_features(
            data.latitude, data.longitude
        )

        # 3. Combine the eight features in the model's expected order.
        inputs = {**rainfall, **terrain}

        sample = pd.DataFrame(
            [{name: inputs[name] for name in features}],
            columns=features,
        )

        # 4. Run the saved IMERG model.
        probabilities = model.predict_proba(sample)[0]
        positive_index = list(model.classes_).index(1)
        score = float(probabilities[positive_index])

        return {
            "location": {
                "latitude": data.latitude,
                "longitude": data.longitude,
            },
            "prediction_horizon": "next_3_days",
            "experimental_score": round(score, 4),
            "threshold": threshold,
            "above_threshold": score >= threshold,
            "rainfall": rainfall,
            "terrain": terrain,
            "notice": (
                "Experimental score, not a calibrated probability "
                "or an official landslide warning."
            ),
        }

    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
@app.post("/predict/susceptibility")
def predict_susceptibility_endpoint(data: LocationInput):
    try:
        extracted = get_location_features(
            data.latitude,
            data.longitude,
        )

        result = predict_susceptibility_model(
            **extracted["susceptibility_inputs"]
        )

        return {
            "location": {
                "latitude": data.latitude,
                "longitude": data.longitude,
            },
            **result,
            "metadata": extracted["metadata"],
        }

    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc