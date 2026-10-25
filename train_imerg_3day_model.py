

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "himalayan_temporal_final_v5_imerg.csv"
MODEL_PATH = ROOT / "models" / "landslide_3day_model_imerg.joblib"

RAIN = ["RAIN_1D", "RAIN_3D", "RAIN_7D", "RAIN_15D", "RAIN_30D"]
NUMERIC = RAIN + ["ELEVATION", "SLOPE"]
FEATURES = NUMERIC + ["LANDCOVER_CODE"]
TARGET = "NEXT_3D"
SEED = 42
GAP_KM = 5.0


def build_model():
    preprocessing = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC),
        ("landcover", OneHotEncoder(handle_unknown="ignore"),
         ["LANDCOVER_CODE"]),
    ])

    return make_pipeline(
        preprocessing,
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
    )


def remove_nearby_training_rows(train_idx, eval_idx, coords):
    """Exclude training points within 5 km of evaluation points."""
    nn = NearestNeighbors(n_neighbors=1, metric="haversine")
    nn.fit(coords[eval_idx])
    distances, _ = nn.kneighbors(coords[train_idx])

    return train_idx[distances[:, 0] * 6371.0 >= GAP_KM]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    required = (
        ["latitude", "longitude"] + FEATURES + [TARGET]
    )
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Dataset contains missing required values.")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("NEXT_3D must contain only 0 and 1.")

    if df[TARGET].nunique() != 2:
        raise ValueError("NEXT_3D must contain both classes.")

    X = df[FEATURES]
    y = df[TARGET].astype(int).to_numpy()

    # Approximate 5 km spatial blocks for location-separated splits.
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()

    lat_bin = np.floor(lat * 111.32 / GAP_KM).astype(int)
    lon_bin = np.floor(
        lon * 111.32 * np.cos(np.radians(lat)) / GAP_KM
    ).astype(int)

    _, groups = np.unique(
        np.column_stack([lat_bin, lon_bin]),
        axis=0,
        return_inverse=True,
    )

    # Reserve spatial blocks for an untouched final test.
    outer = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=SEED
    )
    development_idx, test_idx = next(
        outer.split(X, y, groups)
    )

    # Split the remaining blocks into training and validation.
    inner = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=SEED + 1
    )
    dev_groups = groups[development_idx]
    train_local, val_local = next(
        inner.split(
            X.iloc[development_idx],
            y[development_idx],
            dev_groups,
        )
    )

    train_idx = development_idx[train_local]
    val_idx = development_idx[val_local]

    coords = np.radians(df[["latitude", "longitude"]].to_numpy())

    # Select the decision threshold using validation data only.
    train_for_val = remove_nearby_training_rows(
        train_idx, val_idx, coords
    )
    validation_model = build_model()
    validation_model.fit(
        X.iloc[train_for_val], y[train_for_val]
    )

    val_scores = validation_model.predict_proba(
        X.iloc[val_idx]
    )[:, 1]

    thresholds = np.arange(0.10, 0.901, 0.025)
    threshold_scores = [
        f1_score(
            y[val_idx],
            (val_scores >= threshold).astype(int),
            zero_division=0,
        )
        for threshold in thresholds
    ]
    best_f1 = max(threshold_scores)
    threshold = float(
        thresholds[
            next(i for i, score in enumerate(threshold_scores)
                 if score == best_f1)
        ]
    )

    # Test on spatially separated blocks, excluding nearby training points.
    train_val_idx = np.concatenate([train_idx, val_idx])
    train_for_test = remove_nearby_training_rows(
        train_val_idx, test_idx, coords
    )

    if len(train_for_test) == 0:
        raise ValueError("No training rows remain after spatial filtering.")

    test_model = build_model()
    test_model.fit(X.iloc[train_for_test], y[train_for_test])

    test_scores = test_model.predict_proba(X.iloc[test_idx])[:, 1]
    test_predictions = (test_scores >= threshold).astype(int)
    y_test = y[test_idx]

    metrics = {
        "test_rows": int(len(test_idx)),
        "test_spatial_blocks": int(len(np.unique(groups[test_idx]))),
        "threshold_selected_on_validation": threshold,
        "roc_auc": (
            float(roc_auc_score(y_test, test_scores))
            if len(np.unique(y_test)) == 2 else None
        ),
        "pr_auc": float(
            average_precision_score(y_test, test_scores)
        ),
        "precision": float(
            precision_score(y_test, test_predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, test_predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, test_predictions, zero_division=0)
        ),
    }

    print("Target:", TARGET)
    print("Training class counts:",
          pd.Series(y[train_for_test]).value_counts().to_dict())
    print("Test class counts:", pd.Series(y_test).value_counts().to_dict())
    print("Held-out test metrics:", metrics)

    # Refit a deployment model using all available rows.
    final_model = build_model()
    final_model.fit(X, y)

    artifact = {
        "model": final_model,
        "features": FEATURES,
        "threshold": threshold,
        "target": TARGET,
        "prediction_horizon": "experimental_next_3_days",
        "rainfall_source": "IMERG",
        "evaluation": metrics,
        "limitations": [
            "Constructed historical event labels may be incomplete.",
            "Evaluation results do not establish operational reliability.",
            "Scores are not calibrated real-world probabilities.",
            "Not suitable for official landslide warnings.",
        ],
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    print("Saved model:", MODEL_PATH)
    print("The previous model was not overwritten.")


if __name__ == "__main__":
    main()
