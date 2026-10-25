"""Compare CHIRPS vs IMERG features on identical spatial holdout folds.

Run from the project root:
    python -u compare_v5_chirps_imerg.py

This script only reads the existing datasets and writes a new evaluation CSV.
It does not overwrite datasets or save/replace models.
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
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
DATA = ROOT / "data"
CHIRPS_PATH = DATA / "himalayan_temporal_final_v5.csv"
IMERG_PATH = DATA / "himalayan_temporal_final_v5_imerg.csv"
OUTPUT_PATH = DATA / "v5_chirps_vs_imerg_evaluation.csv"

KEYS = ["latitude_key", "longitude_key", "reference_date_key"]
RAIN = ["RAIN_1D", "RAIN_3D", "RAIN_7D", "RAIN_15D", "RAIN_30D"]
NUMERIC = RAIN + ["ELEVATION", "SLOPE"]
TARGETS = ["NEXT_1D", "NEXT_2D", "NEXT_3D"]
SEED = 42
N_SPLITS = 5
TEST_SIZE = 0.25
MIN_SPATIAL_DISTANCE_KM = 5.0


def load_dataset(path: Path, tag: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    frame = pd.read_csv(path)
    required = {
        "latitude", "longitude", "reference_date", "ELEVATION", "SLOPE",
        "LANDCOVER_CODE", "NEXT_1D", "NEXT_2D", "NEXT_3D", *RAIN,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{tag} dataset is missing columns: {missing}")

    frame = frame.copy()
    frame["reference_date"] = pd.to_datetime(
        frame["reference_date"], errors="coerce"
    ).dt.normalize()
    if frame["reference_date"].isna().any():
        raise ValueError(f"{tag}: found invalid reference_date values")
    if frame[["latitude", "longitude"]].isna().any().any():
        raise ValueError(f"{tag}: missing coordinates found")

    # Stable keys tolerate harmless floating-point representation differences.
    frame["latitude_key"] = frame["latitude"].astype(float).round(6)
    frame["longitude_key"] = frame["longitude"].astype(float).round(6)
    frame["reference_date_key"] = frame["reference_date"].dt.strftime("%Y-%m-%d")
    if frame.duplicated(KEYS).any():
        examples = frame.loc[frame.duplicated(KEYS, keep=False), KEYS].head(5)
        raise ValueError(
            f"{tag}: duplicate location-date keys found; refusing ambiguous match.\n"
            f"Examples:\n{examples.to_string(index=False)}"
        )
    for col in TARGETS:
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(int)
        if not set(frame[col].unique()).issubset({0, 1}):
            raise ValueError(f"{tag}: {col} must contain only 0/1 labels")
    return frame


def remove_nearby_training_rows(train_idx, test_idx, coords_radians):
    nn = NearestNeighbors(n_neighbors=1, metric="haversine")
    nn.fit(coords_radians[test_idx])
    distances, _ = nn.kneighbors(coords_radians[train_idx])
    keep = distances[:, 0] * 6371.0 >= MIN_SPATIAL_DISTANCE_KM
    return train_idx[keep]


def build_model():
    # Match notebook's V5 comparison: numeric features + one-hot landcover.
    prep = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC),
            ("landcover", OneHotEncoder(handle_unknown="ignore"), ["LANDCOVER_CODE"]),
        ]
    )
    return make_pipeline(
        prep,
        RandomForestClassifier(
            n_estimators=200,
            random_state=SEED,
            class_weight="balanced",
            n_jobs=-1,
        ),
    )


def main():
    print("Loading CHIRPS and IMERG datasets...", flush=True)
    chirps = load_dataset(CHIRPS_PATH, "CHIRPS")
    imerg = load_dataset(IMERG_PATH, "IMERG")
    print(f"CHIRPS shape: {chirps.shape}; IMERG shape: {imerg.shape}", flush=True)

    cols_to_compare = KEYS + TARGETS + ["LANDSLIDE"]
    if "LANDSLIDE" not in chirps or "LANDSLIDE" not in imerg:
        raise ValueError("Both datasets must contain LANDSLIDE for row-integrity checks")

    # Check one-to-one sample alignment and unchanged labels, then evaluate a common set.
    joined = chirps[cols_to_compare].merge(
        imerg[cols_to_compare], on=KEYS, how="outer", suffixes=("_chirps", "_imerg"),
        indicator=True, validate="one_to_one"
    )
    unmatched = joined["_merge"] != "both"
    if unmatched.any():
        raise ValueError(
            f"Datasets do not contain exactly the same location-date keys: "
            f"{int(unmatched.sum())} unmatched keys. Stop before evaluation."
        )
    for col in TARGETS + ["LANDSLIDE"]:
        if not joined[f"{col}_chirps"].equals(joined[f"{col}_imerg"]):
            equal = joined[f"{col}_chirps"].to_numpy() == joined[f"{col}_imerg"].to_numpy()
            raise ValueError(f"Labels differ between datasets for {col}: {int((~equal).sum())} rows")

    # Reindex both sources into identical key order.
    imerg_idx = imerg.set_index(KEYS, drop=False)
    chirps_idx = chirps.set_index(KEYS, drop=False)
    common_keys = chirps[KEYS].copy()
    chirps = chirps_idx.loc[pd.MultiIndex.from_frame(common_keys)].reset_index(drop=True)
    imerg = imerg_idx.loc[pd.MultiIndex.from_frame(common_keys)].reset_index(drop=True)

    print("Sample keys and labels match exactly.", flush=True)
    print("Date range:", chirps["reference_date"].min().date(), "to", chirps["reference_date"].max().date(), flush=True)
    print("Targets:", {t: chirps[t].value_counts().sort_index().to_dict() for t in TARGETS}, flush=True)

    # Spatial blocks match the notebook's approximate 5 km grouping.
    lat_km = chirps["latitude"].to_numpy() * 111.32
    lon_km = (
        chirps["longitude"].to_numpy()
        * 111.32
        * np.cos(np.radians(chirps["latitude"].to_numpy()))
    )
    # Build integer spatial-block IDs without NumPy string-array addition,
    # which raises a ufunc error on some NumPy versions.
    lat_bins = np.floor(lat_km / MIN_SPATIAL_DISTANCE_KM).astype(np.int64)
    lon_bins = np.floor(lon_km / MIN_SPATIAL_DISTANCE_KM).astype(np.int64)
    _, groups = np.unique(
        np.column_stack((lat_bins, lon_bins)), axis=0, return_inverse=True
    )
    coords_radians = np.radians(chirps[["latitude", "longitude"]].to_numpy())

    results = []
    for target in TARGETS:
        y = chirps[target].to_numpy()
        splitter = GroupShuffleSplit(
            n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SEED
        )
        folds = list(splitter.split(chirps, y, groups))
        print(f"\nEvaluating {target} across {N_SPLITS} matched spatial folds...", flush=True)

        for fold_no, (train_idx, test_idx) in enumerate(folds, start=1):
            filtered_train_idx = remove_nearby_training_rows(
                train_idx, test_idx, coords_radians
            )
            if len(filtered_train_idx) == 0:
                raise ValueError(f"No training rows remain for {target}, fold {fold_no}")
            if np.unique(y[filtered_train_idx]).size < 2 or np.unique(y[test_idx]).size < 2:
                warnings.warn(f"Fold {fold_no} of {target} has only one class; AUC will be unavailable")

            for source_name, frame in [("CHIRPS", chirps), ("IMERG", imerg)]:
                model = build_model()
                model.fit(frame.iloc[filtered_train_idx][NUMERIC + ["LANDCOVER_CODE"]], y[filtered_train_idx])
                test_frame = frame.iloc[test_idx]
                test_y = test_frame[target].to_numpy()
                probabilities = model.predict_proba(test_frame[NUMERIC + ["LANDCOVER_CODE"]])[:, 1]
                predictions = (probabilities >= 0.5).astype(int)
                results.append({
                    "target": target,
                    "source": source_name,
                    "fold": fold_no,
                    "train_rows_after_5km_filter": len(filtered_train_idx),
                    "test_rows": len(test_idx),
                    "test_positive_rate": float(test_y.mean()),
                    "accuracy": accuracy_score(test_y, predictions),
                    "precision": precision_score(test_y, predictions, zero_division=0),
                    "recall": recall_score(test_y, predictions, zero_division=0),
                    "f1": f1_score(test_y, predictions, zero_division=0),
                    "roc_auc": roc_auc_score(test_y, probabilities) if np.unique(test_y).size == 2 else np.nan,
                    "pr_auc": average_precision_score(test_y, probabilities) if np.unique(test_y).size == 2 else np.nan,
                })
                print(f"  fold {fold_no}/{N_SPLITS} — {source_name} complete", flush=True)

    results_df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    summary = results_df.groupby(["target", "source"])[metric_cols].agg(["mean", "std"]).round(4)
    print("\nMEAN ± STANDARD DEVIATION ACROSS SPATIAL FOLDS", flush=True)
    print(summary.to_string(), flush=True)
    print(f"\nDetailed results saved to: {OUTPUT_PATH}", flush=True)
    print("No input datasets were modified; no model was saved or replaced.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr, flush=True)
        raise
