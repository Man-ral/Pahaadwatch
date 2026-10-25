"""Chronological holdout comparison for CHIRPS vs IMERG V5 datasets.

Uses the temporal cutoff already used in 04_model_training.ipynb:
  training: reference_date before 2016-01-01
  testing:  reference_date on/after 2016-01-01

This is an evaluation-only script. It does not modify input datasets or save models.
Run from the project root or place this file in the project root:
    python -u evaluate_v5_chronological_chirps_imerg.py
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
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CHIRPS_PATH = DATA / "himalayan_temporal_final_v5.csv"
IMERG_PATH = DATA / "himalayan_temporal_final_v5_imerg.csv"
OUTPUT_PATH = DATA / "v5_chronological_chirps_vs_imerg_evaluation.csv"
PREDICTIONS_PATH = DATA / "v5_chronological_chirps_vs_imerg_predictions.csv"

CUTOFF = pd.Timestamp("2016-01-01")
RAIN = ["RAIN_1D", "RAIN_3D", "RAIN_7D", "RAIN_15D", "RAIN_30D"]
NUMERIC = RAIN + ["ELEVATION", "SLOPE"]
CATEGORICAL = ["LANDCOVER_CODE"]
FEATURES = NUMERIC + CATEGORICAL
TARGETS = ["NEXT_1D", "NEXT_2D", "NEXT_3D"]
KEY_COLS = ["latitude_key", "longitude_key", "reference_date_key"]
SEED = 42
THRESHOLD = 0.5


def load_dataset(path: Path, tag: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{tag} dataset not found: {path}")
    df = pd.read_csv(path)
    required = {
        "latitude", "longitude", "reference_date", "LANDSLIDE",
        "ELEVATION", "SLOPE", "LANDCOVER_CODE", *RAIN, *TARGETS,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{tag} dataset missing required columns: {missing}")

    df = df.copy()
    df["reference_date"] = pd.to_datetime(df["reference_date"], errors="coerce").dt.normalize()
    if df["reference_date"].isna().any():
        raise ValueError(f"{tag}: invalid reference_date values found")
    if df[["latitude", "longitude"]].isna().any().any():
        raise ValueError(f"{tag}: missing coordinates found")

    df["latitude_key"] = pd.to_numeric(df["latitude"], errors="raise").astype(float).round(6)
    df["longitude_key"] = pd.to_numeric(df["longitude"], errors="raise").astype(float).round(6)
    df["reference_date_key"] = df["reference_date"].dt.strftime("%Y-%m-%d")
    if df.duplicated(KEY_COLS).any():
        example = df.loc[df.duplicated(KEY_COLS, keep=False), KEY_COLS].head(5)
        raise ValueError(f"{tag}: duplicate location-date keys found:\n{example.to_string(index=False)}")

    for col in FEATURES + TARGETS + ["LANDSLIDE"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if df[col].isna().any():
            raise ValueError(f"{tag}: missing values in {col}")
    for col in TARGETS + ["LANDSLIDE"]:
        vals = set(df[col].astype(int).unique())
        if not vals.issubset({0, 1}):
            raise ValueError(f"{tag}: {col} must contain only 0/1; found {sorted(vals)}")
        df[col] = df[col].astype(int)
    return df


def validate_alignment(chirps: pd.DataFrame, imerg: pd.DataFrame):
    for tag, df in [("CHIRPS", chirps), ("IMERG", imerg)]:
        if "LANDSLIDE" not in df:
            raise ValueError(f"{tag}: LANDSLIDE column is missing")

    cols = KEY_COLS + TARGETS + ["LANDSLIDE"]
    joined = chirps[cols].merge(
        imerg[cols], on=KEY_COLS, how="outer", suffixes=("_chirps", "_imerg"),
        indicator=True, validate="one_to_one"
    )
    if (joined["_merge"] != "both").any():
        raise ValueError(
            "CHIRPS and IMERG location-date keys do not match exactly; "
            f"unmatched rows: {int((joined['_merge'] != 'both').sum())}"
        )
    for col in TARGETS + ["LANDSLIDE"]:
        a = joined[f"{col}_chirps"].to_numpy()
        b = joined[f"{col}_imerg"].to_numpy()
        if not np.array_equal(a, b):
            raise ValueError(f"Labels differ between CHIRPS and IMERG for {col}")

    common_order = pd.MultiIndex.from_frame(chirps[KEY_COLS])
    imerg_indexed = imerg.set_index(KEY_COLS, drop=False)
    imerg_aligned = imerg_indexed.loc[common_order].reset_index(drop=True)
    chirps = chirps.reset_index(drop=True)
    return chirps, imerg_aligned


def build_model():
    # Same model protocol as the previous CHIRPS-vs-IMERG spatial comparison.
    prep = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC),
            ("landcover", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
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


def safe_auc(y_true, probabilities):
    return roc_auc_score(y_true, probabilities) if np.unique(y_true).size == 2 else np.nan


def main():
    print("Loading CHIRPS and IMERG datasets...", flush=True)
    chirps = load_dataset(CHIRPS_PATH, "CHIRPS")
    imerg = load_dataset(IMERG_PATH, "IMERG")
    chirps, imerg = validate_alignment(chirps, imerg)

    print(f"CHIRPS shape: {chirps.shape}; IMERG shape: {imerg.shape}", flush=True)
    print("Location-date keys and all labels match exactly.", flush=True)
    print(
        f"Full date range: {chirps['reference_date'].min().date()} to "
        f"{chirps['reference_date'].max().date()}", flush=True
    )
    print(f"Chronological cutoff: {CUTOFF.date()}", flush=True)

    train_mask = chirps["reference_date"] < CUTOFF
    test_mask = chirps["reference_date"] >= CUTOFF
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Chronological cutoff created an empty training or test set")

    train_idx = np.flatnonzero(train_mask.to_numpy())
    test_idx = np.flatnonzero(test_mask.to_numpy())
    print(
        f"Training: {len(train_idx)} rows, {chirps.loc[train_mask, 'reference_date'].min().date()} to "
        f"{chirps.loc[train_mask, 'reference_date'].max().date()}", flush=True
    )
    print(
        f"Testing:  {len(test_idx)} rows, {chirps.loc[test_mask, 'reference_date'].min().date()} to "
        f"{chirps.loc[test_mask, 'reference_date'].max().date()}", flush=True
    )
    print("Target distribution (train / test):", flush=True)
    for target in TARGETS:
        print(
            f"  {target}: train={chirps.loc[train_mask, target].value_counts().sort_index().to_dict()} | "
            f"test={chirps.loc[test_mask, target].value_counts().sort_index().to_dict()}", flush=True
        )

    results = []
    prediction_rows = []
    for target in TARGETS:
        y_all = chirps[target].to_numpy()
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        if np.unique(y_train).size < 2:
            raise ValueError(f"{target}: training period contains only one class")
        if np.unique(y_test).size < 2:
            warnings.warn(f"{target}: test period contains one class; AUC will be unavailable")

        print(f"\nEvaluating {target} on the untouched chronological test period...", flush=True)
        for source, frame in [("CHIRPS", chirps), ("IMERG", imerg)]:
            X_train = frame.iloc[train_idx][FEATURES]
            X_test = frame.iloc[test_idx][FEATURES]
            model = build_model()
            model.fit(X_train, y_train)
            probability = model.predict_proba(X_test)[:, 1]
            pred = (probability >= THRESHOLD).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
            results.append({
                "target": target,
                "source": source,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "train_start": chirps.iloc[train_idx]["reference_date"].min().date().isoformat(),
                "train_end": chirps.iloc[train_idx]["reference_date"].max().date().isoformat(),
                "test_start": chirps.iloc[test_idx]["reference_date"].min().date().isoformat(),
                "test_end": chirps.iloc[test_idx]["reference_date"].max().date().isoformat(),
                "threshold": THRESHOLD,
                "test_positive_rate": float(y_test.mean()),
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "roc_auc": safe_auc(y_test, probability),
                "pr_auc": average_precision_score(y_test, probability),
                "brier_score": brier_score_loss(y_test, probability),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            })
            for pos, row_idx in enumerate(test_idx):
                row = chirps.iloc[row_idx]
                prediction_rows.append({
                    "target": target,
                    "source": source,
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "reference_date": row["reference_date"].date().isoformat(),
                    "y_true": int(y_test[pos]),
                    "probability": float(probability[pos]),
                    "prediction_at_0_5": int(pred[pos]),
                })
            print(f"  {source} complete", flush=True)

    result_df = pd.DataFrame(results)
    prediction_df = pd.DataFrame(prediction_rows)
    DATA.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)
    prediction_df.to_csv(PREDICTIONS_PATH, index=False)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"]
    print("\nCHRONOLOGICAL HOLDOUT RESULTS (cutoff 2016-01-01)", flush=True)
    print(result_df[["target", "source", *metrics]].round(4).to_string(index=False), flush=True)
    print(f"\nMetrics saved to: {OUTPUT_PATH}", flush=True)
    print(f"Holdout predictions saved to: {PREDICTIONS_PATH}", flush=True)
    print("Evaluation only: input datasets and trained models were not modified or replaced.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr, flush=True)
        raise
