"""Create IMERG-based copies of the V5 temporal datasets without rerunning notebooks.

Place this script in the project root (the folder that contains `data/`), then run:
    python rebuild_v5_imerg_features.py

Original CSVs are read-only. New files and a resumable progress CSV are written to data/.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import ee
import pandas as pd

PROJECT_ID = "integral-server-484809-p2"
WINDOWS = (1, 3, 7, 15, 30)
RAIN_COLS = [f"RAIN_{d}D" for d in WINDOWS]
COUNT_COLS = [f"N_{d}D" for d in WINDOWS]
HALF_HOURLY_IMAGES_PER_DAY = 48
# Native IMERG grid is about 0.1 degree (~11 km).
SCALE_METRES = 11132
BATCH_PAUSE_SECONDS = 0.2

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
if not DATA_DIR.exists() and (SCRIPT_DIR.parent / "data").exists():
    # Also works if placed inside a notebooks/ folder.
    DATA_DIR = SCRIPT_DIR.parent / "data"

POS_PATH = DATA_DIR / "multi_horizon_positive_features_v2.csv"
NEG_PATH = DATA_DIR / "v5_negative_features.csv"
NEG_DEF_PATH = DATA_DIR / "v5_negative_definitions.csv"
PROGRESS_PATH = DATA_DIR / "v5_imerg_rainfall_progress.csv"
POS_OUT = DATA_DIR / "multi_horizon_positive_features_v2_imerg.csv"
NEG_OUT = DATA_DIR / "v5_negative_features_imerg.csv"
FINAL_OUT = DATA_DIR / "himalayan_temporal_final_v5_imerg.csv"


def initialize_earth_engine() -> None:
    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception:
        print("Earth Engine initialization failed; opening authentication flow...")
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)


def ensure_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def make_rainfall_image(reference_date: str, imerg: ee.ImageCollection) -> ee.Image:
    """Build a five-window rainfall image plus per-window observation counts."""
    end = ee.Date(reference_date)
    bands = []

    for days in WINDOWS:
        collection = imerg.filterDate(
            end.advance(-days, "day"), end
        ).select("precipitation")

        # IMERG precipitation is a rate in mm/hour; each image is a 30-min interval.
        rainfall = (
            collection
            .map(lambda image: image.multiply(0.5))
            .sum()
            .rename(f"RAIN_{days}D")
        )
        count_band = (
            ee.Image.constant(collection.size())
            .rename(f"N_{days}D")
            .setDefaultProjection(rainfall.projection())
        )
        bands.extend([rainfall, count_band])

    return ee.Image.cat(bands)


def load_progress() -> pd.DataFrame:
    if not PROGRESS_PATH.exists():
        return pd.DataFrame(columns=["sample_id", *RAIN_COLS, *COUNT_COLS])
    progress = pd.read_csv(PROGRESS_PATH)
    ensure_columns(progress, ["sample_id", *RAIN_COLS, *COUNT_COLS], "Progress file")
    return progress.drop_duplicates("sample_id", keep="last")


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Could not find the project's data folder at {DATA_DIR}. "
            "Put this script in the project root beside the data folder."
        )

    for path in (POS_PATH, NEG_PATH, NEG_DEF_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    positive = pd.read_csv(POS_PATH)
    negative_features = pd.read_csv(NEG_PATH)
    negative_definitions = pd.read_csv(NEG_DEF_PATH)

    required = ["latitude", "longitude", "reference_date"]
    ensure_columns(positive, required + RAIN_COLS, "Positive features")
    ensure_columns(negative_features, required + RAIN_COLS, "Negative features")
    ensure_columns(
        negative_definitions,
        required + ["positive_horizon", "LANDSLIDE"],
        "Negative definitions",
    )

    # Build a compact lookup for rainfall extraction; source tables remain unchanged.
    sample_frames = []
    for source, frame in (("positive", positive), ("negative", negative_features)):
        dates = pd.to_datetime(frame["reference_date"], errors="raise").dt.strftime("%Y-%m-%d")
        sample_frames.append(pd.DataFrame({
            "source": source,
            "row_index": range(len(frame)),
            "latitude": pd.to_numeric(frame["latitude"], errors="raise"),
            "longitude": pd.to_numeric(frame["longitude"], errors="raise"),
            "reference_date": dates,
        }))

    samples = pd.concat(sample_frames, ignore_index=True)
    samples.insert(0, "sample_id", range(len(samples)))

    print(f"Positive rows: {len(positive):,}")
    print(f"Negative feature rows: {len(negative_features):,}")
    print(f"Total rainfall samples: {len(samples):,}")
    print(f"Unique reference dates: {samples['reference_date'].nunique():,}")
    print("Original CSV files will not be overwritten.")

    initialize_earth_engine()
    imerg = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").select("precipitation")

    progress = load_progress()
    completed_ids = set(progress["sample_id"].astype(int).tolist())
    print(f"Already completed samples from progress file: {len(completed_ids):,}")

    # One server-side rainfall image per reference date, sampled at all points for that date.
    # This is substantially fewer calls than one Earth Engine request per row.
    remaining = samples[~samples["sample_id"].isin(completed_ids)]
    by_date = list(remaining.groupby("reference_date", sort=True))
    print(f"Dates needing extraction: {len(by_date):,}")

    for date_index, (reference_date, date_rows) in enumerate(by_date, start=1):
        point_features = []
        for row in date_rows.itertuples(index=False):
            point = ee.Geometry.Point([float(row.longitude), float(row.latitude)])
            point_features.append(ee.Feature(point, {"sample_id": int(row.sample_id)}))

        points_fc = ee.FeatureCollection(point_features)
        rain_image = make_rainfall_image(reference_date, imerg)
        sampled_fc = rain_image.sampleRegions(
            collection=points_fc,
            properties=["sample_id"],
            scale=SCALE_METRES,
            geometries=False,
            tileScale=4,
        )
        response = sampled_fc.getInfo()
        returned = []
        for feature in response.get("features", []):
            props = feature.get("properties", {})
            if "sample_id" not in props:
                continue
            row = {"sample_id": int(props["sample_id"])}
            for days in WINDOWS:
                rain_name = f"RAIN_{days}D"
                count_name = f"N_{days}D"
                if props.get(rain_name) is None or props.get(count_name) is None:
                    raise ValueError(
                        f"Missing IMERG data for date {reference_date}, "
                        f"sample_id={row['sample_id']}, band={rain_name}/{count_name}. "
                        "No new output CSVs have been finalized."
                    )
                count = int(round(float(props[count_name])))
                expected = days * HALF_HOURLY_IMAGES_PER_DAY
                if count != expected:
                    raise ValueError(
                        f"Incomplete IMERG window for {reference_date}: "
                        f"{count_name}={count}, expected {expected}. "
                        "Do not train with incomplete rainfall windows."
                    )
                row[rain_name] = float(props[rain_name])
                row[count_name] = count
            returned.append(row)

        expected_ids = set(date_rows["sample_id"].astype(int))
        returned_ids = {row["sample_id"] for row in returned}
        missing_ids = expected_ids - returned_ids
        if missing_ids:
            raise RuntimeError(
                f"Earth Engine returned {len(returned_ids)}/{len(expected_ids)} rows "
                f"for {reference_date}; missing sample IDs: {sorted(missing_ids)[:10]}"
            )

        progress = pd.concat([progress, pd.DataFrame(returned)], ignore_index=True)
        progress = progress.drop_duplicates("sample_id", keep="last")
        progress.to_csv(PROGRESS_PATH, index=False)
        print(
            f"[{date_index}/{len(by_date)}] {reference_date}: "
            f"{len(returned)} locations; saved {len(progress)}/{len(samples)} rainfall rows"
        )
        time.sleep(BATCH_PAUSE_SECONDS)

    # Verify every sample now has complete rainfall values.
    progress = progress.drop_duplicates("sample_id", keep="last")
    if set(progress["sample_id"].astype(int)) != set(samples["sample_id"].astype(int)):
        raise RuntimeError("Progress file is missing sample IDs; source CSVs were not changed.")
    if progress[RAIN_COLS + COUNT_COLS].isna().any().any():
        raise RuntimeError("Progress file contains missing rainfall/count values.")

    lookup = progress.set_index("sample_id")
    positive_imerg = positive.copy()
    negative_imerg = negative_features.copy()

    for source, target_df in (("positive", positive_imerg), ("negative", negative_imerg)):
        map_rows = samples[samples["source"] == source].sort_values("row_index")
        ids = map_rows["sample_id"].astype(int).to_numpy()
        row_indices = map_rows["row_index"].astype(int).to_numpy()
        target_df.loc[row_indices, RAIN_COLS] = lookup.loc[ids, RAIN_COLS].to_numpy()

    # Save separate feature copies; never overwrite the CHIRPS versions.
    positive_imerg.to_csv(POS_OUT, index=False)
    negative_imerg.to_csv(NEG_OUT, index=False)

    # Rebuild V5 with the same merge and target logic as notebook cell 208.
    pos_final = positive_imerg.copy()
    neg_features_final = negative_imerg.copy()
    neg_defs_final = negative_definitions.copy()
    for frame in (pos_final, neg_features_final, neg_defs_final):
        frame["reference_date"] = pd.to_datetime(frame["reference_date"]).dt.strftime("%Y-%m-%d")

    keys = ["latitude", "longitude", "reference_date"]
    negative_final = neg_features_final.merge(
        neg_defs_final[keys + ["positive_horizon", "LANDSLIDE"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    for target in ("NEXT_1D", "NEXT_2D", "NEXT_3D"):
        negative_final[target] = 0

    common_cols = [
        "latitude", "longitude", "reference_date",
        *RAIN_COLS, "ELEVATION", "SLOPE", "LANDCOVER_CODE", "LANDSLIDE",
        "NEXT_1D", "NEXT_2D", "NEXT_3D",
    ]
    ensure_columns(pos_final, common_cols, "Positive IMERG features")
    ensure_columns(negative_final, common_cols, "Negative IMERG features")

    final = pd.concat(
        [pos_final[common_cols], negative_final[common_cols]],
        ignore_index=True,
    )
    final.to_csv(FINAL_OUT, index=False)

    print("\nDONE — new IMERG feature datasets created.")
    print("Positive copy:", POS_OUT)
    print("Negative copy:", NEG_OUT)
    print("New training dataset:", FINAL_OUT)
    print("Shape:", final.shape)
    print("Missing values:", int(final.isna().sum().sum()))
    print("LANDSLIDE counts:", final["LANDSLIDE"].value_counts(dropna=False).to_dict())
    for target in ("NEXT_1D", "NEXT_2D", "NEXT_3D"):
        print(f"{target} counts:", final[target].value_counts(dropna=False).to_dict())
    print("\nExisting CHIRPS datasets and model files were left untouched.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved; rerun the script to resume.")
        sys.exit(130)
