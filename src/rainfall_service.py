
from pathlib import Path
from datetime import datetime, timedelta
import re

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "imerg_daily"

DATE_PATTERN = re.compile(r"3IMERG\.(\d{8})-")


def get_dynamic_rainfall(latitude, longitude):
    """
    Extract IMERG rainfall totals at a location.

    Requires 30 consecutive daily GeoTIFF files.
    Returns rainfall features in millimetres.
    """
    latitude = float(latitude)
    longitude = float(longitude)

    if not (-90 <= latitude <= 90):
        raise ValueError("Invalid latitude.")

    if not (-180 <= longitude <= 180):
        raise ValueError("Invalid longitude.")

    files_by_date = {}

    for path in DATA_DIR.glob("*.1day.tif"):
        match = DATE_PATTERN.search(path.name)

        if match:
            date = datetime.strptime(
                match.group(1), "%Y%m%d"
            ).date()
            files_by_date[date] = path

    if not files_by_date:
        raise RuntimeError(
            f"No IMERG daily TIFF files found in {DATA_DIR}"
        )

    latest_date = max(files_by_date)

    # Use the 30 consecutive days ending on the latest
    # available IMERG daily file.
    required_dates = [
        latest_date - timedelta(days=offset)
        for offset in range(29, -1, -1)
    ]

    missing = [
        date.isoformat()
        for date in required_dates
        if date not in files_by_date
    ]

    if missing:
        raise RuntimeError(
            "Cannot calculate a complete 30-day rainfall window. "
            f"Missing dates: {', '.join(missing)}"
        )

    daily_values = []

    for date in required_dates:
        path = files_by_date[date]

        with rasterio.open(path) as src:
            value = next(
                src.sample(
                    [(longitude, latitude)],
                    indexes=1,
                    masked=True,
                )
            )[0]

            if np.ma.is_masked(value):
                raise RuntimeError(
                    f"No rainfall value for {date} at this location."
                )

            raw_value = float(value)

            if (
                not np.isfinite(raw_value)
                or raw_value < 0
                or raw_value > 29998
            ):
                raise RuntimeError(
                    f"Missing or invalid IMERG rainfall for {date}: "
                    f"{raw_value}"
                )

            # IMERG daily TIFF values use a scale factor of 10.
            daily_values.append(raw_value / 10.0)

    result = {}

    for days in [1, 3, 7, 15, 30]:
        result[f"RAIN_{days}D"] = round(
            sum(daily_values[-days:]), 2
        )

    result["rainfall_source"] = "IMERG Early Run"
    result["rainfall_data_through"] = latest_date.isoformat()
    result["reference_date"] = (
        latest_date + timedelta(days=1)
    ).isoformat()

    return result
