
from pathlib import Path
from datetime import datetime, timedelta
import re

import numpy as np
import rasterio

DATA_DIR = Path("data/raw/imerg_daily")

# Extract the date from your actual IMERG filenames.
date_pattern = re.compile(r"3IMERG\.(\d{8})-")

files_by_date = {}

for path in DATA_DIR.glob("*.1day.tif"):
    match = date_pattern.search(path.name)

    if match:
        date = datetime.strptime(
            match.group(1), "%Y%m%d"
        ).date()
        files_by_date[date] = path

dates = sorted(files_by_date)

print("Daily files found:", len(dates))

if not dates:
    raise RuntimeError("No matching IMERG daily TIFF files found.")

print("First date:", dates[0])
print("Latest date:", dates[-1])

# Uttarkashi test location
latitude = 30.7268
longitude = 78.4354

# Verify that dates are consecutive.
missing_dates = []
expected = dates[0]

while expected <= dates[-1]:
    if expected not in files_by_date:
        missing_dates.append(expected)
    expected += timedelta(days=1)

if missing_dates:
    raise RuntimeError(
        f"Missing daily files: {missing_dates}"
    )

# Extract rainfall at this location for every daily file.
daily_rainfall = []

for date in dates:
    path = files_by_date[date]

    with rasterio.open(path) as src:
        value = next(
            src.sample(
                [(longitude, latitude)],
                indexes=1,
                masked=True
            )
        )[0]

        if np.ma.is_masked(value):
            raise RuntimeError(
                f"Missing rainfall for {date}"
            )

        raw_value = float(value)

        if not np.isfinite(raw_value) or raw_value < 0:
            raise RuntimeError(
                f"Invalid rainfall value for {date}: {raw_value}"
            )

        rainfall_mm = raw_value / 10.0
        daily_rainfall.append(rainfall_mm)

print("\nRainfall features at Uttarkashi:")

for window in [1, 3, 7, 15, 30]:
    if len(daily_rainfall) < window:
        print(
            f"RAIN_{window}D: unavailable "
            f"(need {window} days; have {len(daily_rainfall)})"
        )
        continue

    total = sum(daily_rainfall[-window:])
    print(f"RAIN_{window}D: {total:.2f} mm")
