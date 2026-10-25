
from datetime import date, timedelta
from pathlib import Path
import csv
import getpass
import math
import shutil
import sys

import requests
import rasterio

# Use the latest daily raster we have verified.
END_DATE = date(2026, 10, 16)
DAYS = 30
LAT, LON = 30.7, 78.4

RAW_DIR = Path("data/raw")
DOWNLOAD_DIR = RAW_DIR / "imerg_daily"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/early/"
SLOT = "S173000-E175959.1050.V07C.1day.tif"

username = input("PPS username: ")
password = getpass.getpass("PPS password: ")
session = requests.Session()
session.auth = (username, password)

daily = []

# Newest date first, so the first N values form each rolling window.
for i in range(DAYS):
    day = END_DATE - timedelta(days=i)
    filename = f"3B-HHR-E.MS.MRG.3IMERG.{day:%Y%m%d}-{SLOT}"
    target = DOWNLOAD_DIR / filename
    existing = RAW_DIR / filename

    if target.exists():
        raster_path = target
    elif existing.exists():
        raster_path = existing
    else:
        url = BASE_URL + filename
        print(f"Downloading {day}...")
        try:
            with session.get(url, stream=True, timeout=90) as response:
                if response.status_code != 200:
                    print(f"Failed for {day}: HTTP {response.status_code}")
                    sys.exit(1)

                with target.open("wb") as f:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
            raster_path = target
        except Exception:
            target.unlink(missing_ok=True)
            raise

    # Read the pixel at the test coordinates.
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Missing CRS: {raster_path}")
        if not (-180 <= LON <= 180 and -90 <= LAT <= 90):
            raise ValueError("Invalid test coordinates")

        raw = float(next(src.sample([(LON, LAT)]))[0])

    if not math.isfinite(raw) or raw < 0:
        raise ValueError(f"Invalid raster pixel for {day}: {raw}")

    # IMERG GIS precipitation is represented in tenths of a millimetre.
    rainfall_mm = raw * 0.1
    daily.append({
        "date": day.isoformat(),
        "raw_value": raw,
        "rainfall_mm": rainfall_mm,
    })

    print(f"{day}: {rainfall_mm:.2f} mm")

session.close()

# Save the daily samples for inspection.
csv_path = RAW_DIR / "imerg_rainfall_sample_30d.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=daily[0].keys())
    writer.writeheader()
    writer.writerows(daily)

print("\nRolling rainfall totals at the test location:")
for window in (1, 3, 7, 15, 30):
    total = sum(item["rainfall_mm"] for item in daily[:window])
    print(f"RAIN_{window}D = {total:.2f} mm")

print(f"\nDaily samples saved to: {csv_path}")
