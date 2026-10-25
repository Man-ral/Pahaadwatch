import math
from datetime import date, datetime, timedelta, timezone

import ee


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EE_PROJECT_ID = "integral-server-484809-p2"

CHIRPS_ID = "UCSB-CHG/CHIRPS/DAILY"
DEM_ID = "USGS/SRTMGL1_003"
WORLDCOVER_ID = "ESA/WorldCover/v200"

# Match the approximate study-area bounds in predict.py.
BOUNDS = {
    "min_latitude": 29.135275,
    "max_latitude": 32.99,
    "min_longitude": 75.74,
    "max_longitude": 80.73,
}

CLIMATOLOGY_START = "2015-01-01"
CLIMATOLOGY_END = "2025-01-01"

WINDOWS = (1, 3, 7, 15, 30)

_EE_INITIALIZED = False


# --------------------------------------------------
# Validation and Earth Engine initialization
# --------------------------------------------------

def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")

    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    return value


def _initialize_earth_engine():
    global _EE_INITIALIZED

    if _EE_INITIALIZED:
        return

    try:
        ee.Initialize(project=EE_PROJECT_ID)
    except Exception as exc:
        raise RuntimeError(
            "Google Earth Engine initialization failed. "
            "Authenticate this Python environment and verify the "
            f"Earth Engine project: {EE_PROJECT_ID}."
        ) from exc

    _EE_INITIALIZED = True


def _validate_location(latitude, longitude):
    latitude = _number(latitude, "latitude")
    longitude = _number(longitude, "longitude")

    bounds = BOUNDS

    if not (
        bounds["min_latitude"] <= latitude <= bounds["max_latitude"]
        and bounds["min_longitude"] <= longitude <= bounds["max_longitude"]
    ):
        raise ValueError(
            "Coordinates are outside the project's approximate study area."
        )

    return latitude, longitude


def _parse_date(value):
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                "reference_date must use YYYY-MM-DD format."
            ) from exc

    raise ValueError(
        "reference_date must be a date, datetime, YYYY-MM-DD string, "
        "or None."
    )


def _require_values(values, required_keys, context):
    for key in required_keys:
        value = values.get(key)

        if value is None:
            raise RuntimeError(
                f"No value returned for {key} while extracting {context}."
            )

        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"Invalid value returned for {key} while extracting {context}."
            ) from exc

        if not math.isfinite(number):
            raise RuntimeError(
                f"Non-finite value returned for {key} while extracting {context}."
            )


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

def get_location_features(latitude, longitude, reference_date=None):
    """
    Retrieve features required by both prediction models.

    If reference_date is omitted, use the day after the latest CHIRPS
    image available in Earth Engine.

    Returns:
        metadata: rainfall observation dates and reference date
        susceptibility_inputs: ready for predict_susceptibility(**inputs)
        temporal_inputs: ready for predict_landslide_3day(**inputs)
    """
    latitude, longitude = _validate_location(latitude, longitude)
    requested_date = _parse_date(reference_date)

    _initialize_earth_engine()

    point = ee.Geometry.Point([longitude, latitude])

    chirps = ee.ImageCollection(CHIRPS_ID).select("precipitation")

    # Determine the latest available CHIRPS observation date.
    latest_timestamp = chirps.aggregate_max("system:time_start").getInfo()

    if latest_timestamp is None:
        raise RuntimeError("The CHIRPS collection contains no observations.")

    latest_date_text = (
        ee.Date(latest_timestamp).format("YYYY-MM-dd").getInfo()
    )
    latest_date = date.fromisoformat(latest_date_text)

    if requested_date is None:
        # Find the newest reference date with a complete 30-day window.
        check_start = latest_date - timedelta(days=45)

        timestamps = (
            chirps
            .filterDate(
                check_start.isoformat(),
                (latest_date + timedelta(days=1)).isoformat(),
            )
            .aggregate_array("system:time_start")
            .getInfo()
        )

        available_dates = {
            datetime.fromtimestamp(
                timestamp / 1000, tz=timezone.utc
            ).date()
            for timestamp in timestamps
        }

        ref_date = None

        for offset in range(15):
            candidate = latest_date + timedelta(days=1 - offset)

            required_dates = {
                candidate - timedelta(days=days)
                for days in range(1, 31)
            }

            if required_dates.issubset(available_dates):
                ref_date = candidate
                break

        if ref_date is None:
            raise RuntimeError(
                "Could not find a complete 30-day CHIRPS window "
                "among the most recent 15 candidate dates."
            )

    else:
        ref_date = requested_date

        if ref_date > latest_date + timedelta(days=1):
            raise ValueError(
                "The requested reference date exceeds the latest "
                f"available CHIRPS observation ({latest_date})."
            )
    # Build each rolling rainfall window.
    window_collections = {}
    window_counts = {
        "climatology": chirps.filterDate(
            CLIMATOLOGY_START, CLIMATOLOGY_END
        ).size()
    }

    end = ee.Date(ref_date.isoformat())

    for days in WINDOWS:
        start = end.advance(-days, "day")

        collection = chirps.filterDate(start, end)
        name = f"RAIN_{days}D"

        window_collections[name] = collection
        window_counts[name] = collection.size()

    # Verify that every requested window contains the expected number
    # of daily images. Incomplete rainfall windows must not be used.
    counts = ee.Dictionary(window_counts).getInfo()

    expected_climatology_days = (
        date(2025, 1, 1) - date(2015, 1, 1)
    ).days

    if counts.get("climatology") != expected_climatology_days:
        raise RuntimeError(
            "The 2015–2024 CHIRPS climatology is incomplete. "
            f"Expected {expected_climatology_days} daily images; "
            f"found {counts.get('climatology')}."
        )

    for days in WINDOWS:
        name = f"RAIN_{days}D"

        if counts.get(name) != days:
            raise RuntimeError(
                f"Incomplete {name} window for reference date "
                f"{ref_date}: expected {days} daily images, "
                f"found {counts.get(name)}."
            )

    # Susceptibility rainfall: reproduce the fixed 2015–2024 average.
    mean_rainfall = (
        chirps.filterDate(CLIMATOLOGY_START, CLIMATOLOGY_END)
        .mean()
        .rename("RAINFALL")
    )

    # Temporal rainfall: sums over the previous N days.
    rainfall_images = [mean_rainfall]

    for days in WINDOWS:
        rainfall_images.append(
            window_collections[f"RAIN_{days}D"]
            .sum()
            .rename(f"RAIN_{days}D")
        )

    rainfall_image = ee.Image.cat(rainfall_images)

    rainfall_values = rainfall_image.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=5000,
        maxPixels=1_000_000,
    ).getInfo()

    _require_values(
        rainfall_values,
        ["RAINFALL"] + [f"RAIN_{days}D" for days in WINDOWS],
        "rainfall",
    )

    # Static terrain and land-cover features.
    dem = ee.Image(DEM_ID)
    slope = ee.Terrain.slope(dem)

    worldcover = (
        ee.ImageCollection(WORLDCOVER_ID)
        .first()
        .select("Map")
    )

    static_image = ee.Image.cat([
        dem.rename("ELEVATION"),
        slope.rename("SLOPE"),
        worldcover.rename("LANDCOVER_CODE"),
    ])

    static_values = static_image.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=30,
        maxPixels=1_000_000,
    ).getInfo()

    _require_values(
        static_values,
        ["ELEVATION", "SLOPE", "LANDCOVER_CODE"],
        "terrain and land cover",
    )

    elevation = float(static_values["ELEVATION"])
    slope_degrees = float(static_values["SLOPE"])
    landcover_value = float(static_values["LANDCOVER_CODE"])

    if not landcover_value.is_integer():
        raise RuntimeError(
            f"WorldCover returned a non-integer land-cover code: "
            f"{landcover_value}."
        )

    landcover_code = int(landcover_value)
    fixed_rainfall = float(rainfall_values["RAINFALL"])

    # Inputs match the current predict.py function signatures.
    susceptibility_inputs = {
        "latitude": latitude,
        "longitude": longitude,
        "elevation": elevation,
        "slope": slope_degrees,
        "rainfall": fixed_rainfall,
        "landcover_code": landcover_code,
    }

    temporal_inputs = {
        "latitude": latitude,
        "longitude": longitude,
        "rain_1d": float(rainfall_values["RAIN_1D"]),
        "rain_3d": float(rainfall_values["RAIN_3D"]),
        "rain_7d": float(rainfall_values["RAIN_7D"]),
        "rain_15d": float(rainfall_values["RAIN_15D"]),
        "rain_30d": float(rainfall_values["RAIN_30D"]),
        "elevation": elevation,
        "slope": slope_degrees,
        "landcover_code": landcover_code,
    }

    current_utc_date = datetime.now(timezone.utc).date()

    return {
        "metadata": {
        "reference_date": ref_date.isoformat(),
        "latest_chirps_observation_date": latest_date.isoformat(),
        "rainfall_window_end_date": (
            ref_date - timedelta(days=1)
        ).isoformat(),
        "rainfall_data_age_days": (
            datetime.now(timezone.utc).date()
            - (ref_date - timedelta(days=1))
        ).days,
        "climatology_period": "2015-01-01 to 2024-12-31",
        },
        "susceptibility_inputs": susceptibility_inputs,
        "temporal_inputs": temporal_inputs,
    }

def get_terrain_features(latitude, longitude):
    """Extract terrain and land-cover features from Earth Engine."""
    latitude, longitude = _validate_location(latitude, longitude)
    _initialize_earth_engine()

    point = ee.Geometry.Point([longitude, latitude])

    dem = ee.Image(DEM_ID)
    slope = ee.Terrain.slope(dem)

    worldcover = (
        ee.ImageCollection(WORLDCOVER_ID)
        .first()
        .select("Map")
    )

    image = ee.Image.cat([
        dem.rename("ELEVATION"),
        slope.rename("SLOPE"),
        worldcover.rename("LANDCOVER_CODE"),
    ])

    values = image.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=30,
        maxPixels=1_000_000,
    ).getInfo()

    _require_values(
        values,
        ["ELEVATION", "SLOPE", "LANDCOVER_CODE"],
        "terrain and land cover",
    )

    landcover = float(values["LANDCOVER_CODE"])
    if not landcover.is_integer():
        raise RuntimeError("Invalid land-cover code returned.")

    return {
        "ELEVATION": float(values["ELEVATION"]),
        "SLOPE": float(values["SLOPE"]),
        "LANDCOVER_CODE": int(landcover),
    }
