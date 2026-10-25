import math
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

SUSCEPTIBILITY_LANDCOVER_CODES = {
    10, 20, 30, 40, 50, 60, 70, 80, 90, 100
}

TEMPORAL_LANDCOVER_CODES = {
    10, 20, 30, 40, 50, 60, 70, 80, 100
}
# --------------------------------------------------
# Paths and study-area limits
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEMPORAL_MODEL_PATH = PROJECT_ROOT / "models" / "landslide_3day_model.joblib"
SUSCEPTIBILITY_MODEL_PATH = (
    PROJECT_ROOT / "models" / "landslide_susceptibility_model.joblib"
)

# Approximate bounding box of the combined Himalayan study dataset.
# Update these values if the verified study-area extent changes.
TRAINING_AREA_BOUNDS = {
    "min_latitude": 29.135275,
    "max_latitude": 32.99,
    "min_longitude": 75.74,
    "max_longitude": 80.73,
}

SUSCEPTIBILITY_CLASS_ORDER = ["Low", "Moderate", "High", "Very High"]

TEMPORAL_FEATURES = {
    "latitude",
    "longitude",
    "RAIN_1D",
    "RAIN_3D",
    "RAIN_7D",
    "RAIN_15D",
    "RAIN_30D",
    "ELEVATION",
    "SLOPE",
    "LANDCOVER_CODE",
}

SUSCEPTIBILITY_FEATURES = {
    "LATITUDE",
    "LONGITUDE",
    "ELEVATION",
    "SLOPE",
    "RAINFALL",
    "LANDCOVER_CODE",
}


# --------------------------------------------------
# Validation helpers
# --------------------------------------------------

def _number(value, name):
    """Accept finite numeric values, but reject strings and booleans."""
    if isinstance(value, (str, bytes, bool)) or type(value).__name__ == "bool_":
        raise ValueError(f"{name} must be a numeric value, not {value!r}.")

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a valid number.") from exc

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite; NaN and infinity are invalid.")

    return result


def _probability(value, name):
    """Validate a probability or decision threshold."""
    result = _number(value, name)

    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")

    return result


def _validate_coordinates(latitude, longitude):
    latitude = _number(latitude, "latitude")
    longitude = _number(longitude, "longitude")

    if not -90 <= latitude <= 90:
        raise ValueError("latitude must be between -90 and 90 degrees.")

    if not -180 <= longitude <= 180:
        raise ValueError("longitude must be between -180 and 180 degrees.")

    bounds = TRAINING_AREA_BOUNDS

    if not (
        bounds["min_latitude"] <= latitude <= bounds["max_latitude"]
        and bounds["min_longitude"] <= longitude <= bounds["max_longitude"]
    ):
        raise ValueError(
            "Coordinates are outside the recorded study-area bounding box. "
            "Predictions outside the model's study region are unsupported."
        )

    return latitude, longitude


def _validate_terrain(elevation, slope):
    elevation = _number(elevation, "elevation")
    slope = _number(slope, "slope")

    if not -500 <= elevation <= 9000:
        raise ValueError("elevation must be between -500 and 9000 metres.")

    if not 0 <= slope <= 90:
        raise ValueError("slope must be between 0 and 90 degrees.")

    return elevation, slope


def _validate_rainfall(value, name):
    rainfall = _number(value, name)

    if rainfall < 0:
        raise ValueError(f"{name} cannot be negative.")

    return rainfall


def _validate_landcover_code(value, allowed_codes, model_name):
    code = _number(value, "landcover_code")

    if not code.is_integer() or int(code) not in allowed_codes:
        raise ValueError(
            f"Invalid landcover_code for {model_name}. "
            f"Allowed codes: {sorted(allowed_codes)}"
        )

    return int(code)


def _validate_cumulative_rainfall(rainfall_values):
    """Check that longer rolling rainfall totals do not decrease."""
    names = ["RAIN_1D", "RAIN_3D", "RAIN_7D", "RAIN_15D", "RAIN_30D"]

    for index in range(len(rainfall_values) - 1):
        current = rainfall_values[index]
        following = rainfall_values[index + 1]

        tolerance = max(
            1e-6,
            abs(current) * 1e-9,
            abs(following) * 1e-9,
        )

        if current > following + tolerance:
            raise ValueError(
                f"Inconsistent cumulative rainfall: {names[index]} "
                f"cannot exceed {names[index + 1]}."
            )


# --------------------------------------------------
# Model artifact validation and loading
# --------------------------------------------------

def _validate_feature_list(features, allowed_features, model_name):
    if not isinstance(features, (list, tuple)) or not features:
        raise RuntimeError(
            f"{model_name} artifact must contain a non-empty feature list."
        )

    if any(not isinstance(feature, str) for feature in features):
        raise RuntimeError(
            f"{model_name} artifact contains an invalid feature name."
        )

    if len(features) != len(set(features)):
        raise RuntimeError(
            f"{model_name} artifact contains duplicate feature names."
        )

    unsupported = set(features) - allowed_features

    if unsupported:
        raise RuntimeError(
            f"{model_name} artifact requires unsupported features: "
            f"{sorted(unsupported)}"
        )

    return list(features)


def _validate_classifier(model, model_name, feature_count):
    if not callable(getattr(model, "predict_proba", None)):
        raise RuntimeError(
            f"{model_name} does not support predict_proba()."
        )

    classes = getattr(model, "classes_", None)

    if classes is None:
        raise RuntimeError(
            f"{model_name} does not expose fitted class labels."
        )

    classes = list(classes)

    # These prediction functions expect a binary classifier with labels 0 and 1.
    if (
        len(classes) != 2
        or sum(label == 1 for label in classes) != 1
        or sum(label == 0 for label in classes) != 1
    ):
        raise RuntimeError(
            f"{model_name} must be a fitted binary classifier "
            "with class labels 0 and 1."
        )

    fitted_feature_count = getattr(model, "n_features_in_", None)

    if (
        fitted_feature_count is not None
        and fitted_feature_count != feature_count
    ):
        raise RuntimeError(
            f"{model_name} expects {fitted_feature_count} features, "
            f"but its artifact lists {feature_count}."
        )


def _load_artifact(path, model_name, allowed_features, required_keys):
    if not path.is_file():
        raise FileNotFoundError(
            f"{model_name} file was not found: {path}"
        )

    try:
        artifact = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {model_name} artifact at {path}. "
            "Check that the file is intact and compatible with the "
            "installed Python and scikit-learn versions."
        ) from exc

    if not isinstance(artifact, dict):
        raise RuntimeError(f"{model_name} artifact must be a dictionary.")

    missing_keys = required_keys - set(artifact)

    if missing_keys:
        raise RuntimeError(
            f"{model_name} artifact is missing keys: {sorted(missing_keys)}"
        )

    model = artifact["model"]

    features = _validate_feature_list(
        artifact["features"], allowed_features, model_name
    )

    _validate_classifier(model, model_name, len(features))

    threshold = _probability(
        artifact["threshold"], f"{model_name} threshold"
    )

    return artifact, model, features, threshold


@lru_cache(maxsize=1)
def _load_temporal_model():
    artifact, model, features, threshold = _load_artifact(
        path=TEMPORAL_MODEL_PATH,
        model_name="Three-day temporal model",
        allowed_features=TEMPORAL_FEATURES,
        required_keys={"model", "features", "threshold"},
    )

    return model, features, threshold


def _validate_susceptibility_classes(classes):
    if not isinstance(classes, dict):
        raise RuntimeError(
            "Susceptibility class boundaries must be a dictionary."
        )

    if set(classes) != set(SUSCEPTIBILITY_CLASS_ORDER):
        raise RuntimeError(
            "Susceptibility boundaries must define Low, Moderate, "
            "High and Very High."
        )

    normalized = {}

    for name in SUSCEPTIBILITY_CLASS_ORDER:
        interval = classes[name]

        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise RuntimeError(
                f"Invalid boundaries for susceptibility class {name}."
            )

        lower, upper = interval

        lower = None if lower is None else _probability(
            lower, f"{name} lower boundary"
        )
        upper = None if upper is None else _probability(
            upper, f"{name} upper boundary"
        )

        if lower is not None and upper is not None and lower >= upper:
            raise RuntimeError(
                f"Invalid boundary interval for susceptibility class {name}."
            )

        normalized[name] = (lower, upper)

    if normalized["Low"][0] is not None:
        raise RuntimeError("The Low class must start at the lowest score.")

    if normalized["Very High"][1] is not None:
        raise RuntimeError("The Very High class must end at the highest score.")

    for index in range(len(SUSCEPTIBILITY_CLASS_ORDER) - 1):
        current_name = SUSCEPTIBILITY_CLASS_ORDER[index]
        next_name = SUSCEPTIBILITY_CLASS_ORDER[index + 1]

        current_upper = normalized[current_name][1]
        next_lower = normalized[next_name][0]

        if current_upper is None or next_lower is None:
            raise RuntimeError("Susceptibility class boundaries are incomplete.")

        if not math.isclose(
            current_upper, next_lower, rel_tol=0.0, abs_tol=1e-9
        ):
            raise RuntimeError(
                "Susceptibility class boundaries have a gap or overlap "
                f"between {current_name} and {next_name}."
            )

    return normalized


@lru_cache(maxsize=1)
def _load_susceptibility_model():
    artifact, model, features, threshold = _load_artifact(
        path=SUSCEPTIBILITY_MODEL_PATH,
        model_name="Susceptibility model",
        allowed_features=SUSCEPTIBILITY_FEATURES,
        required_keys={
            "model",
            "features",
            "threshold",
            "susceptibility_classes",
        },
    )

    classes = _validate_susceptibility_classes(
        artifact["susceptibility_classes"]
    )

    return model, features, threshold, classes


# --------------------------------------------------
# Shared prediction helper
# --------------------------------------------------

def _positive_class_probability(model, features, values, model_name):
    missing_features = set(features) - set(values)

    if missing_features:
        raise RuntimeError(
            f"{model_name} is missing input features: "
            f"{sorted(missing_features)}"
        )

    sample = pd.DataFrame(
        [{feature: values[feature] for feature in features}],
        columns=features,
    )

    try:
        probabilities = model.predict_proba(sample)
    except Exception as exc:
        raise RuntimeError(
            f"{model_name} prediction failed. Check the model artifact, "
            "input features, and installed library versions."
        ) from exc

    if getattr(probabilities, "shape", None) != (
        1, len(model.classes_)
    ):
        raise RuntimeError(
            f"{model_name} returned an unexpected probability shape."
        )

    try:
        row = [_number(p, "model probability") for p in probabilities[0]]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{model_name} returned invalid probability values."
        ) from exc

    if any(p < 0 or p > 1 for p in row):
        raise RuntimeError(
            f"{model_name} returned probabilities outside [0, 1]."
        )

    if not math.isclose(sum(row), 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError(
            f"{model_name} returned probabilities that do not sum to 1."
        )

    positive_index = next(
        index for index, label in enumerate(model.classes_) if label == 1
    )

    return row[positive_index]


# --------------------------------------------------
# Three-day temporal prediction
# --------------------------------------------------

def predict_landslide_3day(
    latitude,
    longitude,
    rain_1d,
    rain_3d,
    rain_7d,
    rain_15d,
    rain_30d,
    elevation,
    slope,
    landcover_code,
):
    model, features, threshold = _load_temporal_model()

    latitude, longitude = _validate_coordinates(latitude, longitude)
    elevation, slope = _validate_terrain(elevation, slope)
    landcover_code = _validate_landcover_code(
    landcover_code, TEMPORAL_LANDCOVER_CODES, "three-day model"
)

    rainfall_values = [
        _validate_rainfall(rain_1d, "RAIN_1D"),
        _validate_rainfall(rain_3d, "RAIN_3D"),
        _validate_rainfall(rain_7d, "RAIN_7D"),
        _validate_rainfall(rain_15d, "RAIN_15D"),
        _validate_rainfall(rain_30d, "RAIN_30D"),
    ]

    _validate_cumulative_rainfall(rainfall_values)

    values = {
        "latitude": latitude,
        "longitude": longitude,
        "RAIN_1D": rainfall_values[0],
        "RAIN_3D": rainfall_values[1],
        "RAIN_7D": rainfall_values[2],
        "RAIN_15D": rainfall_values[3],
        "RAIN_30D": rainfall_values[4],
        "ELEVATION": elevation,
        "SLOPE": slope,
        "LANDCOVER_CODE": landcover_code,
    }

    score = _positive_class_probability(
        model, features, values, "Three-day temporal model"
    )
    predicted_class = int(score >= threshold)

    return {
        "horizon_days": 3,
        "experimental_score": round(score, 4),
        "threshold": threshold,
        "predicted_class": predicted_class,
        "risk_category": (
            "Higher model-indicated risk"
            if predicted_class == 1
            else "Lower model-indicated risk"
        ),
        "warning": (
            "Experimental research output, not a calibrated probability "
            "or an official landslide warning."
        ),
    }


# --------------------------------------------------
# Static susceptibility prediction
# --------------------------------------------------

def predict_susceptibility(
    latitude,
    longitude,
    elevation,
    slope,
    rainfall,
    landcover_code,
):
    model, features, _, classes = _load_susceptibility_model()

    latitude, longitude = _validate_coordinates(latitude, longitude)
    elevation, slope = _validate_terrain(elevation, slope)
    rainfall = _validate_rainfall(rainfall, "rainfall")
    landcover_code = _validate_landcover_code(
    landcover_code, SUSCEPTIBILITY_LANDCOVER_CODES, "susceptibility model"
)

    values = {
        "LATITUDE": latitude,
        "LONGITUDE": longitude,
        "ELEVATION": elevation,
        "SLOPE": slope,
        "RAINFALL": rainfall,
        "LANDCOVER_CODE": landcover_code,
    }

    score = _positive_class_probability(
        model, features, values, "Susceptibility model"
    )

    category = next(
        (
            name
            for name in SUSCEPTIBILITY_CLASS_ORDER
            if (
                classes[name][0] is None
                or score >= classes[name][0]
            )
            and (
                classes[name][1] is None
                or score < classes[name][1]
            )
        ),
        None,
    )

    if category is None:
        raise RuntimeError(
            "The susceptibility score could not be assigned to a class."
        )

    return {
        "model_score": round(score, 4),
        "susceptibility_class": category,
        "warning": (
            "Research estimate, not a calibrated probability "
            "or official warning."
        ),
    }