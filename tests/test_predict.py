import pytest

from src.predict import (
    predict_landslide_3day,
    predict_susceptibility,
)


def test_temporal_model_prediction():
    result = predict_landslide_3day(
        30.7, 78.4, 10, 25, 50, 100, 200, 1800, 30, 20
    )

    assert 0 <= result["experimental_score"] <= 1
    assert result["predicted_class"] in (0, 1)
    assert result["horizon_days"] == 3


def test_susceptibility_model_prediction():
    result = predict_susceptibility(
        30.7, 78.4, 1800, 30, 50, 20
    )

    assert 0 <= result["model_score"] <= 1
    assert result["susceptibility_class"] in {
        "Low", "Moderate", "High", "Very High"
    }


@pytest.mark.parametrize("rainfall", [-1, float("nan"), float("inf")])
def test_invalid_susceptibility_rainfall(rainfall):
    with pytest.raises(ValueError):
        predict_susceptibility(30.7, 78.4, 1800, 30, rainfall, 2)


def test_invalid_slope():
    with pytest.raises(ValueError):
        predict_susceptibility(30.7, 78.4, 1800, 91, 50, 2)


def test_invalid_coordinates():
    with pytest.raises(ValueError):
        predict_susceptibility(95, 78.4, 1800, 30, 50, 2)


def test_inconsistent_temporal_rainfall():
    with pytest.raises(ValueError):
        predict_landslide_3day(
            30.7, 78.4, 20, 10, 50, 100, 200, 1800, 30, 2
        )


def test_missing_elevation():
    with pytest.raises(ValueError):
        predict_susceptibility(30.7, 78.4, None, 30, 50, 2)

def test_missing_temporal_model(monkeypatch, tmp_path):
    import src.predict as predict_module

    monkeypatch.setattr(
        predict_module,
        "TEMPORAL_MODEL_PATH",
        tmp_path / "missing_model.joblib",
    )
    predict_module._load_temporal_model.cache_clear()

    try:
        with pytest.raises(FileNotFoundError, match="not found"):
            predict_landslide_3day(
                30.7, 78.4, 10, 25, 50, 100, 200, 1800, 30, 2
            )
    finally:
        predict_module._load_temporal_model.cache_clear()


def test_missing_susceptibility_model(monkeypatch, tmp_path):
    import src.predict as predict_module

    monkeypatch.setattr(
        predict_module,
        "SUSCEPTIBILITY_MODEL_PATH",
        tmp_path / "missing_model.joblib",
    )
    predict_module._load_susceptibility_model.cache_clear()

    try:
        with pytest.raises(FileNotFoundError, match="not found"):
            predict_susceptibility(30.7, 78.4, 1800, 30, 50, 2)
    finally:
        predict_module._load_susceptibility_model.cache_clear()

@pytest.mark.parametrize("rainfall", [-1, float("nan"), float("inf")])
def test_invalid_temporal_rainfall(rainfall):
    with pytest.raises(ValueError):
        predict_landslide_3day(
            30.7, 78.4, rainfall, 25, 50, 100, 200, 1800, 30, 2
        )


@pytest.mark.parametrize("landcover", [-1, 2.5])
def test_invalid_landcover_code(landcover):
    with pytest.raises(ValueError):
        predict_susceptibility(
            30.7, 78.4, 1800, 30, 50, landcover
        )