import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.features import compute_features
from src.preprocess import clean_dataframe
from src.rul_estimator import estimate_battery_rul


class FixedPredictionModel:
    def __init__(self, predictions: list[float]) -> None:
        self.predictions = np.asarray(predictions, dtype=float)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        assert len(features) == len(self.predictions)
        return self.predictions


def test_pipeline_smoke() -> None:
    """Verifies that the trained model exists, loads properly, and produces valid predictions."""
    root = Path(__file__).resolve().parents[1]
    model_path = root / "models" / "xgb_model.pkl"
    schema_path = root / "models" / "feature_schema.json"
    features_path = root / "data" / "processed" / "features.csv"

    # 1. Verify required files exist
    assert model_path.exists(), f"Model file missing: {model_path}"
    assert schema_path.exists(), f"Schema file missing: {schema_path}"
    assert features_path.exists(), f"Features file missing: {features_path}"

    # 2. Verify model loads
    model = joblib.load(model_path)
    assert hasattr(model, "predict"), "Loaded object does not implement predict()"

    # 3. Verify feature schema
    with open(schema_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    assert isinstance(feature_cols, list) and len(feature_cols) > 0
    assert "capacity" not in feature_cols

    # 4. Run single prediction check
    df = pd.read_csv(features_path)
    assert not df.empty, "Features dataframe is empty"
    sample_input = df[feature_cols].iloc[[0]].fillna(0.0)

    pred = model.predict(sample_input)
    assert len(pred) == 1, "Prediction output shape mismatch"
    assert 0.0 <= pred[0] <= 120.0, f"Predicted SoH {pred[0]} outside realistic range (0-120%)"
    print(f"\n[PASS] Smoke test succeeded. Sample SoH prediction: {pred[0]:.2f}%")


def test_processed_data_has_consistent_battery_health() -> None:
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "processed" / "features.csv")

    assert not df.empty
    assert not df["cycle"].eq(616).any()
    for _, battery in df.groupby("battery_id"):
        assert battery["SoH"].max() == pytest.approx(100.0)
        assert battery.sort_values("cycle")["SoH"].iloc[-1] < 80.0


def test_compute_features_normalizes_health_per_battery() -> None:
    telemetry = pd.DataFrame(
        {
            "battery_id": ["A", "A", "B", "B"],
            "cycle": [1, 2, 1, 2],
            "capacity": [2.0, 1.6, 1.0, 0.8],
            "voltage_mean": [4.0, 3.9, 4.1, 4.0],
            "current_mean": [1.0, 1.0, 2.0, 2.0],
            "temp_mean": [25.0, 26.0, 27.0, 28.0],
        }
    )

    features = compute_features(telemetry)
    soh_by_battery = features.groupby("battery_id")["SoH"].max()
    np.testing.assert_allclose(soh_by_battery.to_numpy(), 100.0)
    assert features.groupby("battery_id")["dV_dt"].first().eq(0.0).all()


def test_clean_dataframe_discards_invalid_capacity_without_cross_cell_fill() -> None:
    telemetry = pd.DataFrame(
        {
            "battery_id": ["A", "A", "A", "B"],
            "cycle": [1, 2, 3, 1],
            "capacity": [2.0, 1.8, np.nan, 1.0],
            "voltage_mean": [4.0, 3.8, 99.0, np.nan],
            "current_mean": [1.0, 1.0, 99.0, np.nan],
            "temp_mean": [25.0, 26.0, 99.0, np.nan],
        }
    )

    cleaned = clean_dataframe(telemetry)

    assert cleaned["cycle"].tolist() == [1, 2, 1]
    np.testing.assert_allclose(cleaned["SoH"].to_numpy(), [100.0, 90.0, 100.0])
    assert cleaned.loc[cleaned["battery_id"] == "B", "voltage_mean"].item() == 0.0


def test_rul_does_not_report_past_prediction_as_future_life() -> None:
    battery = pd.DataFrame(
        {
            "cycle": [1, 2, 3],
            "SoH": [100.0, 90.0, 85.0],
            "sensor": [1.0, 2.0, 3.0],
        }
    )

    result = estimate_battery_rul(
        battery,
        FixedPredictionModel([75.0, 85.0, 90.0]),
        ["sensor"],
    )

    assert result["observed_eol_cycle"] is None
    assert result["rul_cycles"] == "Not estimable from historical data"


def test_rul_reports_zero_after_observed_end_of_life() -> None:
    battery = pd.DataFrame(
        {
            "cycle": [1, 2, 3],
            "SoH": [100.0, 80.0, 75.0],
            "sensor": [1.0, 2.0, 3.0],
        }
    )

    result = estimate_battery_rul(
        battery,
        FixedPredictionModel([95.0, 85.0, 75.0]),
        ["sensor"],
    )

    assert result["observed_eol_cycle"] == 2
    assert result["rul_cycles"] == 0


if __name__ == "__main__":
    test_pipeline_smoke()
