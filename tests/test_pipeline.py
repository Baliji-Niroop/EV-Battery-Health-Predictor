import json
from pathlib import Path
import joblib
import pandas as pd


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

    # 4. Run single prediction check
    df = pd.read_csv(features_path)
    assert not df.empty, "Features dataframe is empty"
    sample_input = df[feature_cols].iloc[[0]].fillna(0.0)

    pred = model.predict(sample_input)
    assert len(pred) == 1, "Prediction output shape mismatch"
    assert 0.0 <= pred[0] <= 120.0, f"Predicted SoH {pred[0]} outside realistic range (0-120%)"
    print(f"\n[PASS] Smoke test succeeded. Sample SoH prediction: {pred[0]:.2f}%")


if __name__ == "__main__":
    test_pipeline_smoke()
