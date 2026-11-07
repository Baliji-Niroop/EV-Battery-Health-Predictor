from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def estimate_battery_rul(
    df_battery: pd.DataFrame,
    model: object,
    feature_cols: list[str],
    failure_threshold_soh: float = 80.0,
) -> dict[str, object]:
    """Estimates the Remaining Useful Life (RUL) for a battery profile.

    Parameters
    ----------
    df_battery : pd.DataFrame
        Cycles for a single battery sorted by cycle number.
    model : object
        Trained regression model supporting predict().
    feature_cols : list[str]
        List of feature columns matching model training schema.
    failure_threshold_soh : float, default 80.0
        Standard EV end-of-life battery health threshold (80.0%).

    Returns
    -------
    dict[str, object]
        Computed RUL metrics including current cycle, current SoH,
        predicted failure cycle, and remaining cycles.
    """
    X_batt = df_battery[feature_cols].fillna(0.0)
    soh_pred = model.predict(X_batt)

    current_cycle = int(df_battery["cycle"].iloc[-1])
    current_soh = float(df_battery["SoH"].iloc[-1])

    # Find first cycle where predicted SoH dips below threshold
    cycles_below_threshold = np.where(soh_pred < failure_threshold_soh)[0]
    actual_cycles = df_battery["cycle"].values

    if len(cycles_below_threshold) > 0:
        k_fail = int(actual_cycles[cycles_below_threshold[0]])
        rul = max(k_fail - current_cycle, 0)
    else:
        k_fail = None
        rul = "Healthy (>80% SoH)"

    return {
        "current_cycle": current_cycle,
        "current_soh": current_soh,
        "predicted_failure_cycle": k_fail,
        "rul_cycles": rul,
        "soh_predictions": soh_pred,
    }


def main() -> None:
    """Loads dataset and trained model to estimate RUL across all batteries."""
    root = Path(__file__).resolve().parents[1]

    data_dir = root / "data"
    input_path = data_dir / "processed" / "features.csv"
    models_dir = root / "models"
    model_path = models_dir / "xgb_model.pkl"
    schema_path = models_dir / "feature_schema.json"

    if not input_path.exists():
        print(f"[ERROR] Features file not found at: {input_path}")
        return

    if not model_path.exists():
        print(f"[ERROR] Model file not found at: {model_path}. Run src/soh_model.py first.")
        return

    print(f"[INFO] Loading features from: {input_path}")
    df = pd.read_csv(input_path)

    print(f"[INFO] Loading model from: {model_path}")
    model = joblib.load(model_path)

    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
    else:
        feature_cols = [c for c in df.columns if c not in ["battery_id", "cycle", "SoH"]]

    batteries = sorted(df["battery_id"].unique())
    print(f"\n" + "=" * 55)
    print("  REMAINING USEFUL LIFE (RUL) SUMMARY")
    print("=" * 55)

    for b_id in batteries:
        df_batt = df[df["battery_id"] == b_id].sort_values("cycle").reset_index(drop=True)
        res = estimate_battery_rul(df_batt, model, feature_cols)

        k_str = f"Cycle {res['predicted_failure_cycle']}" if res["predicted_failure_cycle"] else "None (Healthy)"
        rul_str = f"{res['rul_cycles']} cycles" if isinstance(res['rul_cycles'], int) else str(res['rul_cycles'])

        print(f"[Battery {b_id}]")
        print(f"  * Latest Tested Cycle:   {res['current_cycle']}")
        print(f"  * Current State of Health: {res['current_soh']:.2f}%")
        print(f"  * First Cycle Below 80%:  {k_str}")
        print(f"  * Remaining Useful Life:  {rul_str}")
        print("-" * 55)


if __name__ == "__main__":
    main()
