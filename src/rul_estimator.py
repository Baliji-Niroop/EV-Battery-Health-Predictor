from __future__ import annotations

import json
from pathlib import Path
import joblib
import pandas as pd


def estimate_battery_rul(
    df_battery: pd.DataFrame,
    model: object,
    feature_cols: list[str],
    failure_threshold_soh: float = 80.0,
) -> dict[str, object]:
    """Reports observed end-of-life and RUL for a battery profile.

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
        Observed health, the first measured cycle at or below the threshold,
        and zero RUL if that threshold has already been reached. Future RUL
        cannot be inferred from historical samples alone.
    """
    X_batt = df_battery[feature_cols].fillna(0.0)
    soh_pred = model.predict(X_batt)

    current_cycle = int(df_battery["cycle"].iloc[-1])
    current_soh = float(df_battery["SoH"].iloc[-1])

    if current_soh <= failure_threshold_soh:
        eol_cycle = int(
            df_battery.loc[
                df_battery["SoH"] <= failure_threshold_soh, "cycle"
            ].iloc[0]
        )
        rul: int | str = 0
    else:
        eol_cycle = None
        rul = "Not estimable from historical data"

    return {
        "current_cycle": current_cycle,
        "current_soh": current_soh,
        "observed_eol_cycle": eol_cycle,
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
        raise SystemExit(f"[ERROR] Features file not found at: {input_path}")

    if not model_path.exists():
        raise SystemExit(
            f"[ERROR] Model file not found at: {model_path}. Run src/soh_model.py first."
        )

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
        missing_features = set(feature_cols).difference(df_batt.columns)
        if missing_features:
            raise SystemExit(
                f"[ERROR] Battery {b_id} is missing model features: "
                f"{sorted(missing_features)}"
            )
        res = estimate_battery_rul(df_batt, model, feature_cols)

        eol_cycle = res["observed_eol_cycle"]
        k_str = (
            f"Cycle {eol_cycle}"
            if eol_cycle is not None
            else "Not reached in observed data"
        )
        rul_str = f"{res['rul_cycles']} cycles" if isinstance(res['rul_cycles'], int) else str(res['rul_cycles'])

        print(f"[Battery {b_id}]")
        print(f"  * Latest Tested Cycle:   {res['current_cycle']}")
        print(f"  * Current State of Health: {res['current_soh']:.2f}%")
        print(f"  * First Observed EOL Cycle: {k_str}")
        print(f"  * Remaining Useful Life:  {rul_str}")
        print("-" * 55)


if __name__ == "__main__":
    main()
