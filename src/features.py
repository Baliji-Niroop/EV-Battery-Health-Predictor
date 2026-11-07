from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes engineered features from cleaned battery cycle telemetry.

    Features generated:
    1. capacity_fade_rate: Difference in capacity between consecutive cycles.
    2. rolling_SoH_mean: 5-cycle rolling average of State of Health.
    3. rolling_temp_mean: 5-cycle rolling average of operating temperature.
    4. dV_dt: Cycle-over-cycle change in mean voltage.
    5. dT_dt: Cycle-over-cycle change in mean temperature.
    6. internal_resistance_proxy: Ratio of mean voltage to mean current (Ohms).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned telemetry records containing battery_id, cycle, capacity,
        voltage_mean, current_mean, and temp_mean.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with engineered features.
    """
    required_columns = {
        "battery_id",
        "cycle",
        "capacity",
        "voltage_mean",
        "current_mean",
        "temp_mean",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required telemetry columns: {sorted(missing_columns)}")

    df = df.sort_values(["battery_id", "cycle"]).reset_index(drop=True).copy()
    initial_capacity = df.groupby("battery_id")["capacity"].transform("first")
    df["SoH"] = df["capacity"].div(initial_capacity.where(initial_capacity > 0)) * 100.0

    # Feature 1: Capacity fade rate (per-cycle delta in Ampere-hours)
    df["capacity_fade_rate"] = df.groupby("battery_id")["capacity"].transform(
        lambda x: (x.shift(1) - x).fillna(0.0)
    )

    # Feature 2: Rolling SoH mean (window=5 cycles to smooth noise)
    df["rolling_SoH_mean"] = df.groupby("battery_id")["SoH"].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )

    # Feature 3: Rolling temperature mean (window=5 cycles)
    df["rolling_temp_mean"] = df.groupby("battery_id")["temp_mean"].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )

    # Feature 4: dV_dt - voltage shift across consecutive cycles
    df["dV_dt"] = df.groupby("battery_id")["voltage_mean"].diff().fillna(0.0)

    # Feature 5: dT_dt - thermal shift across consecutive cycles
    df["dT_dt"] = df.groupby("battery_id")["temp_mean"].diff().fillna(0.0)

    # Feature 6: Internal resistance proxy (R = V / I)
    df["internal_resistance_proxy"] = (
        df["voltage_mean"] / df["current_mean"].replace(0, np.nan)
    ).fillna(0.0)

    numeric_columns = df.select_dtypes(include="number").columns
    for col in numeric_columns:
        df[col] = df.groupby("battery_id")[col].transform(
            lambda values: values.ffill().bfill()
        ).fillna(0.0)

    return df


def main() -> None:
    """Loads cleaned data, executes feature engineering, and exports features.csv."""
    root = Path(__file__).resolve().parents[1]

    data_dir = root / "data"
    input_path = data_dir / "processed" / "cleaned_data.csv"
    output_path = data_dir / "processed" / "features.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        if output_path.exists():
            print(f"[INFO] Refreshing existing processed features from: {output_path}")
            df = pd.read_csv(output_path)
        else:
            raise SystemExit(
                f"[ERROR] Cleaned data and existing features are missing. "
                f"Expected {input_path} or {output_path}."
            )
    else:
        print(f"[INFO] Loading cleaned data: {input_path}")
        df = pd.read_csv(input_path)

    if df.empty:
        raise SystemExit("[ERROR] Input data is empty.")

    print("[INFO] Computing engineered features...")
    df_features = compute_features(df)

    df_features.to_csv(output_path, index=False)
    print(f"[SUCCESS] Features saved: {output_path} ({len(df_features)} rows)")


if __name__ == "__main__":
    main()
