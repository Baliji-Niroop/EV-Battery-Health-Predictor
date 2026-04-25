from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    """Load cleaned data and create engineered features."""
    root = Path(__file__).resolve().parents[1]
    
    data_dir = root / "data"
    input_path = data_dir / "processed" / "cleaned_data.csv"
    output_path = data_dir / "processed" / "features.csv"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: {input_path} is missing!")
        print("Please run preprocess.py first to generate the cleaned data.")
        import sys
        sys.exit(1)
    
    print(f"[INFO] Loading cleaned data from: {input_path}")
    
    try:
        df = pd.read_csv(input_path)
        print(f"[INFO] Loaded {len(df)} rows")
        print(f"[INFO] Columns: {list(df.columns)}")
        # Cycle‑count sanity check
        min_cycles = df.groupby("battery_id")["cycle"].count().min()
        if min_cycles < 50:
            print(f"[WARN] Battery with only {min_cycles} cycles detected — results may be unreliable")
    except Exception as exc:
        print(f"[ERROR] Failed to load CSV: {exc}")
        return
    
    if df.empty:
        print("Error: Input data is empty")
        return
    
    # Sort by battery_id and cycle
    df = df.sort_values(["battery_id", "cycle"]).reset_index(drop=True)
    print("Sorted by battery_id and cycle")
    
    # Ensure SoH exists
    if "SoH" not in df.columns:
        print("[DEBUG] Computing SoH from capacity")
        max_capacity = df["capacity"].max(skipna=True)
        if pd.isna(max_capacity) or max_capacity == 0:
            print("[WARN] Cannot compute SoH - no valid capacity data")
            df["SoH"] = np.nan
        else:
            df["SoH"] = (df["capacity"] / max_capacity) * 100.0
    
    # Feature 1: Capacity fade rate (diff of capacity per battery)
    print("[DEBUG] Computing capacity_fade_rate")
    df["capacity_fade_rate"] = df.groupby("battery_id")["capacity"].transform(
        lambda x: (x.shift(1) - x).fillna(0.0)
    )
    
    # Feature 2: Rolling SoH mean (window=5)
    print("[DEBUG] Computing rolling_SoH_mean")
    df["rolling_SoH_mean"] = (
        df.groupby("battery_id")["SoH"].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
    )
    
    # Feature 3: Rolling temp mean (window=5)
    print("[DEBUG] Computing rolling_temp_mean")
    df["rolling_temp_mean"] = (
        df.groupby("battery_id")["temp_mean"].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
    )

    # New Feature 4: dV/dt – voltage change per cycle
    print("[DEBUG] Computing dV_dt")
    df["dV_dt"] = df.groupby("battery_id")["voltage_mean"].diff().fillna(0.0)

    # New Feature 5: dT/dt – temperature change per cycle
    print("[DEBUG] Computing dT_dt")
    df["dT_dt"] = df.groupby("battery_id")["temp_mean"].diff().fillna(0.0)

    # New Feature 6: Internal resistance proxy (Ohm) approximated as voltage/current
    print("[DEBUG] Computing internal_resistance_proxy")
    # Vectorized calculation to replace slow row‑by‑row apply
    df["internal_resistance_proxy"] = (
        df["voltage_mean"] / df["current_mean"].replace(0, np.nan)
    ).fillna(0.0)
    
    # Fill NaN values safely
    print("[DEBUG] Filling NaN values")
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].bfill()
            df[col] = df[col].ffill()
            df[col] = df[col].fillna(0.0)
    
    print(f"\n[INFO] ===== FEATURE ENGINEERING COMPLETE =====")
    print(f"[INFO] Total rows: {len(df)}")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    # Save features
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(output_path, index=False)
        print(f"\n[INFO] Saved features to: {output_path}")
    except Exception as exc:
        print(f"[ERROR] Failed to save CSV: {exc}")


if __name__ == "__main__":
    main()
