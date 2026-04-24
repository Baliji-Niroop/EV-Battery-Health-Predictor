from __future__ import annotations

from pathlib import Path
from typing import List

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
    print(f"[INFO] Sample rows (first 5):")
    print(df.head())
    
    # Save features
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(output_path, index=False)
        print(f"\n[INFO] ✓ Saved features to: {output_path}")
    except Exception as exc:
        print(f"[ERROR] Failed to save CSV: {exc}")


if __name__ == "__main__":
    main()
