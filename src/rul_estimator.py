from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main() -> None:
    """Estimate RUL using trained model predictions."""
    root = Path(__file__).resolve().parents[1]
    
    data_dir = root / "data"
    input_path = data_dir / "processed" / "features.csv"
    models_dir = root / "models"
    model_path = models_dir / "xgb_model.pkl"
    
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return
    
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("[ERROR] Please run src/soh_model.py first")
        return
    
    print(f"[INFO] Loading features from: {input_path}")
    print(f"[INFO] Loading model from: {model_path}")
    
    try:
        df = pd.read_csv(input_path)
        print(f"[INFO] Loaded {len(df)} rows")
    except Exception as exc:
        print(f"[ERROR] Failed to load CSV: {exc}")
        return
    
    try:
        model = joblib.load(model_path)
        print(f"[INFO] Loaded model successfully")
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}")
        return
    
    if df.empty:
        print("[ERROR] Input data is empty")
        return
    
    # Get unique batteries
    batteries = df["battery_id"].unique()
    print(f"[INFO] Processing {len(batteries)} batteries")
    
    print(f"\n[INFO] ===== RUL ESTIMATION =====\n")
    
    # Process each battery
    for battery_id in sorted(batteries):
        df_batt = df[df["battery_id"] == battery_id].sort_values("cycle").reset_index(drop=True)
        
        import json
        schema_path = models_dir / "feature_schema.json"
        if schema_path.exists():
            with open(schema_path) as f:
                feature_cols = json.load(f)
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                print(f"[ERROR] Feature mismatch. Missing: {missing}")
                continue
        else:
            # Fallback if schema doesn't exist yet
            feature_cols = [col for col in df.columns if col not in ["battery_id", "cycle", "SoH"]]
            
        X_batt = df_batt[feature_cols].fillna(0.0)
        
        # Predict SoH
        try:
            soh_pred = model.predict(X_batt)
        except Exception as exc:
            print(f"[WARN] Failed to predict for {battery_id}: {exc}")
            continue
        
        # Find cycle where SoH drops below 80%
        soh_fail = 80.0
        cycles_below_80 = np.where(soh_pred < soh_fail)[0]
        # Map index to actual cycle numbers
        actual_cycles = df_batt["cycle"].values
        current_cycle = int(df_batt["cycle"].iloc[-1])
        if len(cycles_below_80) > 0:
            k_fail_cycle = actual_cycles[cycles_below_80[0]]
            k_fail = int(k_fail_cycle)
            rul = max(k_fail - current_cycle, 0)
        else:
            k_fail = "N/A"
            rul = "Stable (>80% SoH)"
            
        current_soh = df_batt["SoH"].iloc[-1]
        
        print(f"[INFO] Battery {battery_id}:")
        print(f"       Current cycle: {current_cycle}")
        print(f"       Current SoH: {current_soh:.2f}%")
        print(f"       Predicted failure cycle: {k_fail}")
        print(f"       RUL (cycles remaining): {rul}")
        print()


if __name__ == "__main__":
    main()
