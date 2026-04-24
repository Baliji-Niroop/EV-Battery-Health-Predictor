from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


def main() -> None:
    """Load features and train XGBoost regressor for SoH prediction."""
    np.random.seed(42)
    root = Path(__file__).resolve().parents[1]
    
    data_dir = root / "data"
    input_path = data_dir / "processed" / "features.csv"
    
    # Ensure models directory exists
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "xgb_model.pkl"
    
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return
    
    print(f"[INFO] Loading features from: {input_path}")
    
    try:
        df = pd.read_csv(input_path)
        print(f"[INFO] Loaded {len(df)} rows")
    except Exception as exc:
        print(f"[ERROR] Failed to load CSV: {exc}")
        return
    
    if df.empty:
        print("[ERROR] Input data is empty")
        return
    
    # Prepare data
    print("[DEBUG] Preparing training data")
    
    # Target: SoH
    if "SoH" not in df.columns:
        print("[ERROR] SoH column not found")
        return
    
    # Features: drop non-numeric columns and target columns
    feature_cols = [col for col in df.columns if col not in ["battery_id", "cycle", "SoH"]]
    
    # Fill any missing values before training
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    X = df[feature_cols]
    y = df["SoH"]
    
    # Train-test split
    print("[DEBUG] Splitting data: 80/20")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"[DEBUG] Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train XGBoost
    print("[INFO] Training XGBoost Regressor")
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    print("[INFO] Training complete")
    
    print("\n[INFO] ===== BASELINE COMPARISON =====")
    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)
    baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
    print(f"Baseline (Linear Regression) MAE: {baseline_mae:.4f}")
    
    # Evaluate
    print("\n[DEBUG] Evaluating model")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n[INFO] ===== MODEL PERFORMANCE =====")
    print(f"XGBoost MAE: {mae:.4f}")
    print(f"[INFO] RMSE: {rmse:.4f}")
    
    # Save metrics to text file
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = outputs_dir / "metrics.txt"
    try:
        with open(metrics_path, "w") as f:
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"Baseline_MAE: {baseline_mae:.4f}\n")
        print(f"[INFO] ✓ Saved metrics to: {metrics_path}")
    except Exception as exc:
        print(f"[ERROR] Failed to save metrics: {exc}")
    
    # Save a plot of the predictions
    try:
        import matplotlib.pyplot as plt
        outputs_dir = root / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predictions')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
        plt.xlabel('Actual SoH')
        plt.ylabel('Predicted SoH')
        plt.title('Actual vs Predicted State of Health (SoH)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = outputs_dir / "soh_prediction_scatter.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"[INFO] ✓ Saved prediction scatter plot to: {plot_path}")
    except Exception as e:
        print(f"[WARN] Could not generate plot: {e}")
    
    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(model, model_path)
        print(f"\n[INFO] ✓ Saved model to: {model_path}")
    except Exception as exc:
        print(f"[ERROR] Failed to save model: {exc}")


if __name__ == "__main__":
    main()
