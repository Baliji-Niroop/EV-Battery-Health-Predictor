from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, TypedDict

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Use only measurements available without a direct capacity/SoH-derived input.
FEATURE_COLS = [
    "voltage_mean",
    "current_mean",
    "temp_mean",
    "dV_dt",
    "dT_dt",
    "internal_resistance_proxy",
]


class EvaluationResults(TypedDict):
    xgb_mae: float
    xgb_rmse: float
    baseline_mae: float
    baseline_rmse: float
    model: XGBRegressor
    y_test: pd.Series
    y_pred: np.ndarray


def train_and_evaluate(
    df: pd.DataFrame,
    features: list[str],
    test_batteries: Sequence[str] = ("B0007",),
) -> EvaluationResults:
    """Trains XGBoost and Linear Regression on training batteries and tests on held-out battery.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature columns, 'battery_id', and target 'SoH'.
    features : list[str]
        List of feature column names.
    test_batteries : list[str]
        Battery identifiers reserved strictly for testing (avoids data leakage).

    Returns
    -------
    dict[str, float]
        Evaluation metrics (MAE and RMSE in % SoH) for XGBoost and Baseline Linear Regression.
    """
    missing_features = set(features).difference(df.columns)
    if missing_features:
        raise ValueError(f"Missing model features: {sorted(missing_features)}")
    if "battery_id" not in df or "SoH" not in df:
        raise ValueError("Training data must include battery_id and SoH columns.")

    df_clean = df.copy()
    for column in [*features, "SoH"]:
        df_clean[column] = df_clean.groupby("battery_id")[column].transform(
            lambda values: values.ffill().bfill()
        ).fillna(0.0)

    train_mask = ~df_clean["battery_id"].isin(test_batteries)
    test_mask = df_clean["battery_id"].isin(test_batteries)
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Both training and held-out battery rows are required.")

    X_train = df_clean.loc[train_mask, features]
    y_train = df_clean.loc[train_mask, "SoH"]

    X_test = df_clean.loc[test_mask, features]
    y_test = df_clean.loc[test_mask, "SoH"]

    # Baseline: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_mae = float(mean_absolute_error(y_test, lr_pred))
    lr_rmse = float(np.sqrt(mean_squared_error(y_test, lr_pred)))

    # Main Model: XGBoost Regressor
    xgb = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_mae = float(mean_absolute_error(y_test, xgb_pred))
    xgb_rmse = float(np.sqrt(mean_squared_error(y_test, xgb_pred)))

    return {
        "xgb_mae": xgb_mae,
        "xgb_rmse": xgb_rmse,
        "baseline_mae": lr_mae,
        "baseline_rmse": lr_rmse,
        "model": xgb,
        "y_test": y_test,
        "y_pred": xgb_pred,
    }


def main() -> None:
    """Trains the SoH prediction models, saves artifacts, and outputs benchmark metrics."""
    np.random.seed(42)
    root = Path(__file__).resolve().parents[1]

    data_dir = root / "data"
    input_path = data_dir / "processed" / "features.csv"
    models_dir = root / "models"
    outputs_dir = root / "outputs"

    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"[ERROR] Features file not found at: {input_path}")

    print(f"[INFO] Loading features from: {input_path}")
    df = pd.read_csv(input_path)
    missing_features = set(FEATURE_COLS + ["battery_id", "SoH"]).difference(df.columns)
    if missing_features:
        raise SystemExit(f"[ERROR] Features data is missing columns: {sorted(missing_features)}")

    # Primary Benchmark: Battery-wise split (Train on B0005, B0006, B0018; Test on B0007)
    test_battery = ["B0007"]
    print(f"[INFO] Splitting data: Battery-wise split (Held-out test: {test_battery})")
    results = train_and_evaluate(df, FEATURE_COLS, test_batteries=test_battery)

    # Secondary Benchmark: Random 80/20 cycle split (to document data leakage impact)
    df_clean = df.copy()
    for column in [*FEATURE_COLS, "SoH"]:
        df_clean[column] = df_clean.groupby("battery_id")[column].transform(
            lambda values: values.ffill().bfill()
        ).fillna(0.0)
    X_rnd = df_clean[FEATURE_COLS]
    y_rnd = df_clean["SoH"]
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_rnd, y_rnd, test_size=0.2, random_state=42
    )
    xgb_rnd = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    xgb_rnd.fit(X_tr_r, y_tr_r)
    rnd_preds = xgb_rnd.predict(X_te_r)
    rnd_mae = float(mean_absolute_error(y_te_r, rnd_preds))
    rnd_rmse = float(np.sqrt(mean_squared_error(y_te_r, rnd_preds)))

    # Display results
    print("\n" + "=" * 50)
    print("  MODEL EVALUATION RESULTS (Units: % SoH)")
    print("=" * 50)
    print(f"Primary Battery-wise Split (Test: B0007):")
    print(f"  * XGBoost MAE:  {results['xgb_mae']:.4f}% SoH")
    print(f"  * XGBoost RMSE: {results['xgb_rmse']:.4f}% SoH")
    print(f"  * Baseline LR MAE:  {results['baseline_mae']:.4f}% SoH")
    print(f"  * Baseline LR RMSE: {results['baseline_rmse']:.4f}% SoH")
    print(f"\nSecondary Random-cycle Split (With Cycle Leakage):")
    print(f"  * XGBoost MAE:  {rnd_mae:.4f}% SoH")
    print(f"  * XGBoost RMSE: {rnd_rmse:.4f}% SoH")
    print("=" * 50)

    # Save metrics file
    metrics_path = outputs_dir / "metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Primary_Test_Batteries: {test_battery}\n")
        f.write(f"Primary_XGBoost_MAE_pct: {results['xgb_mae']:.4f}\n")
        f.write(f"Primary_XGBoost_RMSE_pct: {results['xgb_rmse']:.4f}\n")
        f.write(f"Primary_Baseline_LR_MAE_pct: {results['baseline_mae']:.4f}\n")
        f.write(f"Primary_Baseline_LR_RMSE_pct: {results['baseline_rmse']:.4f}\n")
        f.write(f"Secondary_Random_MAE_pct: {rnd_mae:.4f}\n")
        f.write(f"Secondary_Random_RMSE_pct: {rnd_rmse:.4f}\n")
        # Legacy compatible keys for backward compatibility
        f.write(f"MAE: {results['xgb_mae']:.4f}\n")
        f.write(f"RMSE: {results['xgb_rmse']:.4f}\n")
        f.write(f"Baseline_MAE: {results['baseline_mae']:.4f}\n")
    print(f"[INFO] Saved metrics to: {metrics_path}")

    # Save model and feature schema
    model_path = models_dir / "xgb_model.pkl"
    schema_path = models_dir / "feature_schema.json"

    joblib.dump(results["model"], model_path)
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    print(f"[SUCCESS] Model saved: {model_path}")
    print(f"[SUCCESS] Feature schema saved: {schema_path}")

    # Generate and save scatter plot
    try:
        import matplotlib.pyplot as plt

        y_test = results["y_test"]
        y_pred = results["y_pred"]

        plt.figure(figsize=(8, 6), dpi=150)
        plt.scatter(y_test, y_pred, alpha=0.6, color="#2563EB", label="Predictions")
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--",
            lw=2,
            label="Ideal Perfect Fit (y = x)",
        )
        plt.xlabel("Actual SoH (%)", fontsize=11, fontweight="bold")
        plt.ylabel("Predicted SoH (%)", fontsize=11, fontweight="bold")
        plt.title(
            f"Actual vs. Predicted SoH on Held-out Battery B0007 (XGBoost MAE: {results['xgb_mae']:.2f}%)",
            fontsize=12,
            fontweight="bold",
        )
        plt.legend(frameon=True)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        scatter_path = outputs_dir / "soh_prediction_scatter.png"
        plt.savefig(scatter_path)
        plt.close()
        print(f"[SUCCESS] Scatter plot saved: {scatter_path}")
    except Exception as exc:
        print(f"[WARN] Failed to plot scatter: {exc}")


if __name__ == "__main__":
    main()
