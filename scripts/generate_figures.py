"""Generates publication-quality figures for the EV Battery Health Predictor repository.

All numbers and curves are computed directly from the processed NASA battery telemetry.
Figures are exported to docs/images/ at 150+ DPI.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.soh_model import FEATURE_COLS, train_and_evaluate

DATA_PATH = ROOT_DIR / "data" / "processed" / "features.csv"
MODEL_PATH = ROOT_DIR / "models" / "xgb_model.pkl"
SCHEMA_PATH = ROOT_DIR / "models" / "feature_schema.json"
IMAGES_DIR = ROOT_DIR / "docs" / "images"

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "figure.titlesize": 13,
})


def load_dataset() -> pd.DataFrame:
    """Loads and prepares the battery features dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df.ffill().bfill().fillna(0.0)


def figure_1_soh_degradation(df: pd.DataFrame) -> None:
    """1. SoH degradation curves for B0005, B0006, B0007 with 80% threshold line."""
    plt.figure(figsize=(9, 5.5), dpi=200)

    battery_colors = {
        "B0005": "#2563EB",  # Blue
        "B0006": "#DC2626",  # Red
        "B0007": "#059669",  # Green
    }

    for b_id in ["B0005", "B0006", "B0007"]:
        b_data = df[df["battery_id"] == b_id].sort_values("cycle")
        plt.plot(
            b_data["cycle"],
            b_data["SoH"],
            label=f"Battery {b_id}",
            color=battery_colors.get(b_id, "#333"),
            linewidth=2.0,
            alpha=0.9,
        )

    plt.axhline(
        y=80.0,
        color="#D97706",
        linestyle="--",
        linewidth=2.0,
        label="80% SoH Standard Failure Threshold (EOL)",
    )

    plt.title("Li-ion Battery State of Health (SoH) Degradation Over Aging Cycles", pad=12)
    plt.xlabel("Cycle Number")
    plt.ylabel("State of Health (%)")
    plt.xlim(0, 620)
    plt.ylim(55, 105)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")
    plt.tight_layout()

    out_file = IMAGES_DIR / "1_soh_degradation_curves.png"
    plt.savefig(out_file)
    plt.close()
    print(f"[SUCCESS] Saved: {out_file}")


def figure_2_predicted_vs_actual(df: pd.DataFrame, model: object, features: list[str]) -> None:
    """2. Predicted vs actual SoH (Scatter + Over-cycle progression) on held-out B0007."""
    test_df = df[df["battery_id"] == "B0007"].sort_values("cycle").reset_index(drop=True)
    y_actual = test_df["SoH"].values
    y_pred = model.predict(test_df[features].fillna(0.0))

    _, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=200)

    # Subplot A: Scatter
    axes[0].scatter(y_actual, y_pred, color="#2563EB", alpha=0.6, edgecolors="none", s=30, label="Predictions")
    axes[0].plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], "r--", lw=2, label="Perfect Fit (y = x)")
    axes[0].set_title("Actual vs. Predicted SoH (Scatter)")
    axes[0].set_xlabel("Actual SoH (%)")
    axes[0].set_ylabel("Predicted SoH (%)")
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Subplot B: Trajectory over observed cycles.
    axes[1].plot(test_df["cycle"], y_actual, label="Actual SoH", color="#111827", lw=2.2)
    axes[1].plot(test_df["cycle"], y_pred, label="XGBoost Predicted SoH", color="#2563EB", lw=1.8, linestyle="--")
    axes[1].axhline(y=80.0, color="#D97706", linestyle=":", lw=1.5, label="80% EOL Threshold")
    axes[1].set_title("Predicted vs. Actual Degradation Trajectory (B0007)")
    axes[1].set_xlabel("Cycle Number")
    axes[1].set_ylabel("State of Health (%)")
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    mae = mean_absolute_error(y_actual, y_pred)
    plt.suptitle(
        f"Battery B0007 Evaluation (Held-out Test Battery, MAE = {mae:.2f}% SoH)",
        y=1.02,
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    out_file = IMAGES_DIR / "2_predicted_vs_actual_soh.png"
    plt.savefig(out_file)
    plt.close()
    print(f"[SUCCESS] Saved: {out_file}")


def figure_3_residual_errors(df: pd.DataFrame, model: object, features: list[str]) -> None:
    """3. Residual error distribution plot."""
    test_df = df[df["battery_id"] == "B0007"].sort_values("cycle")
    y_actual = test_df["SoH"].values
    y_pred = model.predict(test_df[features].fillna(0.0))
    residuals = y_actual - y_pred

    _, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

    # Subplot A: Residuals vs. Cycle
    axes[0].scatter(test_df["cycle"], residuals, color="#4F46E5", alpha=0.6, s=25)
    axes[0].axhline(y=0.0, color="red", linestyle="--", lw=1.5)
    axes[0].set_title("Residuals vs. Aging Cycle (B0007)")
    axes[0].set_xlabel("Cycle Number")
    axes[0].set_ylabel("Residual Error (% SoH: Actual - Predicted)")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Subplot B: Residual Histogram
    axes[1].hist(residuals, bins=25, color="#6366F1", edgecolor="black", alpha=0.8, density=True)
    axes[1].axvline(x=0.0, color="red", linestyle="--", lw=1.5, label="Zero Error")
    axes[1].set_title("Residual Error Distribution (B0007)")
    axes[1].set_xlabel("Prediction Error (% SoH)")
    axes[1].set_ylabel("Density")
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_file = IMAGES_DIR / "3_residual_error_distribution.png"
    plt.savefig(out_file)
    plt.close()
    print(f"[SUCCESS] Saved: {out_file}")


def figure_4_feature_importance(model: object, features: list[str]) -> None:
    """4. XGBoost feature importance chart."""
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)

    clean_names = {
        "capacity": "Capacity (Ah)",
        "capacity_fade_rate": "Capacity Fade Rate (dC)",
        "temp_mean": "Mean Temperature (°C)",
        "voltage_mean": "Mean Voltage (V)",
        "current_mean": "Mean Current (A)",
        "internal_resistance_proxy": "Internal Resistance Proxy (Ω)",
        "dT_dt": "Thermal Rate (dT/dt)",
        "dV_dt": "Voltage Rate (dV/dt)",
    }
    feature_labels = [clean_names.get(features[i], features[i]) for i in sorted_idx]

    plt.figure(figsize=(9, 5), dpi=200)
    bars = plt.barh(range(len(sorted_idx)), importances[sorted_idx], color="#0284C7", edgecolor="#0369A1", height=0.6)
    plt.yticks(range(len(sorted_idx)), feature_labels)
    plt.xlabel("XGBoost Relative Feature Importance (Gain / Weight)")
    plt.title("XGBoost Model Feature Importance for State of Health Prediction")
    plt.grid(True, axis="x", linestyle="--", alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center", ha="left", fontsize=9)

    plt.xlim(0, max(importances) * 1.15)
    plt.tight_layout()

    out_file = IMAGES_DIR / "4_xgboost_feature_importance.png"
    plt.savefig(out_file)
    plt.close()
    print(f"[SUCCESS] Saved: {out_file}")


def figure_5_rul_estimation_example(df: pd.DataFrame) -> None:
    """5. Shows retrospective RUL from an observed battery lifetime."""
    test_df = df[df["battery_id"] == "B0007"].sort_values("cycle").reset_index(drop=True)

    checkpoint_cycle = 140
    checkpoint = test_df.loc[test_df["cycle"] <= checkpoint_cycle].iloc[-1]
    checkpoint_cycle = int(checkpoint["cycle"])
    checkpoint_soh = float(checkpoint["SoH"])
    eol = test_df.loc[test_df["SoH"] <= 80.0].iloc[0]
    eol_cycle = int(eol["cycle"])
    eol_soh = float(eol["SoH"])
    rul_cycles = eol_cycle - checkpoint_cycle

    plt.figure(figsize=(9.5, 5.5), dpi=200)
    plt.plot(test_df["cycle"], test_df["SoH"], label="Observed SoH", color="#2563EB", lw=2.2)
    plt.axhline(y=80.0, color="#DC2626", linestyle="--", lw=1.8, label="80% EOL Threshold")

    # Annotate Checkpoint
    plt.scatter([checkpoint_cycle], [checkpoint_soh], color="#1E40AF", s=80, zorder=5)
    plt.annotate(
        f"Operational Checkpoint\nCycle {checkpoint_cycle} (SoH: {checkpoint_soh:.1f}%)",
        xy=(checkpoint_cycle, checkpoint_soh),
        xytext=(checkpoint_cycle - 70, checkpoint_soh + 6),
        arrowprops=dict(facecolor="#1E40AF", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        fontsize=9,
    )

    # Annotate Failure Point
    plt.scatter([eol_cycle], [eol_soh], color="#DC2626", s=90, zorder=5)
    plt.annotate(
        f"Observed EOL\nCycle {eol_cycle} (SoH: {eol_soh:.1f}%)",
        xy=(eol_cycle, eol_soh),
        xytext=(eol_cycle + 20, eol_soh + 5),
        arrowprops=dict(facecolor="#DC2626", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        fontsize=9,
    )

    # Arrow representing RUL
    plt.annotate(
        "",
        xy=(eol_cycle, 77.5),
        xytext=(checkpoint_cycle, 77.5),
        arrowprops=dict(arrowstyle="<->", color="#059669", lw=2.5),
    )
    plt.text(
        (checkpoint_cycle + eol_cycle) / 2,
        75.2,
        f"Retrospective RUL = {rul_cycles} cycles",
        ha="center",
        color="#059669",
        fontweight="bold",
        fontsize=10,
    )

    plt.title("Retrospective RUL from an Observed Battery Lifetime", pad=12)
    plt.xlabel("Aging Cycle Number")
    plt.ylabel("State of Health (%)")
    plt.ylim(68, 102)
    plt.legend(frameon=True, loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_file = IMAGES_DIR / "5_rul_prediction_example.png"
    plt.savefig(out_file)
    plt.close()
    print(f"[SUCCESS] Saved: {out_file}")


def figure_6_model_comparison(df: pd.DataFrame) -> None:
    """6. Compares sensor-only models using leave-one-battery-out evaluation."""
    batteries = sorted(df["battery_id"].unique())
    lr_maes = []
    xgb_maes = []

    for b in batteries:
        results = train_and_evaluate(df, FEATURE_COLS, [b])
        lr_maes.append(results["baseline_mae"])
        xgb_maes.append(results["xgb_mae"])

    # Average
    lr_maes.append(float(np.mean(lr_maes)))
    xgb_maes.append(float(np.mean(xgb_maes)))
    categories = [f"Test: {battery}" for battery in batteries] + ["Average"]

    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(9, 5.5), dpi=200)
    bars_lr = plt.bar(x - width / 2, lr_maes, width, label="Baseline Linear Regression", color="#94A3B8", edgecolor="#475569")
    bars_xgb = plt.bar(x + width / 2, xgb_maes, width, label="XGBoost Regressor (Ours)", color="#2563EB", edgecolor="#1D4ED8")

    plt.ylabel("Mean Absolute Error (% SoH)")
    plt.title("Model Comparison on Sensor Telemetry: Baseline Linear Regression vs. XGBoost", pad=12)
    plt.xticks(x, categories, fontweight="bold")
    plt.legend(frameon=True)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar in bars_lr:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.15, f"{h:.2f}%", ha="center", va="bottom", fontsize=8.5)

    for bar in bars_xgb:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.15, f"{h:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1E40AF")

    plt.ylim(0, max(lr_maes) * 1.18)
    plt.tight_layout()

    out_file = IMAGES_DIR / "6_model_comparison_bar_chart.png"
    plt.savefig(out_file)
    plt.close()
    print(f"[SUCCESS] Saved: {out_file}")


def main() -> None:
    """Generates all Phase 3 figures from real data."""
    print("=" * 60)
    print("  GENERATING PUBLICATION-READY FIGURES FOR REPOSITORY")
    print("=" * 60)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    model = joblib.load(MODEL_PATH)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        features = json.load(f)

    figure_1_soh_degradation(df)
    figure_2_predicted_vs_actual(df, model, features)
    figure_3_residual_errors(df, model, features)
    figure_4_feature_importance(model, features)
    figure_5_rul_estimation_example(df)
    figure_6_model_comparison(df)

    print("\n[COMPLETE] All 6 figures generated and saved to docs/images/")


if __name__ == "__main__":
    main()
