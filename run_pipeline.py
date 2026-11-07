from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def print_step_header(step_num: int, total_steps: int, title: str) -> None:
    """Prints a clear, formatted step header."""
    print(
        f"\n{'=' * 65}\n  Step {step_num}/{total_steps}: {title}\n{'=' * 65}",
        flush=True,
    )


def run_subcommand(script_name: str, step_num: int, total_steps: int, title: str) -> bool:
    """Executes a pipeline step script and reports status."""
    script_path = SRC_DIR / script_name
    print_step_header(step_num, total_steps, title)

    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return False

    result = subprocess.run(
        [sys.executable, str(script_path)],
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"[ERROR] {title} failed with exit code {result.returncode}")
        return False
    return True


def main() -> None:
    """Executes the end-to-end EV Battery Health & RUL ML Pipeline."""
    print("\n" + "#" * 65, flush=True)
    print("  EV BATTERY HEALTH & RUL PREDICTION PIPELINE")
    print("  Author: Baliji Niroop | Machine Learning Pipeline")
    print("#" * 65)

    raw_dir = DATA_DIR / "raw"
    features_path = DATA_DIR / "processed" / "features.csv"

    # Step 1: Preprocessing raw NASA .mat files
    raw_files = list(raw_dir.glob("*.mat")) + list(raw_dir.glob("*.zip")) if raw_dir.exists() else []

    if raw_files:
        if not run_subcommand("preprocess.py", 1, 4, "Preprocessing Raw NASA Data"):
            print("[FATAL] Pipeline stopped at Step 1.")
            sys.exit(1)
    else:
        print_step_header(1, 4, "Preprocessing Raw NASA Data")
        if features_path.exists():
            print("[INFO] No raw .mat files found in data/raw/.")
            print("[INFO] Found existing preprocessed features at data/processed/features.csv.")
            print("[INFO] Proceeding with existing features.")
        else:
            print("[ERROR] Neither raw dataset (.mat files) nor features.csv were found.")
            print("Please see docs/DATASET.md for instructions on downloading the dataset.")
            sys.exit(1)

    # Step 2: Feature Engineering
    if not run_subcommand("features.py", 2, 4, "Feature Engineering"):
        print("[FATAL] Pipeline stopped at Step 2.")
        sys.exit(1)

    # Step 3: Model Training (XGBoost & Baseline Linear Regression)
    if not run_subcommand("soh_model.py", 3, 4, "Training XGBoost & Evaluating Models"):
        print("[FATAL] Pipeline stopped at Step 3.")
        sys.exit(1)

    # Step 4: RUL Estimation
    if not run_subcommand("rul_estimator.py", 4, 4, "Estimating Remaining Useful Life (RUL)"):
        print("[FATAL] Pipeline stopped at Step 4.")
        sys.exit(1)

    # Summary
    print("\n" + "=" * 65)
    print("  PIPELINE EXECUTION COMPLETE (4/4 Steps Successful)")
    print("=" * 65)
    print("Trained model saved to:  models/xgb_model.pkl")
    print("Metrics exported to:     outputs/metrics.txt")
    print("Scatter plot saved to:   outputs/soh_prediction_scatter.png")
    print("\nTo launch the interactive dashboard, run:")
    print("  streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    main()
