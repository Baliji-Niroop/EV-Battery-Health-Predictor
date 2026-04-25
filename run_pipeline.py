from __future__ import annotations

import subprocess
import sys
from pathlib import Path

print("EV Battery Health Pipeline v1.0")


def print_header(step_name: str) -> None:
    """Print formatted header for each step."""
    print("\n" + "=" * 60)
    print(f"  {step_name}")
    print("=" * 60)


def check_file_exists(file_path: Path) -> bool:
    """Check if expected output file exists."""
    return file_path.exists()


def run_script(script_path: Path, expected_output: Path, step_name: str) -> bool:
    """Run a Python script and verify output file was created."""
    print_header(step_name)
    
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print(f"[ERROR] Script failed with return code {result.returncode}")
            return False
        
        # Wait and check for output file
        if expected_output.exists():
            print(f"[SUCCESS] {step_name} completed")
            print(f"[INFO] Output file: {expected_output}")
            return True
        else:
            print(f"[ERROR] Expected output file not found: {expected_output}")
            print(f"[ERROR] {step_name} may have failed or had no data to process")
            return False
            
    except Exception as exc:
        print(f"[ERROR] Failed to run {step_name}: {exc}")
        return False


def get_data_dir() -> Path:
    """Get the data directory."""
    root = Path(__file__).resolve().parents[0]
    return root / "data"


def get_processed_dir() -> Path:
    """Get the processed data directory."""
    data_dir = get_data_dir()
    processed_dir = data_dir / "processed"
    return processed_dir


def get_models_dir() -> Path:
    """Get the models directory."""
    root = Path(__file__).resolve().parents[0]
    return root / "models"


def main() -> None:
    """Run the entire ML pipeline."""
    root = Path(__file__).resolve().parents[0]
    src_dir = root / "src"
    
    print("\n" + "=" * 60)
    print("  EV BATTERY HEALTH - ML PIPELINE AUTOMATION")
    print("=" * 60)
    print(f"Project root: {root}")
    
    # Step 1: Preprocessing
    preprocess_script = src_dir / "preprocess.py"
    cleaned_data_path = get_processed_dir() / "cleaned_data.csv"
    raw_dir = get_data_dir() / "raw"
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        print("No raw data found, using existing processed data")
    else:
        if not run_script(preprocess_script, cleaned_data_path, "STEP 1: PREPROCESSING"):
            print("\n[FATAL] Pipeline aborted at preprocessing stage")
            sys.exit(1)
    
    # Step 2: Feature Engineering
    features_script = src_dir / "features.py"
    features_path = get_processed_dir() / "features.csv"
    if not cleaned_data_path.exists() and features_path.exists():
        print("Cleaned data not found, using existing features data")
    else:
        if not run_script(features_script, features_path, "STEP 2: FEATURE ENGINEERING"):
            print("\n[FATAL] Pipeline aborted at feature engineering stage")
            sys.exit(1)
            
    # Ensure processed features exist before proceeding
    if not features_path.exists():
        raise FileNotFoundError("No processed data found. Please run preprocessing with dataset.")
    
    # Step 3: Model Training
    model_script = src_dir / "soh_model.py"
    model_path = get_models_dir() / "xgb_model.pkl"
    
    if not run_script(model_script, model_path, "STEP 3: MODEL TRAINING"):
        print("\n[FATAL] Pipeline aborted at model training stage")
        sys.exit(1)
    
    # Step 4: RUL Estimation
    rul_script = src_dir / "rul_estimator.py"
    
    print_header("STEP 4: RUL ESTIMATION")
    
    if not rul_script.exists():
        print(f"[ERROR] Script not found: {rul_script}")
        print("\n[FATAL] Pipeline aborted at RUL estimation stage")
        sys.exit(1)
    
    try:
        result = subprocess.run(
            [sys.executable, str(rul_script)],
            capture_output=False,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print(f"[ERROR] Script failed with return code {result.returncode}")
            print("\n[FATAL] Pipeline aborted at RUL estimation stage")
            sys.exit(1)
        
        print("[SUCCESS] RUL estimation completed")
        
    except Exception as exc:
        print(f"[ERROR] Failed to run RUL estimation: {exc}")
        print("\n[FATAL] Pipeline aborted at RUL estimation stage")
        sys.exit(1)
    
    # All steps completed
    print("\nAll steps completed successfully.")
    print("Model trained, outputs generated.")
    print("Run dashboard using:")
    print("streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    main()
