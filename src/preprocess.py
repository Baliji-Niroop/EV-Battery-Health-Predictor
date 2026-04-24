from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import tempfile
import zipfile

import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_mat_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load a MATLAB .mat file safely."""
    try:
        return loadmat(file_path, squeeze_me=True, struct_as_record=False)
    except Exception as exc:
        print(f"[ERROR] Failed to load {file_path.name}: {exc}")
        return None


def safe_get(obj: Any, field: str) -> Any:
    """Safely get a field from dict, object, or structured array."""
    try:
        if obj is None:
            return None
        if isinstance(obj, dict) and field in obj:
            return obj[field]
        if hasattr(obj, field):
            return getattr(obj, field)
        if isinstance(obj, np.ndarray) and obj.dtype.names and field in obj.dtype.names:
            return obj[field]
    except Exception:
        return None
    return None


def to_array(values: Any) -> np.ndarray:
    """Convert values to a clean 1D float array."""
    if values is None:
        return np.array([], dtype=float)
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
        return arr[~np.isnan(arr)]
    except Exception:
        return np.array([], dtype=float)


def safe_mean(values: Any) -> float:
    """Compute a safe mean for a numeric array-like input."""
    arr = to_array(values)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def safe_scalar(values: Any) -> float:
    """Extract a scalar value from array-like input."""
    arr = to_array(values)
    if arr.size:
        return float(arr[-1])
    try:
        return float(values)
    except Exception:
        return float("nan")


def find_cycle_container(mat: Dict[str, Any]) -> Any:
    """Find the correct battery structure inside .mat file."""
    for key in mat:
        if key.startswith("__"):
            continue

        obj = mat[key]

        if hasattr(obj, "cycle"):
            return obj

        if isinstance(obj, np.ndarray):
            try:
                obj = obj.item()
                if hasattr(obj, "cycle"):
                    return obj
            except Exception:
                continue

    return None


def extract_cycles(container: Any) -> List[Any]:
    """Extract cycles from a container."""
    if container is None:
        return []
    if hasattr(container, "cycle"):
        cycles = container.cycle
    elif isinstance(container, np.ndarray) and container.dtype.names and "cycle" in container.dtype.names:
        cycles = container["cycle"]
    else:
        return []

    if isinstance(cycles, np.ndarray):
        return list(cycles.flatten())
    if isinstance(cycles, (list, tuple)):
        return list(cycles)
    return [cycles]


def parse_cycle(cycle: Any, idx: int, battery_id: str) -> Dict[str, Any]:
    """Parse a single cycle into a flat record."""
    data = safe_get(cycle, "data")

    cycle_num = idx

    voltage = safe_get(data, "Voltage_measured")
    if voltage is None:
        voltage = safe_get(data, "voltage")
    
    current = safe_get(data, "Current_measured")
    if current is None:
        current = safe_get(data, "current")
    
    temp = safe_get(data, "Temperature_measured")
    if temp is None:
        temp = safe_get(data, "temperature")
    
    capacity = safe_get(data, "Capacity")
    if capacity is None:
        capacity = safe_get(data, "capacity")

    return {
        "battery_id": battery_id,
        "cycle": cycle_num,
        "capacity": safe_scalar(capacity),
        "voltage_mean": safe_mean(voltage),
        "current_mean": safe_mean(current),
        "temp_mean": safe_mean(temp),
    }


def process_file(file_path: Path) -> pd.DataFrame:
    """Process a single .mat file into a dataframe."""
    mat = load_mat_file(file_path)
    if mat is None:
        return pd.DataFrame()

    print(f"[DEBUG] Keys in mat file: {list(mat.keys())}")

    container = find_cycle_container(mat)
    cycles = extract_cycles(container)
    records: List[Dict[str, Any]] = []

    print(f"[DEBUG] Found {len(cycles)} cycles in {file_path.name}")

    for idx, cycle in enumerate(cycles, start=1):
        try:
            records.append(parse_cycle(cycle, idx, file_path.stem))
        except Exception as exc:
            print(f"[WARN] Skipping cycle {idx} in {file_path.name}: {exc}")

    return pd.DataFrame(records)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and compute SoH for the combined dataframe."""
    if df.empty:
        return df

    # Replace deprecated fillna(method=) with modern chaining
    df = df.ffill().bfill().fillna(0)
    df = df.dropna(how="all", subset=["capacity", "voltage_mean", "current_mean", "temp_mean"])

    # Compute SoH per battery (capacity normalized by max per battery)
    df["SoH"] = df.groupby("battery_id")["capacity"].transform(
        lambda x: (x / x.max()) * 100.0
    )


    return df


def main() -> None:
    """Run preprocessing and save cleaned data."""
    np.random.seed(42)
    print(f"[INFO] Random state fixed at 42 for reproducibility")
    root = Path(__file__).resolve().parents[1]
    
    data_dir = root / "data"
    raw_dir = data_dir / "raw"
    output_path = data_dir / "processed" / "cleaned_data.csv"
    
    # Create new structure if it doesn't exist
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"[ERROR] Raw data directory not found: {raw_dir}")
        return

    print(f"Looking for data in: {raw_dir}")
    
    mat_files = sorted(raw_dir.glob("*.mat"))
    print(f"Found {len(mat_files)} .mat files directly")
    
    if not mat_files:
        zip_files = sorted(raw_dir.glob("*.zip"))
        print(f"Found {len(zip_files)} .zip files")
        
        if not zip_files:
            print(f"Error: No .mat or .zip files found in {raw_dir}")
            print("To run this project, you need the NASA Battery Dataset.")
            print("1. Download the dataset from: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
            print("2. Extract the files and put the .mat files into the data/raw/ folder.")
            print("3. Run this script again.")
            import sys
            sys.exit(1)
        
        if zip_files:
            # Create a temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                for zip_file in zip_files:
                    print(f"[INFO] Extracting {zip_file.name}")
                    try:
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            zip_ref.extractall(temp_path)
                        print(f"[INFO] Extracted {zip_file.name} successfully")
                    except Exception as exc:
                        print(f"[ERROR] Failed to extract {zip_file.name}: {exc}")
                
                # Find all .mat files in the extracted directory
                mat_files = sorted(temp_path.rglob("*.mat"))
                print(f"[INFO] Found {len(mat_files)} .mat files in extracted archives")
                
                # Process extracted files
                all_frames: List[pd.DataFrame] = []
                wanted_batteries = ["B0005", "B0006", "B0007", "B0018"]
                for mat_file in mat_files:
                    if mat_file.stem not in wanted_batteries:
                        print(f"[INFO] Skipping {mat_file.name} (not in wanted list)")
                        continue
                    
                    print(f"[INFO] Processing {mat_file.name}")
                    df_file = process_file(mat_file)
                    if not df_file.empty:
                        print(f"[DEBUG] Added {len(df_file)} cycles from {mat_file.name}")
                        all_frames.append(df_file)
                
                # Combine and save
                if all_frames:
                    df = pd.concat(all_frames, ignore_index=True)
                    df = clean_dataframe(df)
                    
                    print(f"\n[INFO] ===== PREPROCESSING COMPLETE =====")
                    print(f"[INFO] Total cycles extracted: {len(df)}")
                    print(f"[INFO] Columns: {list(df.columns)}")
                    if 'SoH' in df.columns:
                        print(f"[INFO] SoH range: {df['SoH'].min():.2f}% to {df['SoH'].max():.2f}%")
                    print(f"\n[INFO] Sample rows (first 5):")
                    print(df.head())
                    
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        df.to_csv(output_path, index=False)
                        print(f"\n[INFO] ✓ Saved cleaned data to: {output_path}")
                    except Exception as exc:
                        print(f"[ERROR] Failed to save CSV: {exc}")
                else:
                    print("Error: No cycles extracted from .mat files.")
                    import sys
                    sys.exit(1)
    else:
        # Process .mat files directly
        all_frames: List[pd.DataFrame] = []
        wanted_batteries = ["B0005", "B0006", "B0007", "B0018"]
        for mat_file in mat_files:
            if mat_file.stem not in wanted_batteries:
                print(f"[INFO] Skipping {mat_file.name} (not in wanted list)")
                continue
                
            print(f"[INFO] Processing {mat_file.name}")
            df_file = process_file(mat_file)
            if not df_file.empty:
                print(f"[DEBUG] Added {len(df_file)} cycles from {mat_file.name}")
                all_frames.append(df_file)
        
        if all_frames:
            df = pd.concat(all_frames, ignore_index=True)
            df = clean_dataframe(df)
            
            print(f"\n[INFO] ===== PREPROCESSING COMPLETE =====")
            print(f"[INFO] Total cycles extracted: {len(df)}")
            print(f"[INFO] Columns: {list(df.columns)}")
            if 'SoH' in df.columns:
                print(f"[INFO] SoH range: {df['SoH'].min():.2f}% to {df['SoH'].max():.2f}%")
            print(f"\n[INFO] Sample rows (first 5):")
            print(df.head())
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                df.to_csv(output_path, index=False)
                print(f"\n[INFO] ✓ Saved cleaned data to: {output_path}")
            except Exception as exc:
                print(f"[ERROR] Failed to save CSV: {exc}")
        else:
            print("[ERROR] No cycles extracted from .mat files.")


if __name__ == "__main__":
    main()
