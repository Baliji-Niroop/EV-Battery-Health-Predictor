from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import tempfile
import zipfile

import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_mat_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Safely loads a MATLAB .mat file.

    Parameters
    ----------
    file_path : Path
        Path to the .mat file.

    Returns
    -------
    Optional[Dict[str, Any]]
        Parsed dictionary of MATLAB structures, or None if reading fails.
    """
    try:
        return loadmat(file_path, squeeze_me=True, struct_as_record=False)
    except Exception as exc:
        print(f"[ERROR] Failed to load {file_path.name}: {exc}")
        return None


def safe_get(obj: Any, field: str) -> Any:
    """Safely extracts an attribute or dict key from nested structures.

    Parameters
    ----------
    obj : Any
        Object, dictionary, or numpy structured array to inspect.
    field : str
        Target attribute or dictionary key name.

    Returns
    -------
    Any
        Extracted value if present, otherwise None.
    """
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
    """Converts input sequence to a clean 1D float array, stripping NaNs.

    Parameters
    ----------
    values : Any
        Input array or list.

    Returns
    -------
    np.ndarray
        Clean 1D float array.
    """
    if values is None:
        return np.array([], dtype=float)
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
        return arr[~np.isnan(arr)]
    except Exception:
        return np.array([], dtype=float)


def safe_mean(values: Any) -> float:
    """Computes the arithmetic mean of an array safely, ignoring NaNs.

    Parameters
    ----------
    values : Any
        Input array or numeric sequence.

    Returns
    -------
    float
        Computed mean, or nan if array is empty.
    """
    arr = to_array(values)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def safe_scalar(values: Any) -> float:
    """Extracts a scalar number from a single value or the final array element.

    Parameters
    ----------
    values : Any
        Input array, scalar, or sequence.

    Returns
    -------
    float
        Extracted scalar float, or nan on failure.
    """
    arr = to_array(values)
    if arr.size:
        return float(arr[-1])
    try:
        return float(values)
    except Exception:
        return float("nan")


def find_cycle_container(mat: Dict[str, Any]) -> Any:
    """Locates the battery cycle container inside a parsed .mat dictionary.

    Parameters
    ----------
    mat : Dict[str, Any]
        Dictionary of variables loaded from the .mat file.

    Returns
    -------
    Any
        The container holding cycle records, or None if not found.
    """
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
    """Extracts the list of cycle objects from the battery container.

    Parameters
    ----------
    container : Any
        The battery struct or record.

    Returns
    -------
    List[Any]
        A flat list of cycle records.
    """
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
    """Parses a single battery cycle into tabular summary metrics.

    Parameters
    ----------
    cycle : Any
        Raw cycle object containing operational measurements.
    idx : int
        Sequential cycle index.
    battery_id : str
        Battery identifier (e.g. 'B0005').

    Returns
    -------
    Dict[str, Any]
        Dictionary with cycle-level metrics (voltage, current, temp, capacity).
    """
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
    """Processes a single NASA .mat file into a cycle-level DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the battery .mat file.

    Returns
    -------
    pd.DataFrame
        DataFrame of parsed cycle records.
    """
    mat = load_mat_file(file_path)
    if mat is None:
        return pd.DataFrame()

    container = find_cycle_container(mat)
    cycles = extract_cycles(container)
    records: List[Dict[str, Any]] = []

    for idx, cycle in enumerate(cycles, start=1):
        try:
            records.append(parse_cycle(cycle, idx, file_path.stem))
        except Exception as exc:
            print(f"[WARN] Skipping cycle {idx} in {file_path.name}: {exc}")

    return pd.DataFrame(records)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans parsed cycle records and calculates State of Health (SoH).

    SoH is calculated as the ratio of current measured capacity to nominal
    maximum capacity of that cell, scaled to a percentage (0-100%).

    Parameters
    ----------
    df : pd.DataFrame
        Raw cycle records.

    Returns
    -------
    pd.DataFrame
        Cleaned records with computed SoH column.
    """
    if df.empty:
        return df

    # Fill missing values using modern forward-then-backward filling
    df = df.ffill().bfill().fillna(0.0)
    df = df.dropna(how="all", subset=["capacity", "voltage_mean", "current_mean", "temp_mean"])

    # Compute SoH per battery: (capacity / max_capacity) * 100%
    df["SoH"] = df.groupby("battery_id")["capacity"].transform(
        lambda x: (x / x.max()) * 100.0 if x.max() > 0 else np.nan
    )

    return df


def main() -> None:
    """Main execution function for raw data preprocessing."""
    np.random.seed(42)
    root = Path(__file__).resolve().parents[1]

    data_dir = root / "data"
    raw_dir = data_dir / "raw"
    output_path = data_dir / "processed" / "cleaned_data.csv"

    (data_dir / "raw").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Scanning raw data directory: {raw_dir}")

    mat_files = sorted(raw_dir.glob("*.mat"))

    if not mat_files:
        zip_files = sorted(raw_dir.glob("*.zip"))

        if not zip_files:
            print(f"\n[ERROR] No .mat or .zip files found in {raw_dir}")
            print("\n" + "=" * 65)
            print("  DATASET DOWNLOAD INSTRUCTIONS")
            print("=" * 65)
            print("To run raw preprocessing, you need the NASA Li-ion Battery Dataset:")
            print("1. Download '5. Battery Data Set.zip' from NASA Prognostics Center:")
            print("   https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/")
            print("2. Extract the archive and place B0005.mat, B0006.mat, B0007.mat into:")
            print(f"   {raw_dir}")
            print("3. Re-run: python run_pipeline.py\n")
            print("Note: If data/processed/features.csv is already present, the pipeline")
            print("will use the preprocessed features directly.")
            print("=" * 65 + "\n")
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for zip_file in zip_files:
                print(f"[INFO] Extracting {zip_file.name}")
                try:
                    with zipfile.ZipFile(zip_file, "r") as zip_ref:
                        zip_ref.extractall(temp_path)
                except Exception as exc:
                    print(f"[ERROR] Failed to extract {zip_file.name}: {exc}")

            mat_files = sorted(temp_path.rglob("*.mat"))

    all_frames: List[pd.DataFrame] = []
    wanted_batteries = ["B0005", "B0006", "B0007", "B0018"]

    for mat_file in mat_files:
        if mat_file.stem not in wanted_batteries:
            continue
        print(f"[INFO] Parsing cycle telemetry from {mat_file.name}")
        df_file = process_file(mat_file)
        if not df_file.empty:
            all_frames.append(df_file)

    if all_frames:
        df = pd.concat(all_frames, ignore_index=True)
        df = clean_dataframe(df)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[SUCCESS] Cleaned data saved: {output_path} ({len(df)} rows)")
    else:
        print("[WARN] No valid battery cycles could be parsed from the files provided.")


if __name__ == "__main__":
    main()
