# Project Guide

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run `python -m pytest -q` to verify the checked-in model and data.
4. Run `python run_pipeline.py` to refresh features, train the SoH model, and report end-of-life status.
5. Launch the dashboard with `python -m streamlit run dashboard/app.py`.

The supplied processed dataset makes the project runnable without downloading the raw NASA files. To regenerate it from source, place the supported NASA `.mat` files in `data/raw/` and run the pipeline.

To regenerate browser screenshots, install `requirements-dev.txt`, install Playwright's Chromium browser with `python -m playwright install chromium`, start Streamlit on port 8501, and run `python scripts/capture_screenshots.py`.

## Pipeline

| Stage | Module | Output |
|---|---|---|
| Parse valid discharge telemetry and calculate SoH against the first measured capacity | `src/preprocess.py` | `data/processed/cleaned_data.csv` |
| Recompute per-battery features and SoH labels | `src/features.py` | `data/processed/features.csv` |
| Train sensor-only XGBoost and compare with linear regression on a held-out battery | `src/soh_model.py` | `models/`, `outputs/` |
| Report the first observed cycle at or below 80% SoH | `src/rul_estimator.py` | Console summary |
| View measured and predicted SoH for each battery | `dashboard/app.py` | Streamlit dashboard |

## Model and RUL scope

The model uses voltage, current, temperature, their cycle-to-cycle changes, and a voltage/current resistance proxy. Capacity and capacity-derived inputs are excluded because they directly encode the SoH target.

The current implementation identifies end-of-life in the recorded profile. If the latest measured SoH remains above 80%, it reports future RUL as unavailable; it does not extrapolate a future failure cycle. The 80% threshold is a commonly used convention, not a universal safety guarantee. See [the methodology](./METHODOLOGY.md), [results](./RESULTS.md), and [limitations](./LIMITATIONS_AND_FUTURE_WORK.md) for details.
