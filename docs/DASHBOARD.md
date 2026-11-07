# Streamlit Dashboard Guide

The repository includes a **Streamlit** dashboard for inspecting historical battery health, model SoH predictions, and observed end-of-life status.

---

## 1. Launching the Dashboard

### Windows (One-Click)
Double-click `run_app.bat` or run in PowerShell:
```powershell
python -m streamlit run dashboard/app.py
```

### macOS / Linux
Execute the launch script:
```bash
./run_app.sh
```
Or directly:
```bash
python3 -m streamlit run dashboard/app.py
```

Open your web browser at **`http://localhost:8501`**.

---

## 2. Dashboard Interface Tour

### View 1: Health Status Tab
![Health Status View](images/dashboard_view_health_status.png)
*Figure 1: The primary monitoring view displays Current SoH (%), latest cycle number, total lifetime cycles, and average cell temperature with clear visual status banners (e.g. "Battery in EXCELLENT condition").*

### View 2: State of Health (SoH) Trend Tab
![SoH Trend View](images/dashboard_view_soh_trend.png)
*Figure 2: Interactive timeseries plot tracing the complete capacity fade curve across all recorded cycles, alongside minimum, maximum, and average historical SoH metrics.*

### View 3: Model Predictions & RUL Estimation Tab
![Predictions and RUL View](images/dashboard_view_predictions_rul.png)
*Figure 3: Overlay comparing measured and predicted SoH on the available profile. The dashboard reports observed EOL and zero RUL after the measured profile crosses 80%; it does not extrapolate a future threshold crossing.*

### View 4: Critical / Warning State Alert
![Warning Alert View](images/dashboard_view_warning_state.png)
*Figure 4: The dashboard flags poor health for B0006, whose latest measured SoH is below the configured status threshold.*

---

## 3. Sidebar Controls
* **Battery Selector Dropdown:** Switch dynamically between batteries (`B0005`, `B0006`, `B0007`, `B0018`) to inspect telemetry and predictions for each unit independently.
* **Telemetry Overview:** Instant sidebar readout of selected battery metadata.

## 4. RUL Scope

The supplied data contains completed historical battery profiles. For profiles still above the 80% threshold, the dashboard reports that future RUL is unavailable rather than treating the last observed cycle as a predicted failure. The SoH model does not extrapolate beyond recorded cycles.
