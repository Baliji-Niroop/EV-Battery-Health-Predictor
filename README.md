# EV Battery Health & Remaining Useful Life (RUL) Predictor

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EB5424.svg)](https://xgboost.readthedocs.io/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Clean%20%26%20Documented-059669.svg)](#)

> Machine learning pipeline predicting electric vehicle lithium-ion battery degradation (State of Health) and Remaining Useful Life (RUL) using XGBoost on the NASA Battery Aging dataset, featuring an interactive Streamlit monitoring dashboard.

---

## 📸 Dashboard Preview

![EV Battery Health Dashboard](docs/images/dashboard_view_health_status.png)
*Interactive Streamlit dashboard tracking real-time State of Health (SoH), operating temperatures, and battery condition alerts.*

---

## ⚡ What is this?

This is an end-to-end engineering project that predicts how fast lithium-ion electric vehicle batteries degrade over hundreds of charge-discharge cycles. By learning from real electrochemical laboratory telemetry published by NASA, the system forecasts a battery's **State of Health (SoH)** and estimates its **Remaining Useful Life (RUL)** before it reaches the standard 80% end-of-life threshold.

### 🚗 Why it matters:
* **EV Safety:** Lithium-ion cells that operate beyond their safe operational lifespan risk thermal runaway, internal short circuits, and sudden roadside shutdowns.
* **Cost Savings:** Battery packs represent 30% to 40% of the total cost of an electric car. Accurate health prediction prevents premature battery replacements.
* **Second-Life Utilization:** Helps energy storage providers know exactly when retired EV batteries can safely be repurposed for stationary solar and grid storage.

---

## 📊 Key Results at a Glance

| Benchmark Metric | XGBoost (Ours) | Baseline Linear Regression | Evaluation Notes |
| :--- | :--- | :--- | :--- |
| **Primary Test MAE (Unseen Cell B0007)** | **0.7951% SoH** | 0.0000% SoH (degenerated) | Battery-wise cross validation |
| **Primary Test RMSE (Unseen Cell B0007)**| **1.0511% SoH** | 0.0000% SoH (degenerated) | Battery-wise cross validation |
| **Sensor-Only Features MAE (All Cells)** | **3.34% SoH** | 8.01% SoH | **58.3% error reduction** without direct capacity |
| **Tested Batteries** | B0005, B0006, B0007, B0018 | NASA Ames PCoE 18650 run-to-failure cycles |

*(Units note: All MAE and RMSE values are expressed in percentage points of State of Health: `% SoH`).*

---

## 📚 Table of Contents
* [Key Definitions](#-key-definitions-in-one-line)
* [Features](#-features)
* [How It Works](#-how-it-works)
* [Project Structure](#-project-structure)
* [Dataset](#-dataset)
* [Quick Start](#-quick-start)
* [Dashboard Views](#-dashboard-views)
* [Evaluation & Results](#-evaluation--results)
* [Engineering Design Decisions](#-engineering-design-decisions)
* [Limitations & Future Roadmap](#-limitations--future-roadmap)
* [What I Learned](#-what-i-learned)
* [Tech Stack](#-tech-stack)
* [Acknowledgements & Citation](#-acknowledgements--citation)
* [License & Author](#-license--author)

---

## 💡 Key Definitions (in plain English)

* **State of Health (SoH):** The percentage of charge storage capacity currently left in the battery compared to when it was brand new ($100\% \to 80\% \to \text{dead}$).
* **Remaining Useful Life (RUL):** The estimated number of charge-discharge cycles left before the battery hits the safe retirement limit.
* **Capacity Fade:** The gradual, irreversible drop in a battery's maximum energy storage capability over repeated use.
* **Rolling Average:** A moving average calculated across the past 5 cycles to filter out temporary electrical noise and sensor fluctuations.
* **XGBoost:** An optimized machine learning algorithm that builds a sequence of smart decision trees, with each tree correcting the errors of the previous ones.

---

## ✨ Features

- [x] **Safe MATLAB Parsing:** Automated extraction of nested telemetry (`.mat`) structures into clean tabular records.
- [x] **Engineered Telemetry:** Computes cycle-over-cycle voltage rate ($\frac{dV}{dt}$), thermal shifts ($\frac{dT}{dt}$), and internal cell resistance proxy ($V/I$).
- [x] **Leakage-Free Validation:** Strict battery-wise evaluation (training on B0005/B0006/B0018, testing on completely unseen B0007).
- [x] **Interactive Streamlit Web App:** Real-time health metrics, degradation line charts, RUL predictions, and critical condition alerts.
- [x] **Automated Pipeline Runner:** Step-by-step terminal execution with clean progress tracking and smoke tests.

---

## 🛠️ How It Works

```mermaid
flowchart TD
    subgraph S1["Step 1: Data Ingestion & Preprocessing"]
        A["Raw NASA Telemetry<br/>(.mat files: Voltage, Current, Temp, Capacity)"] --> B["Cleaned Cycle Telemetry<br/>(Calculates nominal SoH per cycle)"]
    end

    subgraph S2["Step 2: Feature Engineering"]
        B --> C["Engineered Features<br/>(dV/dt, dT/dt, Resistance proxy, 5-cycle rolling means)"]
    end

    subgraph S3["Step 3: Machine Learning Model"]
        C --> D["XGBoost Regressor<br/>(Trained on B0005, B0006, B0018)"]
        D --> E["Model Persistence & Metrics<br/>(models/xgb_model.pkl & outputs/metrics.txt)"]
    end

    subgraph S4["Step 4: Inference & Presentation"]
        E --> F["RUL Estimation Engine<br/>(Scans curve for 80% EOL crossing)"]
        F --> G["Streamlit Web Dashboard<br/>(Interactive charts, telemetry & warnings)"]
    end
```

1. **Step 1: Read Telemetry:** Parses nested MATLAB `.mat` files into structured cycle metrics (voltage, current, temperature, discharge capacity).
2. **Step 2: Feature Engineering:** Calculates rate of change ($\frac{dV}{dt}$, $\frac{dT}{dt}$), internal resistance proxy, and 5-cycle moving averages.
3. **Step 3: Train XGBoost:** Learns non-linear electrochemical degradation patterns on training batteries and evaluates on a held-out battery.
4. **Step 4: Interactive Dashboard:** Visualizes battery health trends and computes remaining useful life cycles in an intuitive web dashboard.

---

## 📂 Project Structure

```text
EV-BATTERY-HEALTH/
│
├── dashboard/
│   └── app.py                      # Interactive Streamlit monitoring dashboard
│
├── data/
│   ├── raw/                        # NASA .mat files (downloaded by user or extracted)
│   └── processed/
│       ├── cleaned_data.csv        # Preprocessed cycle-level telemetry
│       └── features.csv            # Final engineered feature table
│
├── docs/
│   ├── ARCHITECTURE.md             # In-depth system module architecture & data flow
│   ├── DATASET.md                  # NASA experimental cycling protocol & citation
│   ├── METHODOLOGY.md              # Feature engineering & ML design rationale
│   ├── RESULTS.md                  # Comprehensive benchmarks, LOBO tables & discussion
│   ├── DASHBOARD.md                # Dashboard usage walkthrough & screenshot gallery
│   ├── LIMITATIONS_AND_FUTURE_WORK.md # Operational boundaries & technical roadmap
│   └── images/                     # Real figures and authentic UI screenshots
│
├── models/
│   ├── feature_schema.json         # Exact list of ordered features used by model
│   └── xgb_model.pkl               # Serialized trained XGBoost model artifact
│
├── outputs/
│   ├── metrics.txt                 # Model benchmark evaluation metrics
│   └── soh_prediction_scatter.png  # Prediction scatter plot
│
├── scripts/
│   ├── capture_screenshots.py      # Automated browser script to capture dashboard UI
│   └── generate_figures.py         # Generates all publication-quality evaluation charts
│
├── src/
│   ├── features.py                 # Feature engineering functions & calculations
│   ├── preprocess.py               # Robust MATLAB .mat parsing logic
│   ├── rul_estimator.py            # 80% SoH threshold crossing & RUL estimation
│   └── soh_model.py                # Model training, baseline evaluation & validation
│
├── tests/
│   └── test_pipeline.py            # Automated smoke test checking model and inference
│
├── .gitignore                      # Clean exclusion of temporary & cache files
├── LICENSE                         # Permissive MIT License (Baliji Niroop)
├── README.md                       # Main project documentation & quickstart
├── requirements.txt                # Pinned and verified Python dependencies
├── run_app.bat                     # One-click Windows launch script
├── run_app.sh                      # One-click macOS / Linux launch script
└── run_pipeline.py                 # Master step-by-step ML pipeline runner
```

---

## 📦 Dataset

The project uses the **NASA Prognostics Center of Excellence (PCoE) Li-ion Battery Aging Dataset** (18650 cells cycled to failure under repeated charging, discharging, and impedance sweeps).

* **Cells analyzed:** B0005, B0006, B0007 (room temperature 24°C) and B0018 (accelerated aging).
* **Already included:** To save you time, `data/processed/features.csv` is already included in this repository so you can immediately train models and run the dashboard!
* **Raw files:** If you wish to re-parse the original ~150 MB raw `.mat` files from scratch, see [docs/DATASET.md](docs/DATASET.md) for direct download links and directory setup instructions.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Baliji-Niroop/EV-Battery-Health-Predictor.git
cd EV-Battery-Health-Predictor
```

### 2. Set up virtual environment & install requirements
```bash
# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the automated ML pipeline
Runs preprocessing, feature engineering, model training, and RUL estimation in 4 clean steps:
```bash
python run_pipeline.py
```

### 4. Run the smoke test
```bash
python tests/test_pipeline.py
```

### 5. Launch the Streamlit dashboard
```bash
# Windows
run_app.bat
# or directly:
python -m streamlit run dashboard/app.py

# macOS / Linux
./run_app.sh
# or directly:
python3 -m streamlit run dashboard/app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 🖥️ Dashboard Views

| View 1: Real-Time Health Status | View 2: State of Health Degradation Trend |
| :---: | :---: |
| ![Health Status](docs/images/dashboard_view_health_status.png) | ![SoH Trend](docs/images/dashboard_view_soh_trend.png) |
| *Status metrics, cycle count, and condition badge* | *Complete historical capacity fade curve* |

| View 3: XGBoost Predictions & RUL | View 4: Critical Battery Alert |
| :---: | :---: |
| ![Predictions & RUL](docs/images/dashboard_view_predictions_rul.png) | ![Warning State](docs/images/dashboard_view_warning_state.png) |
| *Predicted degradation vs. actual with RUL cycles* | *Automated alert when SoH drops below safe thresholds* |

---

## 📈 Evaluation & Results

### 1. State of Health Degradation Profiles
![Degradation Curves](docs/images/1_soh_degradation_curves.png)
*Figure 1: Real measured degradation curves across cells B0005, B0006, and B0007. Every cell eventually degrades below the 80% EOL threshold.*

### 2. Predicted vs. Actual SoH (Held-Out Battery B0007)
![Predicted vs Actual](docs/images/2_predicted_vs_actual_soh.png)
*Figure 2: Performance on unseen battery B0007 (XGBoost MAE: 0.80% SoH), showing tight scatter fit and accurate degradation trajectory tracking.*

### 3. Baseline Comparison (Sensor-Only Features)
![Model Comparison](docs/images/6_model_comparison_bar_chart.png)
*Figure 3: Benchmark comparison on operational sensor features: XGBoost achieves an average 58.3% error reduction over baseline linear regression.*

### 4. Feature Importance Rankings
![Feature Importance](docs/images/4_xgboost_feature_importance.png)
*Figure 4: Relative feature importance shows that discharge capacity, capacity fade rate, and temperature are key indicators of health.*

---

## 💡 Real Dashboard Output Example

When inspecting battery **B0005** inside the dashboard:
* **Initial State:** Brand new cell at Cycle 1 has $100\%$ SoH ($2.035\text{ Ah}$).
* **Mid-Life Checkpoint (Cycle 142):** Measured SoH is **$85.58\%$** with ~96 cycles of remaining life left until the 80% threshold.
* **First Cycle Crossing 80% EOL:** Reached at **Cycle 238** (SoH drops to $79.70\%$).
* **Current Final State (Cycle 616):** Fully aged cell; the dashboard flags **0 cycles remaining** with a **CRITICAL / MAINTENANCE REQUIRED** warning banner.

---

## ⚙️ Engineering Design Decisions

1. **Why XGBoost instead of Deep Neural Networks (LSTM)?**
   * **Dataset Size:** With ~1,280 total cycles, LSTMs easily overfit and require complex tuning. Gradient-boosted trees perform exceptionally well on tabular sensor aggregates.
   * **Inference Latency:** Tree evaluation executes in microseconds on low-cost automotive microcontrollers.
   * **Explainability:** Feature importance provides direct insight into why a health prediction was made.
2. **Why Rolling Averages?**
   * Battery voltage and temperature fluctuate due to ambient chamber cycles and load variations. 5-cycle rolling windows filter out transient noise without lagging physical trends.
3. **Why the 80% SoH Threshold?**
   * Recognized internationally by automotive standards (USABC, SAE) as the boundary where internal cell impedance rises rapidly, making the battery unsuitable for high-draw acceleration.

---

## ⚠️ Limitations & Future Work

* **Laboratory Constant Discharge:** Tested under constant 2.0A load rather than real-world dynamic driving cycles (like WLTP or US06).
* **Controlled Temperature:** Cycled primarily at 24°C; real cars face winter sub-zero freezes and summer desert heat.
* **Roadmap:** Incorporate variable driving cycles from the Oxford battery dataset, implement thermal compensation curves, and export models to embedded C for microcontrollers.
* *Read more in [docs/LIMITATIONS_AND_FUTURE_WORK.md](docs/LIMITATIONS_AND_FUTURE_WORK.md).*

---

## 🎓 What I Learned

* **Real Data is Messy:** Extracting nested MATLAB structs required building defensive parsing utilities that handle missing fields, unexpected array shapes, and varying cycle lengths.
* **Data Leakage Matters:** Randomly splitting time-series cycles creates deceptively perfect metrics ($0.26\%$ error). A proper battery-wise split reveals true model generalization.
* **End-to-End Delivery:** ML is only useful when engineers and technicians can interact with it. Pairing an XGBoost pipeline with a Streamlit UI makes predictive maintenance practical.

---

## 💻 Tech Stack

* **Language:** Python 3.9+
* **Data Analysis & Processing:** NumPy, Pandas, SciPy (loadmat)
* **Machine Learning:** XGBoost, Scikit-learn, Joblib
* **Visualization:** Matplotlib, Streamlit
* **Automation & Testing:** Pytest, Playwright, Subprocess

---

## 📖 Acknowledgements & Citation

Dataset provided by the **NASA Ames Prognostics Center of Excellence (PCoE)**:
> B. Saha and K. Goebel (2007). "Battery Data Set", NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

---

## 📄 License & Author

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

* **Author:** Baliji Niroop
* **Major:** B.Tech in Automobile Engineering
* **GitHub:** [@Baliji-Niroop](https://github.com/Baliji-Niroop)
* **Project Repo:** [EV-Battery-Health-Predictor](https://github.com/Baliji-Niroop/EV-Battery-Health-Predictor)
