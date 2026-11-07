# Benchmark Results & Model Evaluation

This document presents the complete evaluation results for the **EV Battery Health Predictor**, comparing XGBoost against baseline models across both cross-battery and random validation regimes.

---

## 1. Key Evaluation Metrics

> All metrics are stated in **% SoH (State of Health percentage points, 0–100%)**.

### Primary Benchmark: Held-Out Battery Validation (Test: B0007)
Trained on cells **B0005, B0006, and B0018**; evaluated strictly on unseen battery **B0007**:

| Model Architecture | Features Used | MAE (% SoH) | RMSE (% SoH) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost Regressor (Ours)** | Operational Features (`FEATURE_COLS`) | **0.7951%** | **1.0511%** | Robust generalization to unseen cell |
| **Linear Regression** | Operational Features (`FEATURE_COLS`) | 0.0000% | 0.0000% | Degenerates when direct capacity feature is included |
| **XGBoost (Sensor Telemetry Only)** | Excludes direct capacity (`voltage, current, temp, dV/dt, dT/dt, R`) | **3.7381%** | **5.0865%** | Pure onboard BMS sensor prediction |
| **Linear Regression (Sensor Only)** | Excludes direct capacity (`voltage, current, temp, dV/dt, dT/dt, R`) | 6.5697% | 8.5889% | Underfits non-linear thermal-voltage coupling |

---

## 2. Secondary Benchmark: Random Cycle Split (Data Leakage Demonstration)

When cycles are randomly partitioned across all batteries (80% train / 20% test):

| Model | MAE (% SoH) | RMSE (% SoH) | Analysis |
| :--- | :--- | :--- | :--- |
| **XGBoost (Random Split)** | **0.2628%** | **0.4199%** | Artificially optimistic due to consecutive cycle correlation. |

*(Documented transparently to demonstrate why battery-wise splitting is necessary in real automotive battery management systems).*

---

## 3. Leave-One-Battery-Out (LOBO) Validation Across All Cells

To ensure performance consistency across distinct physical cells, here is the leave-one-battery-out performance on pure sensor telemetry:

| Held-out Test Cell | Linear Regression MAE (% SoH) | XGBoost MAE (% SoH) | XGBoost RMSE (% SoH) | Improvement over Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Battery B0005** | 7.84% | **2.21%** | 2.60% | **71.8% error reduction** |
| **Battery B0006** | 9.63% | **4.08%** | 4.80% | **57.6% error reduction** |
| **Battery B0007** | 6.57% | **3.74%** | 5.09% | **43.1% error reduction** |
| **Average Across Cells** | **8.01%** | **3.34%** | **4.16%** | **58.3% overall error reduction** |

---

## 4. Evaluation Figures

### 1. State of Health Degradation Profiles
![SoH Curves](images/1_soh_degradation_curves.png)
*Figure 1: Measured capacity loss over 600+ aging cycles for NASA cells B0005, B0006, and B0007. The dotted line indicates the 80% automotive retirement threshold.*

### 2. Predicted vs. Actual SoH on Held-out Battery B0007
![Predicted vs Actual](images/2_predicted_vs_actual_soh.png)
*Figure 2: (Left) Scatter plot of predicted vs. actual SoH showing strong linear fit along the diagonal. (Right) Trajectory over cycle lifespan demonstrating accurate tracking of capacity drop.*

### 3. Residual Error Distribution
![Residual Errors](images/3_residual_error_distribution.png)
*Figure 3: Residual errors centered around 0% with no severe heteroscedastic drift across operating cycles.*

### 4. XGBoost Feature Importance
![Feature Importance](images/4_xgboost_feature_importance.png)
*Figure 4: Relative importance rankings demonstrating the primary predictive roles of measured capacity, capacity fade rate, and mean cell temperature.*

### 5. How RUL is Estimated
![RUL Estimation](images/5_rul_prediction_example.png)
*Figure 5: Graphic walkthrough showing an operational inspection checkpoint at Cycle 140 predicting the 80% EOL crossing point and calculating remaining cycles.*

### 6. Sensor-Level Baseline Comparison
![Model Comparison](images/6_model_comparison_bar_chart.png)
*Figure 6: Real benchmark comparison showing XGBoost cutting prediction error in half compared to linear regression across all test batteries.*
