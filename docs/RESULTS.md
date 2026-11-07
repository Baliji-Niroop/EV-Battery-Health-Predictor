# Benchmark Results & Model Evaluation

This document presents the complete evaluation results for the **EV Battery Health Predictor**, comparing XGBoost against baseline models across both cross-battery and random validation regimes.

---

## 1. Key Evaluation Metrics

> All metrics are stated in **% SoH (State of Health percentage points, 0–100%)**.

### Primary Benchmark: Held-Out Battery Validation (Test: B0007)
Trained on cells **B0005, B0006, and B0018**; evaluated strictly on unseen battery **B0007**:

| Model Architecture | Features Used | MAE (% SoH) | RMSE (% SoH) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost Regressor (Ours)** | Sensor-only (`FEATURE_COLS`) | **1.7999%** | **2.1841%** | Held-out battery B0007 |
| **Linear Regression** | Sensor-only (`FEATURE_COLS`) | 7.6508% | 10.0095% | Baseline on the same features and split |

---

## 2. Secondary Benchmark: Random Cycle Split (Data Leakage Demonstration)

When cycles are randomly partitioned across all batteries (80% train / 20% test):

| Model | MAE (% SoH) | RMSE (% SoH) | Analysis |
| :--- | :--- | :--- | :--- |
| **XGBoost (Random Split)** | **0.8492%** | **1.1760%** | Artificially optimistic due to consecutive cycle correlation. |

*(Documented transparently to demonstrate why battery-wise splitting is necessary in real automotive battery management systems).*

---

## 3. Leave-One-Battery-Out (LOBO) Validation Across All Cells

To ensure performance consistency across distinct physical cells, here is the leave-one-battery-out performance on pure sensor telemetry:

| Held-out Test Cell | Linear Regression MAE (% SoH) | XGBoost MAE (% SoH) | XGBoost RMSE (% SoH) | Improvement over Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Battery B0005** | 9.22% | **1.52%** | 1.95% | **83.6% error reduction** |
| **Battery B0006** | 12.26% | **4.44%** | 5.47% | **63.8% error reduction** |
| **Battery B0007** | 7.65% | **1.80%** | 2.18% | **76.5% error reduction** |
| **Battery B0018** | 10.28% | **2.71%** | 3.25% | **73.6% error reduction** |
| **Average Across Cells** | **9.85%** | **2.62%** | **3.21%** | **73.4% overall error reduction** |

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

### 5. Retrospective RUL Example
![RUL Estimation](images/5_rul_prediction_example.png)
*Figure 5: Historical B0007 profile showing the observed 80% crossing after a Cycle 140 checkpoint. This is retrospective, not a future forecast.*

### 6. Sensor-Level Baseline Comparison
![Model Comparison](images/6_model_comparison_bar_chart.png)
*Figure 6: Leave-one-battery-out sensor-only benchmark across all four cells; XGBoost reduces mean absolute error by 73.4% versus linear regression.*
