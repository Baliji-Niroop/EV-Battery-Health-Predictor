# Limitations & Future Roadmap

Engineering transparency is essential when building machine learning solutions for safety-critical automotive systems. This document outlines the real physical boundaries of the current models and provides a realistic technical roadmap.

---

## 1. Known Physical & Operational Limitations

1. **Laboratory vs. Real-World Driving Cycles:**
   * The NASA PCoE dataset was recorded under controlled test chambers with fixed discharge loads (constant 2.0 A discharge).
   * Real electric vehicles encounter highly dynamic current spikes (regenerative braking, sudden highway accelerations, stop-and-go city traffic).
2. **Constant Temperature Artifacts:**
   * Test cells B0005, B0006, and B0007 were cycled at a controlled ambient temperature of 24°C.
   * In reality, ambient temperatures swing between winter extremes (-20°C) and hot summer driving (45°C+), heavily influencing electrolyte conductivity and internal resistance.
3. **Limited Battery Sample Size:**
   * Training on 3–4 cells (18650 cylindrical LiCoO2/graphite chemistry) limits statistical generalization to other modern EV chemistries (such as Lithium Iron Phosphate - LFP, or Nickel Manganese Cobalt - NMC 811).
4. **No Future RUL Forecast:**
   * This release reports the first observed cycle at or below 80% SoH and zero RUL for profiles that have already crossed the threshold.
   * A future threshold estimate for a still-healthy profile requires validated future feature values or a separate degradation-trend forecast; the current model does not extrapolate.
5. **Not Safety-Certified:**
   * The project is for research and demonstration. It is not a real-time battery safety monitor or certified BMS and must not be used as the sole basis for battery operation or retirement decisions.

---

## 2. Realistic Technical Roadmap

- [ ] **Standardized Drive Cycle Integration:** Incorporate experimental datasets with variable current profiles (e.g. UDDS, US06, WLTP driving cycles from the Oxford Battery Degradation Dataset or CALCE).
- [ ] **Thermal Compensation Mapping:** Train temperature-dependent XGBoost models using NASA thermal datasets (e.g., cell B0018 cycled at 43°C and low-temperature runs).
- [ ] **Electrochemical Impedance Spectroscopy (EIS) Feature Extraction:** Extract high-frequency Nyquist arc parameters ($R_{electrolyte}$, $R_{SEI}$, $R_{ct}$) from raw impedance cycles to monitor SEI layer growth directly.
- [ ] **Embedded BMS Microcontroller Export:** Convert the trained tree ensemble to pure C code (using `m2cgen` or `treelite`) for real-time deployment onto an ARM Cortex-M or ESP32 microcontroller board.
