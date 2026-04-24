from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"


def configure_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="EV Battery Health Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load features CSV."""
    features_path = PROCESSED_DIR / "features.csv"
    if not features_path.exists():
        st.info("Precomputed results not found. Run: python run_pipeline.py")
        st.stop()
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(features_path)
        return df
    except Exception as exc:
        st.error(f"Failed to load features: {exc}")
        return pd.DataFrame()


@st.cache_resource
def load_model() -> object:
    """Load trained XGBoost model."""
    model_path = MODELS_DIR / "xgb_model.pkl"
    if not model_path.exists():
        st.info("Precomputed model not found. Run: python run_pipeline.py")
        return None
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return None


def main() -> None:
    """Run Streamlit dashboard."""
    configure_page()
    
    st.title("EV Battery Health & RUL Estimation")
    st.write("Real-time monitoring and predictive maintenance for battery systems")
    
    # Load data
    df = load_data()
    if df.empty:
        st.error("No data available. Please run preprocessing first.")
        return
    
    # Load model
    model = load_model()
    
    # Sidebar: Select battery
    batteries = sorted(df["battery_id"].unique())
    if len(batteries) == 0:
        st.error("No battery data available")
        return
    
    selected_battery = st.sidebar.selectbox("Select Battery", batteries)
    
    df_battery = df[df["battery_id"] == selected_battery].sort_values("cycle").reset_index(drop=True)
    
    if df_battery.empty:
        st.error(f"No data for battery {selected_battery}")
        return
    
    st.sidebar.write(f"**Battery ID:** {selected_battery}")
    st.sidebar.write(f"**Cycles:** {len(df_battery)}")
    
    if len(df_battery) > 0:
        st.sidebar.write(f"**Current SoH:** {df_battery['SoH'].iloc[-1]:.2f}%")
    else:
        st.sidebar.write("**Current SoH:** N/A")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["Health Status", "SoH Trend", "Predictions"])
    
    # TAB 1: Health Status
    with tab1:
        st.subheader("Battery Health Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        if len(df_battery) == 0:
            st.error("No data for this battery")
            return
        
        current_soh = df_battery["SoH"].iloc[-1]
        current_cycle = int(df_battery["cycle"].iloc[-1])
        total_cycles = len(df_battery)
        avg_temp = df_battery["temp_mean"].mean() if "temp_mean" in df_battery.columns else 0
        
        col1.metric("Current SoH (%)", f"{current_soh:.2f}")
        col2.metric("Current Cycle", current_cycle)
        col3.metric("Total Cycles", total_cycles)
        col4.metric("Avg Temp (°C)", f"{avg_temp:.2f}")
        
        st.divider()
        
        # Health status indicators
        if current_soh >= 90:
            st.success("✓ Battery in EXCELLENT condition")
        elif current_soh >= 80:
            st.info("⚠ Battery in GOOD condition")
        elif current_soh >= 70:
            st.warning("⚠ Battery in FAIR condition - Monitor closely")
        else:
            st.error("✗ Battery in POOR condition - Maintenance recommended")
    
    # TAB 2: SoH Trend
    with tab2:
        st.subheader("State of Health Trend")
        
        st.line_chart(df_battery, x="cycle", y="SoH")

        
        # Statistics
        st.subheader("Statistics")
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Max SoH", f"{df_battery['SoH'].max():.2f}%")
        col2.metric("Min SoH", f"{df_battery['SoH'].min():.2f}%")
        col3.metric("Avg SoH", f"{df_battery['SoH'].mean():.2f}%")
    
    # TAB 3: Predictions & RUL
    with tab3:
        st.subheader("RUL Estimation & Predictions")
        
        if model is not None:
            # Get feature columns (exclude metadata)
            feature_cols = [col for col in df_battery.columns if col not in ["battery_id", "cycle", "SoH"]]
            
            try:
                X_battery = df_battery[feature_cols].fillna(0.0)
                soh_pred = model.predict(X_battery)
                
                # Create prediction dataframe
                df_battery_pred = df_battery.copy()
                df_battery_pred["soh_pred"] = soh_pred
                
                # Rename columns for clarity in legend
                plot_df = df_battery_pred[["cycle", "SoH", "soh_pred"]].set_index("cycle")
                plot_df = plot_df.rename(columns={"SoH": "Actual SoH", "soh_pred": "Predicted SoH"})
                st.line_chart(plot_df)

                
                # RUL Calculation
                st.subheader("Remaining Useful Life (RUL)")
                
                # Find cycle where predicted SoH drops below 80%
                cycles_below_80 = np.where(soh_pred < 80.0)[0]
                if len(cycles_below_80) > 0:
                    # Map index back to actual cycle number
                    k_fail = df_battery["cycle"].iloc[cycles_below_80[0]]
                else:
                    # No failure predicted; use last cycle as fallback
                    k_fail = df_battery["cycle"].iloc[-1]
                current_cycle = int(df_battery["cycle"].iloc[-1])
                rul = max(k_fail - current_cycle, 0)

                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted Failure Cycle", int(k_fail))
                col2.metric("RUL (cycles remaining)", rul)
                
                if rul <= 0:
                    col3.error("⚠ CRITICAL")
                elif rul <= 100:
                    col3.warning("⚠ WARNING")
                else:
                    col3.success("✓ HEALTHY")
                
            except Exception as exc:
                st.error(f"Failed to generate predictions: {exc}")
        else:
            st.info("Precomputed model not found. Run: python run_pipeline.py")
    
    # Footer
    st.divider()
    st.caption("EV Battery Health Dashboard | Data-driven predictive maintenance")


if __name__ == "__main__":
    main()
