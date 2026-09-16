import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Turbofan RUL Predictor",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ NASA Turbofan Remaining Useful Life Predictor")
st.markdown("Predict the remaining useful life (RUL) of a turbofan engine from its latest sensor readings (FD001).")

# ------------------------------------------------------------
# Load model artifacts
# ------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    artifacts = joblib.load("rul_model_fd001.joblib")
    return artifacts

try:
    artifacts = load_artifacts()
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_cols = artifacts["feature_cols"]
    rul_clip = artifacts.get("rul_clip", 125)
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.stop()

# ------------------------------------------------------------
# Sidebar – input method
# ------------------------------------------------------------
st.sidebar.header("Input Method")
input_method = st.sidebar.radio(
    "How do you want to provide sensor data?",
    ["Manual sliders", "Upload CSV (last cycle)"]
)

# ------------------------------------------------------------
# Helper: create input dataframe
# ------------------------------------------------------------
def get_manual_inputs():
    st.subheader("Enter latest sensor readings")
    cols = st.columns(3)
    values = {}
    for i, col in enumerate(feature_cols):
        with cols[i % 3]:
            # Reasonable default ranges based on FD001 statistics
            values[col] = st.number_input(
                col,
                value=0.0,
                format="%.4f",
                key=col
            )
    return pd.DataFrame([values])

def get_csv_input():
    st.subheader("Upload a CSV containing the last cycle of an engine")
    st.markdown("The CSV must contain at least these columns:")
    st.code(", ".join(feature_cols))
    
    uploaded = st.file_uploader("Choose a CSV file", type=["csv", "txt"])
    if uploaded is not None:
        df = pd.read_csv(uploaded, sep=None, engine="python")  # auto-detect separator
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return None
        # Take the last row if multiple rows are present
        return df[feature_cols].tail(1).reset_index(drop=True)
    return None

# ------------------------------------------------------------
# Get data
# ------------------------------------------------------------
if input_method == "Manual sliders":
    input_df = get_manual_inputs()
else:
    input_df = get_csv_input()

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
if input_df is not None and st.button("Predict RUL", type="primary"):
    # Scale
    X_scaled = scaler.transform(input_df[feature_cols])
    
    # Predict
    pred = model.predict(X_scaled)[0]
    pred = max(0, pred)  # RUL cannot be negative
    
    # Display result
    st.success("### Prediction complete")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted RUL (cycles)", f"{pred:.1f}")
    with col2:
        st.metric("Clipped training RUL", rul_clip)
    with col3:
        risk = "High" if pred < 30 else "Medium" if pred < 80 else "Low"
        st.metric("Risk level", risk)
    
    # Simple visual gauge
    st.progress(min(pred / 150, 1.0))
    st.caption("Progress bar scaled to 0–150 cycles")

    # Show the values that were used
    with st.expander("Sensor values used for prediction"):
        st.dataframe(input_df.style.format("{:.4f}"))

# ------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------
st.markdown("---")
st.subheader("Model Feature Importance")

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(importance["feature"], importance["importance"], color="steelblue")
ax.set_xlabel("Importance")
ax.set_title("XGBoost Feature Importance")
st.pyplot(fig)

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption("Model trained on NASA C-MAPSS FD001 • Piecewise-linear RUL (clip=125) • XGBoost")