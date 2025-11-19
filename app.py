# app.py - Streamlit app (safe for Streamlit Cloud)
import streamlit as st
# === IMPORTANT: set page config BEFORE any other Streamlit calls ===
st.set_page_config(layout="centered", page_title="Solar Production Line Failure Prediction System")

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# ---------------------------
# Model location configuration
# ---------------------------
# Preferred: put the model file in the repo root next to app.py
MODEL_PATH = Path(__file__).parent / "tuned_predictive_maintenance_model.joblib"

# Fallback: a URL (or a file path that your deployment system will resolve)
# NOTE: your uploaded local path (so you can convert it to a downloadable URL if needed)
# Provided local path: /mnt/data/tuned_predictive_maintenance_model.joblib
MODEL_DOWNLOAD_URL = "file:///mnt/data/tuned_predictive_maintenance_model.joblib"

# ---------------------------
# Utility: download model once (if not present)
# ---------------------------
def ensure_model(path: Path, download_url: str):
    """
    Ensure the model exists at `path`. If not, try to download it from download_url.
    If the download_url is a local file URL (file://...), attempt a local copy.
    """
    if path.exists():
        return path

    # Try to copy local file if download_url is a file:// path
    if download_url.startswith("file://"):
        local_src = download_url.replace("file://", "")
        try:
            if os.path.exists(local_src):
                # copy file
                with open(local_src, "rb") as rf, open(path, "wb") as wf:
                    wf.write(rf.read())
                return path
        except Exception as e:
            st.sidebar.error(f"Failed to copy model from local path {local_src}: {e}")

    # Otherwise try HTTP(S) download
    try:
        import requests
        resp = requests.get(download_url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return path
    except Exception as e:
        st.sidebar.error(f"Could not download model from {download_url}: {e}")
        return None

# ---------------------------
# Load model with caching
# ---------------------------
@st.cache_resource
def load_model_cached(path: str):
    return joblib.load(path)

# Try to ensure model exists on disk (copy/download fallback)
if not MODEL_PATH.exists():
    ensured = ensure_model(MODEL_PATH, MODEL_DOWNLOAD_URL)
    if not ensured:
        st.sidebar.error(
            f"Model not found at {MODEL_PATH}. Put 'tuned_predictive_maintenance_model.joblib' in the repo root "
            f"or set MODEL_DOWNLOAD_URL to a reachable URL."
        )
        model = None
    else:
        model = load_model_cached(str(MODEL_PATH))
else:
    try:
        model = load_model_cached(str(MODEL_PATH))
    except Exception as e:
        st.sidebar.error(f"Failed to load model at {MODEL_PATH}: {e}")
        model = None

# ---------------------------
# App UI
# ---------------------------
st.title("Solar Production Line Failure Prediction System")
st.write("Upload a CSV (same schema as training after cleaning/FE) or manually input features.")

uploaded = st.file_uploader("Upload CSV (preprocessed / features)", type=["csv"])

# feature list (same order used for training)
FEATURES = [
    "ACTIVITY",
    "TimesFlag",
    "time_delta_s",
    "time_delta_rolling_avg",
    "time_delta_rolling_std",
    "time_delta_s_lag_1",
    "time_delta_s_lag_2",
    "operator_action_rolling_count_1h",
    "shift_A",
    "shift_B",
    "route_IVretest",
    "route_MainProcess",
    "route_PostReworkProcess",
    "route_ReworkProcess",
]

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.warning("Uploaded CSV is missing required features: " + ", ".join(missing))
    else:
        X = df[FEATURES].fillna(0)
        if model is None:
            st.error("Model not loaded — cannot predict.")
        else:
            preds = model.predict(X)
            probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(preds))
            df["pred_is_failure"] = preds
            df["pred_prob_failure"] = probs
            st.subheader("Predictions (first 50 rows)")
            st.dataframe(df.head(50))

else:
    st.write("Or manually enter one sample:")
    # 3-per-row layout
    cols = st.columns(3)
    with cols[0]:
        ACTIVITY = st.number_input("ACTIVITY", value=1.0, format="%.3f")
    with cols[1]:
        TimesFlag = st.number_input("TimesFlag", value=0.0, format="%.3f")
    with cols[2]:
        time_delta_s = st.number_input("time_delta_s", value=60.0, format="%.3f")

    cols = st.columns(3)
    with cols[0]:
        time_delta_rolling_avg = st.number_input("time_delta_rolling_avg", value=60.0, format="%.3f")
    with cols[1]:
        time_delta_rolling_std = st.number_input("time_delta_rolling_std", value=20.0, format="%.3f")
    with cols[2]:
        time_delta_s_lag_1 = st.number_input("time_delta_s_lag_1", value=60.0, format="%.3f")

    cols = st.columns(3)
    with cols[0]:
        time_delta_s_lag_2 = st.number_input("time_delta_s_lag_2", value=60.0, format="%.3f")
    with cols[1]:
        operator_action_rolling_count_1h = st.number_input("operator_action_rolling_count_1h", value=0.0, format="%.3f")
    with cols[2]:
        shift = st.selectbox("Shift", ["A", "B"])

    cols = st.columns(3)
    with cols[0]:
        route = st.selectbox("Route", ["MainProcess", "IVretest", "PostReworkProcess", "ReworkProcess"])
    with cols[1]:
        shift_A = 1 if shift == "A" else 0
        shift_B = 1 if shift == "B" else 0
    with cols[2]:
        route_IVretest = 1 if route == "IVretest" else 0
        route_MainProcess = 1 if route == "MainProcess" else 0
        route_PostReworkProcess = 1 if route == "PostReworkProcess" else 0
        route_ReworkProcess = 1 if route == "ReworkProcess" else 0

    if st.button("Predict"):
        sample = np.array(
            [
                [
                    ACTIVITY,
                    TimesFlag,
                    time_delta_s,
                    time_delta_rolling_avg,
                    time_delta_rolling_std,
                    time_delta_s_lag_1,
                    time_delta_s_lag_2,
                    operator_action_rolling_count_1h,
                    shift_A,
                    shift_B,
                    route_IVretest,
                    route_MainProcess,
                    route_PostReworkProcess,
                    route_ReworkProcess,
                ]
            ]
        )

        if model is None:
            st.error("Model not loaded — cannot predict.")
        else:
            try:
                pred = model.predict(sample)[0]
                prob = model.predict_proba(sample)[0, 1] if hasattr(model, "predict_proba") else None
                st.success(f"Predicted is_failure: **{int(pred)}**" + (f" (prob: {prob:.3f})" if prob is not None else ""))
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Tip: put the same feature columns (and order) as the training pipeline in your CSV upload.")
