import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ARTIFACT_DIR = Path("artifacts")
LATEST_PATH = ARTIFACT_DIR / "latest_model.json"
DEFAULT_MODEL_PATH = ARTIFACT_DIR / "salary_model.joblib"
DEFAULT_METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"


@st.cache_resource
def load_model_and_metadata() -> tuple[object, dict]:
    model_path = DEFAULT_MODEL_PATH
    metadata_path = DEFAULT_METADATA_PATH

    if LATEST_PATH.exists():
        latest = json.loads(LATEST_PATH.read_text(encoding="utf-8"))
        model_path = Path(latest["model_path"])
        metadata_path = Path(latest["metadata_path"])

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run training first: python scripts/train.py"
        )

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="centered")
st.title("Experience to Salary Predictor")
st.caption("Predict salary from experience using the latest trained model version.")

try:
    model, _ = load_model_and_metadata()
except Exception as exc:
    st.error(str(exc))
    st.stop()

col1, col2 = st.columns(2)
with col1:
    months = st.number_input(
        "Experience (months)", min_value=0.0, max_value=600.0, value=24.0, step=1.0
    )
with col2:
    years = months / 12
    st.metric("Experience (years)", f"{years:.2f}")

if st.button("Predict Salary", type="primary"):
    features = pd.DataFrame(
        [{"experience_months": float(months)}], columns=["experience_months"]
    )
    predicted_salary_thousands = float(model.predict(features)[0])
    predicted_salary = predicted_salary_thousands * 1000

    st.success("Prediction completed")
    st.metric("Predicted Salary (thousands)", f"{predicted_salary_thousands:,.2f}")
    st.metric("Predicted Salary", f"{predicted_salary:,.2f}")

