# src/webapp/app.py
"""Streamlit UI for the Breast Cancer Risk research prototype.

The app presents a form based on the *pre‑diagnosis* feature schema defined in
`config/feature_schema.yaml`. Users can input patient characteristics and obtain a
risk prediction (Low / Moderate / High). All inputs are processed in‑memory – no
data is saved or transmitted.
"""

import streamlit as st
import yaml
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Now safe to import predict
from src.predict.predict import predict

# ---------------------------------------------------------------------------
# Load schema
# ---------------------------------------------------------------------------
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "config" / "feature_schema.yaml"
with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema = yaml.safe_load(f)  # dict with top‑level "features"

features = schema["features"]

# Define life_keys based on schema, excluding any keys not present
_raw_life_keys = [
    "Hormone_Replacement_Therapy",
    "Oral_Contraceptive_Use",
    "Smoking_Status",
    "Alcohol_Intake",
    "Physical_Activity",
    "Diet_Quality",
    "Sleep_Hours",
    "Stress_Level",
    "Diabetes",
    "Hypertension",
    "Breast_Density",
    "Previous_Radiation_Exposure",
    "Estrogen_Level",
    "Progesterone_Level",
    "CA15_3",
    "CEA_Level",
    "Genomic_Risk_Score",
]
# Keep only keys that exist in the feature schema
life_keys = [k for k in _raw_life_keys if k in features]


# ---------------------------------------------------------------------------
# Helper to render input widgets per feature type
# ---------------------------------------------------------------------------
def render_input(name, meta):
    ftype = meta["type"]
    desc = meta.get("description", "")
    if ftype == "numeric":
        # No hard bounds – use a generic number input
        return st.number_input(label=name, help=desc, format="%f")
    else:  # categorical
        # If allowed_values are known, present a selectbox, otherwise a text input
        options = meta.get("allowed_values")
        if options:
            return st.selectbox(label=name, options=options, help=desc)
        else:
            return st.text_input(label=name, help=desc)

# ---------------------------------------------------------------------------
# Build the form – sections correspond to the task description
# ---------------------------------------------------------------------------
st.title("🧬 Breast Cancer Risk – Research Prototype")
st.caption(
    "Synthetic‑label dataset. Not for screening, diagnosis, treatment, or clinical decision‑making."
)

with st.form(key="risk_form"):
    st.subheader("Demographics & Anthropometrics")
    demo_vals = {}
    for key in ["Age", "Gender", "Height_cm", "Weight_kg", "BMI"]:
        demo_vals[key] = render_input(key, features[key])

    st.subheader("Reproductive & Hormonal History")
    repro_keys = [
        "Menopausal_Status",
        "Age_First_Menstruation",
        "Age_First_Pregnancy",
        "Parity",
        "Breastfeeding_History",
    ]
    repro_vals = {k: render_input(k, features[k]) for k in repro_keys}

    st.subheader("Family / Genetic History")
    fam_keys = ["Family_History", "BRCA_Mutation", "Previous_Benign_Breast_Disease"]
    fam_vals = {k: render_input(k, features[k]) for k in fam_keys}

    st.subheader("Lifestyle & Clinical Factors")
    # Use the filtered list of lifestyle keys defined earlier
    life_vals = {k: render_input(k, features[k]) for k in life_keys}

    # Combine all inputs into a single dict
    input_data = {**demo_vals, **repro_vals, **fam_vals, **life_vals}

    submit = st.form_submit_button("Predict Risk")

if submit:
    try:
        result = predict(input_data)
        st.success(f"**Predicted Risk:** {result['prediction']}")
        # Show model scores (not calibrated probabilities)
        st.subheader("Model Scores (raw class probabilities)")
        for label, score in result["scores"].items():
            st.write(f"{label}: {score:.3f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("---")
st.caption(
    "**Disclaimer:** This research prototype reproduces synthetic labels from the `BCRSD_30K.csv` dataset. It is **not** a validated estimate of real‑world breast cancer risk and must not be used for clinical decisions."
)
