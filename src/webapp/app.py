# src/webapp/app.py
"""Streamlit UI for the Breast Cancer Risk research prototype.

The app presents a structured form based on the pre-diagnosis feature schema.
Users can input patient characteristics using predefined structured choices
(dropdowns and radio buttons for categorical/binary features, and bounded numeric inputs)
to obtain a model risk prediction (Low / Moderate / High). All inputs are processed
in-memory – no data is saved or transmitted.
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.predict.predict import predict
from src.webapp.form_config import (
    BINARY_CONFIG,
    CATEGORICAL_OPTIONS,
    MULTI_CAT_CONFIG,
    NUMERIC_CONFIG,
    format_form_payload,
)


# ---------------------------------------------------------------------------
# Page Configuration & Header
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Risk – Research Prototype",
    page_icon="🧬",
    layout="centered",
)

st.title("🧬 Breast Cancer Risk – Research Prototype")

st.write(
    "Please enter patient characteristics using the structured choices below. "
    "Fields marked with **(*)** are primary indicators. Unselected or unknown fields "
    "are handled cleanly by the model's preprocessing imputation pipeline."
)


# ---------------------------------------------------------------------------
# Structured Form
# ---------------------------------------------------------------------------
with st.form(key="risk_form"):
    raw_inputs = {}

    # 1. Demographics & Anthropometrics
    st.subheader("1. Demographics & Anthropometrics")
    col1, col2 = st.columns(2)

    with col1:
        cfg = NUMERIC_CONFIG["Age"]
        raw_inputs["Age"] = st.number_input(
            label=f"{cfg['label']} ({cfg['unit']}) *",
            min_value=cfg["min_value"],
            max_value=cfg["max_value"],
            step=cfg["step"],
            value=None,
            placeholder="e.g. 52",
            help=cfg["help"],
        )

        cfg = NUMERIC_CONFIG["Height_cm"]
        raw_inputs["Height_cm"] = st.number_input(
            label=f"{cfg['label']} ({cfg['unit']})",
            min_value=cfg["min_value"],
            max_value=cfg["max_value"],
            step=cfg["step"],
            value=None,
            placeholder="e.g. 162.0",
            help=cfg["help"],
        )

    with col2:
        cfg = BINARY_CONFIG["Gender"]
        raw_inputs["Gender"] = st.radio(
            label=f"{cfg['label']} *",
            options=cfg["options"],
            index=None,
            horizontal=True,
            help=cfg["help"],
        )

        cfg = NUMERIC_CONFIG["Weight_kg"]
        raw_inputs["Weight_kg"] = st.number_input(
            label=f"{cfg['label']} ({cfg['unit']})",
            min_value=cfg["min_value"],
            max_value=cfg["max_value"],
            step=cfg["step"],
            value=None,
            placeholder="e.g. 68.0",
            help=cfg["help"],
        )

    st.caption("ℹ️ **Body Mass Index (BMI):** Automatically calculated from Height and Weight.")

    st.markdown("---")

    # 2. Reproductive & Hormonal History
    st.subheader("2. Reproductive & Hormonal History")
    col3, col4 = st.columns(2)

    with col3:
        cfg = BINARY_CONFIG["Menopausal_Status"]
        raw_inputs["Menopausal_Status"] = st.radio(
            label=cfg["label"],
            options=cfg["options"],
            index=None,
            horizontal=True,
            help=cfg["help"],
        )

        cfg = NUMERIC_CONFIG["Age_First_Menstruation"]
        raw_inputs["Age_First_Menstruation"] = st.number_input(
            label=f"{cfg['label']} ({cfg['unit']})",
            min_value=cfg["min_value"],
            max_value=cfg["max_value"],
            step=cfg["step"],
            value=None,
            placeholder="e.g. 12",
            help=cfg["help"],
        )

        cfg = MULTI_CAT_CONFIG["Breastfeeding_History"]
        raw_inputs["Breastfeeding_History"] = st.selectbox(
            label=cfg["label"],
            options=cfg["options"],
            index=None,
            placeholder="Select breastfeeding history...",
            help=cfg["help"],
        )

    with col4:
        cfg = NUMERIC_CONFIG["Age_First_Pregnancy"]
        raw_inputs["Age_First_Pregnancy"] = st.number_input(
            label=f"{cfg['label']} ({cfg['unit']})",
            min_value=cfg["min_value"],
            max_value=cfg["max_value"],
            step=cfg["step"],
            value=None,
            placeholder="0 if never pregnant",
            help=cfg["help"],
        )

        cfg = NUMERIC_CONFIG["Parity"]
        raw_inputs["Parity"] = st.number_input(
            label=cfg["label"],
            min_value=cfg["min_value"],
            max_value=cfg["max_value"],
            step=cfg["step"],
            value=None,
            placeholder="e.g. 1",
            help=cfg["help"],
        )

    st.markdown("---")

    # 3. Family / Genetic History
    st.subheader("3. Family & Genetic History")
    col5, col6 = st.columns(2)

    with col5:
        cfg = BINARY_CONFIG["Family_History"]
        raw_inputs["Family_History"] = st.radio(
            label=cfg["label"],
            options=cfg["options"],
            index=None,
            horizontal=True,
            help=cfg["help"],
        )

        cfg = BINARY_CONFIG["Previous_Benign_Breast_Disease"]
        raw_inputs["Previous_Benign_Breast_Disease"] = st.radio(
            label=cfg["label"],
            options=cfg["options"],
            index=None,
            horizontal=True,
            help=cfg["help"],
        )

    with col6:
        cfg = BINARY_CONFIG["BRCA_Mutation"]
        raw_inputs["BRCA_Mutation"] = st.radio(
            label=cfg["label"],
            options=cfg["options"],
            index=None,
            horizontal=True,
            help=cfg["help"],
        )

    st.markdown("---")

    # 4. Lifestyle & Clinical Factors
    st.subheader("4. Lifestyle & Clinical Factors")
    col7, col8 = st.columns(2)

    with col7:
        cfg = BINARY_CONFIG["Hormone_Replacement_Therapy"]
        raw_inputs["Hormone_Replacement_Therapy"] = st.radio(
            label=cfg["label"],
            options=cfg["options"],
            index=None,
            horizontal=True,
            help=cfg["help"],
        )

        cfg = MULTI_CAT_CONFIG["Smoking_Status"]
        raw_inputs["Smoking_Status"] = st.selectbox(
            label=cfg["label"],
            options=cfg["options"],
            index=None,
            placeholder="Select smoking status...",
            help=cfg["help"],
        )

        cfg = MULTI_CAT_CONFIG["Physical_Activity"]
        raw_inputs["Physical_Activity"] = st.selectbox(
            label=cfg["label"],
            options=cfg["options"],
            index=None,
            placeholder="Select activity level...",
            help=cfg["help"],
        )

        cfg = NUMERIC_CONFIG["Sleep_Hours"]
        raw_inputs["Sleep_Hours"] = st.number_input(
            label=f"{cfg['label']} ({cfg['unit']})",
            min_value=cfg["min_value"],
            max_value=cfg["max_value"],
            step=cfg["step"],
            value=None,
            placeholder="e.g. 7.0",
            help=cfg["help"],
        )

        cfg = BINARY_CONFIG["Diabetes"]
        raw_inputs["Diabetes"] = st.radio(
            label=cfg["label"],
            options=cfg["options"],
            index=None,
            horizontal=True,
            help=cfg["help"],
        )

        cfg = MULTI_CAT_CONFIG["Breast_Density"]
        raw_inputs["Breast_Density"] = st.selectbox(
            label=cfg["label"],
            options=cfg["options"],
            index=None,
            placeholder="Select BI-RADS density category...",
            help=cfg["help"],
        )

    with col8:
        cfg = BINARY_CONFIG["Oral_Contraceptive_Use"]
        raw_inputs["Oral_Contraceptive_Use"] = st.radio(
            label=cfg["label"],
            options=cfg["options"],
            index=None,
            horizontal=True,
            help=cfg["help"],
        )

        cfg = MULTI_CAT_CONFIG["Alcohol_Intake"]
        raw_inputs["Alcohol_Intake"] = st.selectbox(
            label=cfg["label"],
            options=cfg["options"],
            index=None,
            placeholder="Select intake (or unrecorded)...",
            help=cfg["help"],
        )

        cfg = MULTI_CAT_CONFIG["Diet_Quality"]
        raw_inputs["Diet_Quality"] = st.selectbox(
            label=cfg["label"],
            options=cfg["options"],
            index=None,
            placeholder="Select diet quality...",
            help=cfg["help"],
        )

        cfg = MULTI_CAT_CONFIG["Stress_Level"]
        raw_inputs["Stress_Level"] = st.selectbox(
            label=cfg["label"],
            options=cfg["options"],
            index=None,
            placeholder="Select stress level...",
            help=cfg["help"],
        )

        cfg = BINARY_CONFIG["Hypertension"]
        raw_inputs["Hypertension"] = st.radio(
            label=cfg["label"],
            options=cfg["options"],
            index=None,
            horizontal=True,
            help=cfg["help"],
        )

        cfg = BINARY_CONFIG["Previous_Radiation_Exposure"]
        raw_inputs["Previous_Radiation_Exposure"] = st.radio(
            label=cfg["label"],
            options=cfg["options"],
            index=None,
            horizontal=True,
            help=cfg["help"],
        )

    submit = st.form_submit_button("Predict Risk", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction Execution & Results Display
# ---------------------------------------------------------------------------
if submit:
    try:
        # Prepare payload with auto-calculated BMI and clean missing handling
        input_data = format_form_payload(raw_inputs)

        # Show automatically calculated BMI
        if input_data.get("BMI") is not None:
            st.info(f"📊 Automatically calculated BMI: **{input_data['BMI']} kg/m²** (computed from Height: {input_data.get('Height_cm')} cm, Weight: {input_data.get('Weight_kg')} kg)")

        result = predict(input_data)
        prediction = result["prediction"]
        scores = result.get("scores", {})

        st.subheader("Prediction Result")
        if prediction == "Low":
            st.success(f"### Predicted Risk Level: **{prediction}**")
        elif prediction == "Moderate":
            st.warning(f"### Predicted Risk Level: **{prediction}**")
        else:
            st.error(f"### Predicted Risk Level: **{prediction}**")

        st.write("#### Class Probabilities (Raw Model Scores)")
        col_s1, col_s2, col_s3 = st.columns(3)
        cols_map = {"Low": col_s1, "Moderate": col_s2, "High": col_s3}
        for label, prob in scores.items():
            target_col = cols_map.get(label, st)
            target_col.metric(label=f"{label} Risk", value=f"{prob * 100:.1f}%")

    except Exception as exc:
        st.error(f"Error during risk prediction: {exc}")

st.markdown("---")
st.caption(
    "**Disclaimer:** This research prototype reproduces synthetic labels from the `BCRSD_30K.csv` dataset. It is **not** a validated estimate of "
    "real-world breast cancer risk and must not be used for clinical decisions."
)
