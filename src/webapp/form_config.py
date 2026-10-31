# src/webapp/form_config.py
"""Form configuration and metadata for the Breast Cancer Risk Streamlit web app.

Defines human-friendly labels, widget types, exact allowed categories matching the
training dataset, numeric ranges/steps, and helper functions for payload formatting.
"""

from typing import Any, Dict, List, Optional


# Exact unique categories from the training dataset (BCRSD_30K.csv) and pre-risk pipeline
CATEGORICAL_OPTIONS: Dict[str, List[str]] = {
    "Gender": ["Female", "Male"],
    "Menopausal_Status": ["Pre", "Post"],
    "Breastfeeding_History": [
        "No",
        "Short (<6 months)",
        "Moderate (6-12 months)",
        "Long (>12 months)",
        "Not Applicable",
    ],
    "Family_History": ["No", "Yes"],
    "BRCA_Mutation": ["No", "Yes"],
    "Previous_Benign_Breast_Disease": ["No", "Yes"],
    "Hormone_Replacement_Therapy": ["No", "Yes"],
    "Oral_Contraceptive_Use": ["No", "Yes"],
    "Smoking_Status": ["Never", "Former", "Current"],
    "Alcohol_Intake": ["Low", "Moderate", "High"],
    "Physical_Activity": ["Low", "Moderate", "High"],
    "Diet_Quality": ["Poor", "Fair", "Good"],
    "Stress_Level": ["Low", "Moderate", "High"],
    "Diabetes": ["No", "Yes"],
    "Hypertension": ["No", "Yes"],
    "Breast_Density": ["A", "B", "C", "D"],
    "Previous_Radiation_Exposure": ["No", "Yes"],
}

# Binary features (2 discrete choices)
BINARY_CONFIG: Dict[str, Dict[str, Any]] = {
    "Gender": {
        "label": "Gender",
        "options": CATEGORICAL_OPTIONS["Gender"],
        "help": "Self-reported biological sex / gender.",
        "required": True,
    },
    "Menopausal_Status": {
        "label": "Menopausal Status",
        "options": CATEGORICAL_OPTIONS["Menopausal_Status"],
        "help": "Menopausal state: 'Pre' (pre-menopausal) or 'Post' (post-menopausal).",
        "required": False,
    },
    "Family_History": {
        "label": "Family History of Breast Cancer",
        "options": CATEGORICAL_OPTIONS["Family_History"],
        "help": "First-degree biological relative (mother, sister, daughter) diagnosed with breast cancer.",
        "required": False,
    },
    "BRCA_Mutation": {
        "label": "BRCA Genetic Mutation Status",
        "options": CATEGORICAL_OPTIONS["BRCA_Mutation"],
        "help": "Documented positive genetic test for BRCA1 or BRCA2 mutation.",
        "required": False,
    },
    "Previous_Benign_Breast_Disease": {
        "label": "Previous Benign Breast Disease",
        "options": CATEGORICAL_OPTIONS["Previous_Benign_Breast_Disease"],
        "help": "History of biopsy-proven benign breast condition (e.g., fibroadenoma, cyst, ductal hyperplasia).",
        "required": False,
    },
    "Hormone_Replacement_Therapy": {
        "label": "Hormone Replacement Therapy (HRT)",
        "options": CATEGORICAL_OPTIONS["Hormone_Replacement_Therapy"],
        "help": "Current or previous regular use of systemic postmenopausal hormone replacement therapy.",
        "required": False,
    },
    "Oral_Contraceptive_Use": {
        "label": "Oral Contraceptive Use",
        "options": CATEGORICAL_OPTIONS["Oral_Contraceptive_Use"],
        "help": "Current or previous regular use of oral birth control pills.",
        "required": False,
    },
    "Diabetes": {
        "label": "Diabetes Mellitus",
        "options": CATEGORICAL_OPTIONS["Diabetes"],
        "help": "Clinical diagnosis of Type 1 or Type 2 diabetes.",
        "required": False,
    },
    "Hypertension": {
        "label": "Hypertension (High Blood Pressure)",
        "options": CATEGORICAL_OPTIONS["Hypertension"],
        "help": "Clinical diagnosis or treatment for chronic hypertension.",
        "required": False,
    },
    "Previous_Radiation_Exposure": {
        "label": "Prior Chest Radiation Exposure",
        "options": CATEGORICAL_OPTIONS["Previous_Radiation_Exposure"],
        "help": "Prior therapeutic radiation treatment directed to the mantle or chest area.",
        "required": False,
    },
}

# Multi-category features (> 2 discrete choices)
MULTI_CAT_CONFIG: Dict[str, Dict[str, Any]] = {
    "Breastfeeding_History": {
        "label": "Breastfeeding History",
        "options": CATEGORICAL_OPTIONS["Breastfeeding_History"],
        "help": "Cumulative duration of breastfeeding across all births (select 'Not Applicable' if nulliparous).",
        "required": False,
    },
    "Smoking_Status": {
        "label": "Smoking Status",
        "options": CATEGORICAL_OPTIONS["Smoking_Status"],
        "help": "Tobacco smoking history (Never, Former, or Current smoker).",
        "required": False,
    },
    "Alcohol_Intake": {
        "label": "Alcohol Intake",
        "options": CATEGORICAL_OPTIONS["Alcohol_Intake"],
        "help": "Reported alcohol consumption level. Leave unselected if non-drinker or unrecorded.",
        "required": False,
    },
    "Physical_Activity": {
        "label": "Physical Activity Level",
        "options": CATEGORICAL_OPTIONS["Physical_Activity"],
        "help": "General weekly physical activity / exercise frequency.",
        "required": False,
    },
    "Diet_Quality": {
        "label": "Diet Quality",
        "options": CATEGORICAL_OPTIONS["Diet_Quality"],
        "help": "Self-reported overall nutritional quality of diet.",
        "required": False,
    },
    "Stress_Level": {
        "label": "Perceived Stress Level",
        "options": CATEGORICAL_OPTIONS["Stress_Level"],
        "help": "Self-reported perceived daily stress level.",
        "required": False,
    },
    "Breast_Density": {
        "label": "Mammographic Breast Density (BI-RADS)",
        "options": CATEGORICAL_OPTIONS["Breast_Density"],
        "help": "BI-RADS density category: A (Almost entirely fatty), B (Scattered density), C (Heterogeneously dense), D (Extremely dense).",
        "required": False,
    },
}

# Numeric features with continuous or discrete integer values
NUMERIC_CONFIG: Dict[str, Dict[str, Any]] = {
    "Age": {
        "label": "Age",
        "unit": "years",
        "min_value": 18,
        "max_value": 100,
        "step": 1,
        "is_integer": True,
        "help": "Patient age in years at consultation (training range: 25–85, median: 53).",
        "required": True,
    },
    "Height_cm": {
        "label": "Height",
        "unit": "cm",
        "min_value": 100.0,
        "max_value": 220.0,
        "step": 0.5,
        "is_integer": False,
        "help": "Standing height in centimeters (training range: 134.2–195.3, median: 161.1).",
        "required": False,
    },
    "Weight_kg": {
        "label": "Weight",
        "unit": "kg",
        "min_value": 30.0,
        "max_value": 200.0,
        "step": 0.5,
        "is_integer": False,
        "help": "Body weight in kilograms (training range: 34.2–143.3, median: 67.5).",
        "required": False,
    },
    "BMI": {
        "label": "Body Mass Index (BMI)",
        "unit": "kg/m²",
        "min_value": 10.0,
        "max_value": 60.0,
        "step": 0.1,
        "is_integer": False,
        "help": "Body mass index in kg/m² (training range: 16.0–48.0, median: 26.0). Auto-calculated from Height and Weight if left empty.",
        "required": False,
    },
    "Age_First_Menstruation": {
        "label": "Age at First Menstruation",
        "unit": "years",
        "min_value": 8,
        "max_value": 25,
        "step": 1,
        "is_integer": True,
        "help": "Age when menstrual periods began (training range: 9–17, median: 12).",
        "required": False,
    },
    "Age_First_Pregnancy": {
        "label": "Age at First Pregnancy",
        "unit": "years",
        "min_value": 0,
        "max_value": 55,
        "step": 1,
        "is_integer": True,
        "help": "Age at first pregnancy/delivery in years (training range: 0–41, median: 23). Enter 0 if never pregnant.",
        "required": False,
    },
    "Parity": {
        "label": "Parity (Full-Term Births)",
        "unit": "births",
        "min_value": 0,
        "max_value": 15,
        "step": 1,
        "is_integer": True,
        "help": "Number of full-term pregnancies / deliveries (training range: 0–4, median: 1).",
        "required": False,
    },
    "Sleep_Hours": {
        "label": "Average Nightly Sleep",
        "unit": "hours",
        "min_value": 2.0,
        "max_value": 16.0,
        "step": 0.5,
        "is_integer": False,
        "help": "Average nightly sleep duration in hours (training range: 4.0–10.0, median: 7.0).",
        "required": False,
    },
}


def calculate_bmi(height_cm: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
    """Calculate BMI (kg/m²) from height in cm and weight in kg."""
    if height_cm is not None and weight_kg is not None:
        try:
            h = float(height_cm)
            w = float(weight_kg)
            if h > 0 and w > 0:
                return round(w / ((h / 100.0) ** 2), 2)
        except (ValueError, TypeError, ZeroDivisionError):
            return None
    return None


def format_form_payload(raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Format and validate form inputs into the dictionary expected by predict.py.

    - BMI is ALWAYS automatically calculated from Height_cm and Weight_kg.
    - Preserves exact model feature names.
    - Preserves categorical string values matching training data.
    - Preserves numeric floats/ints.
    - Converts None/empty strings to None so the model pipeline imputer acts correctly.
    """
    payload = dict(raw_inputs)

    # Always auto-calculate BMI from height and weight
    height = payload.get("Height_cm")
    weight = payload.get("Weight_kg")
    calc = calculate_bmi(height, weight)
    if calc is not None:
        payload["BMI"] = calc
    elif "BMI" not in payload:
        payload["BMI"] = None

    return payload

