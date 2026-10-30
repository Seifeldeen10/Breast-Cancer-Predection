# src/predict/predict.py
"""Prediction module for the Breast Cancer Risk prototype.

This module loads the *pre‑diagnosis* risk model pipeline saved at
`models/risk_pre_pipeline.pkl`. It validates the input dictionary against
the approved schema defined in `config/feature_schema.yaml` and returns
the predicted risk category together with the raw class scores.

The function does **not** perform any data persistence – user‑provided
values are processed only in‑memory.
"""

import yaml
from pathlib import Path
import joblib
import numpy as np

# Load the approved schema once at import time
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "config" / "feature_schema.yaml"
with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema = yaml.safe_load(f)  # now loads YAML correctly

FEATURES = schema["features"]

# Path to the pre‑risk pipeline (trained on pre‑diagnosis features only)
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "risk_pre_pipeline.pkl"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Pre‑risk model not found at {MODEL_PATH}. Run the training script first.")

_PIPELINE = joblib.load(MODEL_PATH)


def _validate_input(data: dict) -> dict:
    """Validate and coerce a raw input mapping.

    * All required features must be present.
    * Numeric values must be castable to ``float``; ``ValueError`` is raised
      for invalid entries.
    * Categorical values are kept as strings – the pipeline's ``OneHotEncoder``
      will handle unseen categories with ``handle_unknown='ignore'``.
    * Missing values (``None`` or empty string) are converted to ``np.nan``
      so that the pipeline's ``SimpleImputer`` can act.
    """
    validated = {}
    for name, meta in FEATURES.items():
        if name not in data:
            raise ValueError(f"Missing required feature: {name}")
        raw = data[name]
        # Treat empty string / explicit None as missing
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            validated[name] = np.nan
            continue
        if meta["type"] == "numeric":
            try:
                validated[name] = float(raw)
            except Exception as exc:
                raise ValueError(f"Feature '{name}' expects a numeric value, got {raw!r}") from exc
        else:  # categorical
            validated[name] = str(raw)
    return validated


def predict(input_dict: dict) -> dict:
    """Return a prediction for a single patient.

    Parameters
    ----------
    input_dict: dict
        Mapping from feature name to value (strings or numbers).

    Returns
    -------
    dict
        ``{"prediction": <label>, "scores": {label: prob, ...}}``
    """
    validated = _validate_input(input_dict)
    # Convert to a single‑row DataFrame preserving column order expected by the pipeline
    import pandas as pd
    df = pd.DataFrame([validated])
    # Ensure the column order matches the training feature list
    # Determine expected pipeline features (numeric + categorical)
    pipeline_features = (
        list(_PIPELINE.named_steps["preprocess"].transformers_[0][2])
        + list(_PIPELINE.named_steps["preprocess"].transformers_[1][2])
    )
    # Ensure all expected features exist in the DataFrame; fill missing with NaN
    for col in pipeline_features:
        if col not in df.columns:
            df[col] = np.nan
    # Keep only the pipeline features in the correct order
    df = df[[col for col in pipeline_features if col in df.columns]]
    pred_label = _PIPELINE.predict(df)[0]
    pred_proba = _PIPELINE.predict_proba(df)[0]
    class_labels = _PIPELINE.named_steps["clf"].classes_
    scores = {label: float(prob) for label, prob in zip(class_labels, pred_proba)}
    return {"prediction": pred_label, "scores": scores}
