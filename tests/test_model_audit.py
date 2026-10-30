import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "risk_pre_pipeline.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "risk_pre_metrics.json"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "BCRSD_30K.csv"

def test_metrics_structure():
    """Ensure metrics JSON contains expected keys and both models."""
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "logistic_regression" in metrics
    assert "random_forest" in metrics
    for model_key in ["logistic_regression", "random_forest"]:
        m = metrics[model_key]
        for key in ["accuracy", "macro_f1", "balanced_accuracy", "precision_per_class", "recall_per_class", "confusion_matrix"]:
            assert key in m, f"{key} missing in {model_key} metrics"

def test_feature_exclusion_and_schema():
    """Verify excluded columns are not part of the pipeline input schema."""
    pipe = joblib.load(MODEL_PATH)
    # Verify that the pipeline uses exactly the approved pre‑risk features
    import yaml
    PRE_FEATURES_PATH = PROJECT_ROOT / "config" / "pre_risk_features.yaml"
    with open(PRE_FEATURES_PATH, "r", encoding="utf-8") as f:
        pre_cfg = yaml.safe_load(f)
    approved = set(pre_cfg["pre_risk_features"])

    preprocess = pipe.named_steps["preprocess"]
    numeric_cols = []
    categorical_cols = []
    for name, transformer, cols in preprocess.transformers_:
        if name == "num":
            numeric_cols = set(cols)
        elif name == "cat":
            categorical_cols = set(cols)
    pipeline_features = numeric_cols.union(categorical_cols)

    # All approved features must be present
    missing = approved - pipeline_features
    assert not missing, f"Missing approved features in pipeline: {missing}"
    # No extra features beyond approved list should be present
    extra = pipeline_features - approved
    assert not extra, f"Unexpected features in pipeline: {extra}"

def test_prediction_handling_missing_and_unseen():
    """Pipeline should handle NaNs and unseen categorical values without error."""
    pipe = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    sample = df.iloc[[0]].copy()
    # Introduce missing values in a numeric column
    numeric_cols = sample.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) > 0:
        sample.loc[sample.index[0], numeric_cols[0]] = np.nan
    # Introduce an unseen category in a categorical column (if any exists)
    cat_cols = sample.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        new_val = "UNSEEN_CATEGORY"
        sample.loc[sample.index[0], cat_cols[0]] = new_val
    # Prediction should not raise
    preds = pipe.predict(sample)
    assert len(preds) == 1
    # Also check predict_proba works
    probs = pipe.predict_proba(sample)
    assert probs.shape[0] == 1 and probs.shape[1] == len(pipe.named_steps["clf"].classes_)

def test_prediction_determinism():
    """Repeated predictions on the same input should be identical (deterministic)."""
    pipe = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    sample = df.iloc[[0]].copy()
    pred1 = pipe.predict(sample)
    pred2 = pipe.predict(sample)
    assert np.array_equal(pred1, pred2)
