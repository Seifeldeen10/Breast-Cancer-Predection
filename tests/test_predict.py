import pytest
import json
from pathlib import Path
import joblib
import numpy as np

# Fixture from conftest ensures model is trained

def load_pipeline():
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "risk_pre_pipeline.pkl"
    return joblib.load(model_path)

def load_schema():
    project_root = Path(__file__).resolve().parents[1]
    schema_path = project_root / "config" / "feature_schema.yaml"
    import yaml
    with open(schema_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["features"]

def test_predict_valid_input():
    """A valid minimal input should produce a prediction dict with scores."""
    pipeline = load_pipeline()
    schema = load_schema()
    # Build a minimal valid input using the first value of each feature; numeric -> 0, categorical -> placeholder
    input_data = {}
    for name, meta in schema.items():
        if meta["type"] == "numeric":
            input_data[name] = 0
        else:
            input_data[name] = "PLACEHOLDER"
    from src.predict.predict import predict
    result = predict(input_data)
    assert isinstance(result, dict)
    assert "prediction" in result and "scores" in result
    assert isinstance(result["scores"], dict)
    classes = pipeline.named_steps["clf"].classes_
    assert set(result["scores"].keys()) == set(classes)

def test_predict_missing_required_feature():
    """Missing a required feature should raise ValueError."""
    from src.predict.predict import predict
    schema = load_schema()
    required_feature = next(iter(schema))
    input_data = {k: (0 if v["type"] == "numeric" else "PLACEHOLDER") for k, v in schema.items() if k != required_feature}
    with pytest.raises(ValueError) as exc:
        predict(input_data)
    assert required_feature in str(exc.value)

def test_predict_invalid_numeric_value():
    """Providing a non‑numeric string for a numeric column should raise ValueError."""
    from src.predict.predict import predict
    schema = load_schema()
    numeric_feature = None
    for name, meta in schema.items():
        if meta["type"] == "numeric":
            numeric_feature = name
            break
    assert numeric_feature is not None
    input_data = {}
    for name, meta in schema.items():
        if meta["type"] == "numeric":
            input_data[name] = "not_a_number"
        else:
            input_data[name] = "PLACEHOLDER"
    with pytest.raises(ValueError) as exc:
        predict(input_data)
    assert numeric_feature in str(exc.value)

def test_predict_unseen_categorical_allowed():
    """An unseen categorical value should be accepted (handled by OneHotEncoder)."""
    from src.predict.predict import predict
    schema = load_schema()
    input_data = {}
    for name, meta in schema.items():
        if meta["type"] == "numeric":
            input_data[name] = 0
        else:
            input_data[name] = "UNSEEN_CATEGORY_VALUE"
    result = predict(input_data)
    assert "prediction" in result and "scores" in result
