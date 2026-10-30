# src/train/train_pre_model.py
"""Train the pre‑diagnosis risk model.

This script trains a scikit‑learn pipeline using only the features listed in
`config/pre_risk_features.yaml`. The resulting pipeline is saved as
`models/risk_pre_pipeline.pkl` and a companion metrics JSON is written to
`models/risk_pre_metrics.json`.
"""

import json
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "BCRSD_30K.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "risk_pre_pipeline.pkl"
METRICS_PATH = MODEL_DIR / "risk_pre_metrics.json"
FEATURES_PATH = PROJECT_ROOT / "config" / "pre_risk_features.yaml"

# ---------------------------------------------------------------------------
# Load data and feature list
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
TARGET_COL = "Breast_Cancer_Risk"
EXCLUDE = ["Patient_ID", "Cancer_Severity", TARGET_COL]

# Load approved pre‑risk feature names (YAML)
import yaml
with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
pre_features = cfg["pre_risk_features"]

# Ensure the listed features exist in the dataset
missing = set(pre_features) - set(df.columns)
if missing:
    raise ValueError(f"Pre‑risk feature list contains columns not present in the data: {missing}")

X = df[pre_features]
y = df[TARGET_COL]

# Determine column types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# Preprocess pipelines
numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocess = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# Split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models
log_reg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
log_pipe = Pipeline([("preprocess", preprocess), ("clf", log_reg)])
rf_pipe = Pipeline([("preprocess", preprocess), ("clf", rf)])

# Train
log_pipe.fit(X_train, y_train)
rf_pipe.fit(X_train, y_train)

# Evaluation helper
def evaluate(pipe, X, y):
    preds = pipe.predict(X)
    return {
        "accuracy": accuracy_score(y, preds),
        "balanced_accuracy": balanced_accuracy_score(y, preds),
        "macro_f1": f1_score(y, preds, average="macro"),
        "precision_per_class": precision_score(y, preds, average=None, zero_division=0).tolist(),
        "recall_per_class": recall_score(y, preds, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
        "class_labels": pipe.named_steps["clf"].classes_.tolist(),
    }

log_metrics = evaluate(log_pipe, X_test, y_test)
rf_metrics = evaluate(rf_pipe, X_test, y_test)

# Choose best by macro_f1
best_pipe = rf_pipe if rf_metrics["macro_f1"] >= log_metrics["macro_f1"] else log_pipe
best_metrics = rf_metrics if rf_metrics["macro_f1"] >= log_metrics["macro_f1"] else log_metrics

# Save pipeline and metrics
joblib.dump(best_pipe, MODEL_PATH)
output = {
    "logistic_regression": log_metrics,
    "random_forest": rf_metrics,
    "selected_model": "random_forest" if best_pipe is rf_pipe else "logistic_regression",
    "selected_metrics": best_metrics,
    "feature_columns": pre_features,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "random_state": 42,
    "train_test_split": {"test_size": 0.2, "stratify": True},
}
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("Pre-risk model training complete.")
print("Pipeline saved to", MODEL_PATH)
print("Metrics saved to", METRICS_PATH)
