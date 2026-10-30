import os
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
    classification_report,
)

# Paths
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "BCRSD_30K.csv"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "risk_pipeline.pkl"
METRICS_PATH = MODEL_DIR / "risk_metrics.json"

# Load data
df = pd.read_csv(DATA_PATH)

# Target and feature selection
TARGET_COL = "Breast_Cancer_Risk"
# Columns to exclude from features
EXCLUDE_COLS = ["Patient_ID", "Cancer_Severity", TARGET_COL]
# Identify feature columns
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

# Separate X and y
y = df[TARGET_COL]
X = df[feature_cols]

# Determine column types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# Preprocess pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Split data (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models
log_reg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

# Pipelines
log_reg_pipe = Pipeline(steps=[("preprocess", preprocess), ("clf", log_reg)])
rf_pipe = Pipeline(steps=[("preprocess", preprocess), ("clf", rf)])

# Train models
log_reg_pipe.fit(X_train, y_train)
rf_pipe.fit(X_train, y_train)

# Evaluation function
def evaluate(pipe, X_test, y_test):
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "balanced_accuracy": balanced_accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "precision_per_class": precision_score(y_test, preds, average=None, zero_division=0).tolist(),
        "recall_per_class": recall_score(y_test, preds, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "class_labels": pipe.named_steps["clf"].classes_.tolist() if hasattr(pipe.named_steps["clf"], "classes_") else list(sorted(y_test.unique()))
    }
    return metrics

log_metrics = evaluate(log_reg_pipe, X_test, y_test)
rf_metrics = evaluate(rf_pipe, X_test, y_test)

# Choose best model (higher macro_f1)
best_pipe = rf_pipe if rf_metrics["macro_f1"] >= log_metrics["macro_f1"] else log_reg_pipe
best_metrics = rf_metrics if rf_metrics["macro_f1"] >= log_metrics["macro_f1"] else log_metrics

# Save best pipeline
joblib.dump(best_pipe, MODEL_PATH)

# Save metrics
output = {
    "logistic_regression": log_metrics,
    "random_forest": rf_metrics,
    "selected_model": "random_forest" if best_pipe is rf_pipe else "logistic_regression",
    "selected_metrics": best_metrics,
    "feature_columns": feature_cols,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "random_state": 42,
    "train_test_split": {
        "test_size": 0.2,
        "stratify": True
    }
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("Training complete. Model saved to", MODEL_PATH)
print("Metrics saved to", METRICS_PATH)
