# src/eda/run_eda.py
"""Generate exploratory data analysis artifacts for the two raw datasets.
- reports/eda_summary.json
- reports/eda_summary.md
- reports/eda_distributions.png (Age histogram for both datasets)
"""

import pathlib
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = pathlib.Path("s:/Personal Projects/Breast Cancer Predection")
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / name)

datasets = {
    "BCRSD_30K.csv": load("BCRSD_30K.csv"),
    "breast_cancer_data___.csv": load("breast_cancer_data___.csv"),
}

summary = {}
for fname, df in datasets.items():
    # schema
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    # missing
    missing_counts = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()}
    # numeric stats
    numeric_stats = {}
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        numeric_stats[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
        }
    # categorical top 5 (including NaN as key)
    categorical_top5 = {}
    for col in df.select_dtypes(include="object").columns:
        vc = df[col].value_counts(dropna=False).head(5).to_dict()
        # Convert possible NaN key to string "nan"
        categorical_top5[col] = {str(k): int(v) for k, v in vc.items()}
    # potential target columns (heuristic)
    target_candidates = [c for c in df.columns if any(tok in c.lower() for tok in ["status", "stage", "risk", "severity", "diagnosis", "outcome"])]
    target_distributions = {}
    for col in target_candidates:
        vc = df[col].value_counts(dropna=False).to_dict()
        target_distributions[col] = {str(k): int(v) for k, v in vc.items()}
    # duplicate rows
    duplicate_rows = int(df.duplicated().sum())
    # assemble summary
    summary[fname] = {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "schema": schema,
        "missing_counts": missing_counts,
        "numeric_stats": numeric_stats,
        "categorical_top5": categorical_top5,
        "target_distributions": target_distributions,
        "duplicate_rows": duplicate_rows,
    }

# Cross‑tabulations for the 5k dataset (if present)
df5k = datasets["breast_cancer_data___.csv"]
if "Diagnosis_Status" in df5k.columns:
    cross = {}
    for col in ["Cancer_Stage", "Treatment", "Metastasis_Status", "Survival_Status"]:
        if col in df5k.columns:
            ct = pd.crosstab(df5k["Diagnosis_Status"], df5k[col], dropna=False)
            cross[col] = ct.to_dict()
    summary["breast_cancer_data___.csv"]["diagnosis_cross"] = cross

# Write JSON artifact
json_path = REPORTS_DIR / "eda_summary.json"
json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

# Write Markdown artifact (human readable)
md_lines = ["# Exploratory Data Analysis Summary", ""]
for fname, data in summary.items():
    md_lines.append(f"## {fname}")
    md_lines.append(f"- Rows: {data['shape']['rows']}, Columns: {data['shape']['columns']}")
    md_lines.append(f"- Duplicate rows: {data['duplicate_rows']}")
    md_lines.append("### Schema (column : dtype)")
    for col, dt in data["schema"].items():
        md_lines.append(f"- `{col}` : {dt}")
    md_lines.append("### Missing values (count)")
    if data["missing_counts"]:
        for col, cnt in data["missing_counts"].items():
            md_lines.append(f"- `{col}` : {cnt}")
    else:
        md_lines.append("_None_")
    md_lines.append("### Target‑like column distributions")
    for col, dist in data["target_distributions"].items():
        md_lines.append(f"- **{col}**")
        for val, cnt in dist.items():
            md_lines.append(f"  - `{val}` : {cnt}")
    if "diagnosis_cross" in data:
        md_lines.append("### Diagnosis cross‑tabulations")
        for other, table in data["diagnosis_cross"].items():
            md_lines.append(f"- vs **{other}**")
            # table is a nested dict {diag: {val: count}}
            md_lines.append("```json")
            md_lines.append(json.dumps(table, indent=2))
            md_lines.append("```")
    md_lines.append("")
md_path = REPORTS_DIR / "eda_summary.md"
md_path.write_text("\n".join(md_lines), encoding="utf-8")

# Simple visualization: Age histogram for both datasets (if Age column exists)
plt.figure(figsize=(8, 4))
if "Age" in datasets["BCRSD_30K.csv"].columns:
    plt.hist(datasets["BCRSD_30K.csv"]["Age"].dropna(), bins=30, alpha=0.5, label="BCRSD_30K")
if "Age" in datasets["breast_cancer_data___.csv"].columns:
    plt.hist(datasets["breast_cancer_data___.csv"]["Age"].dropna(), bins=30, alpha=0.5, label="5k dataset")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt_path = REPORTS_DIR / "eda_distributions.png"
plt.savefig(plt_path)
print("EDA artifacts generated")
