# Breast Cancer Digital Twin

## Overview

This repository provides a **reproducible data‑audit pipeline** for candidate breast‑cancer datasets.  The audit inspects raw tabular files (CSV, TSV, Excel) and produces:

* A **machine‑readable JSON** report with dimensions, data types, missing‑value statistics, duplicate counts, categorical/value distributions, and flags for potential data‑quality issues.
* A **human‑readable Markdown** report that summarises the same information and explicitly notes uncertainties (e.g., unknown provenance, possible leakage).

The pipeline does **not** merge datasets, train models, or upload any data to external services.

---

## Repository Structure

```
Breast Cancer Digital Twin/
├─ .gitignore               # Excludes compiled artifacts, env, raw data, reports, secrets
├─ README.md                # You are reading it!
├─ requirements.txt         # Python dependencies
├─ config/
│   └─ feature_map.yaml     # Candidate feature mapping for risk vs. stage prediction
├─ data/
│   ├─ raw/                 # Place downloaded Kaggle datasets here (git‑ignored)
│   └─ processed/           # Intermediate artefacts (git‑ignored)
├─ src/
│   ├─ __init__.py
│   └─ audit/
│       ├─ __init__.py
│       └─ audit.py        # Core audit script
├─ tests/
│   ├─ __init__.py
│   └─ test_audit.py       # Unit tests using synthetic fixtures
└─ reports/                # Generated JSON/MD reports (git‑ignored)
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone <repo‑url> "Breast Cancer Digital Twin"
cd "Breast Cancer Digital Twin"
```

### 2. Create a Python virtual environment

```bash
python -m venv .venv
# Activate (PowerShell)
.\.venv\Scripts\Activate.ps1
# Or cmd
\.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Obtain the candidate datasets

| Dataset | Kaggle URL | How to download |
|--------|------------|-----------------|
| Breast Cancer Dataset | https://www.kaggle.com/datasets/emmanuelighwaya/breast-cancer-dataset | Use the Kaggle CLI or manual download; place the extracted CSV/TSV/Excel files under `data/raw/`. |
| Breast Cancer Risk & Severity Dataset | https://www.kaggle.com/datasets/gowtha69/breast-cancer-risk-and-severity-dataset | Same procedure; put files under `data/raw/`. |

If you cannot obtain a file, you can still run the audit on any other tabular file you have.

---

## Running the Audit

```bash
python -m src.audit.audit path/to/your/file.csv
```

The script will create two reports in the `reports/` folder:
* `<filename>_report.json`
* `<filename>_report.md`

If the provided file does not exist or has an unsupported extension, a clear error message is shown.

---

## Feature Mapping

The file `config/feature_map.yaml` contains **provisional** mappings that separate candidate columns for two modelling tasks:
* **risk_prediction** – features that could be used to predict the probability of developing cancer.
* **stage_prediction** – features that could be used to predict cancer stage at diagnosis.

All entries are **candidates**; they must be matched against the actual column names in each dataset before any modelling.

---

## Testing

```bash
pytest -q
```

The test suite uses tiny synthetic CSV/TSV/Excel fixtures that cover:
* Supported file types
* Missing values and duplicate rows
* Correct JSON/Markdown generation
* Graceful handling of missing or unsupported files

---

## Compatibility & Deployment

* Developed and tested on **Python 3.11** (compatible with 3.12).
* No external services are called – the pipeline runs fully locally.
* The repository can be deployed on free‑tier cloud notebooks (e.g., Google Colab) by installing the same `requirements.txt` and mounting a drive for the `data/` folder.

---

## License & Attribution

* The code in this repository is released under the **MIT License**.
* Dataset licenses are determined by the original Kaggle uploads.  If a license cannot be identified, the provenance entry in the report will be marked as **unknown**.

---

## Contributing

Feel free to open issues or pull requests if you find bugs or want to extend the audit (e.g., additional heuristics, new output formats).  Please keep the repository free of patient‑identifiable data.

---

*Happy auditing!*
