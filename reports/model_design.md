# Model Design Documentation

## Prediction Task Definition
- **Target column**: `Breast_Cancer_Risk`
- **Target categories**: `Low`, `Moderate`, `High`
- **Input features**: All columns in `BCRSD_30K.csv` except:
  - `Patient_ID` (unique identifier, not predictive)
  - `Cancer_Severity` (derived outcome label, excluded to avoid leakage)
  - `Breast_Cancer_Risk` (the target itself)

## Feature Selection Rationale
The remaining 44 columns represent pre‑diagnostic patient characteristics such as:
- Demographics (Age, Gender, Height_cm, Weight_kg, BMI)
- Reproductive history (Menopausal_Status, Age_First_Menstruation, Age_First_Pregnancy, Parity, Breastfeeding_History)
- Family / genetic factors (Family_History, BRCA_Mutation, Previous_Benign_Breast_Disease)
- Lifestyle (Hormone_Replacement_Therapy, Oral_Contraceptive_Use, Smoking_Status, Alcohol_Intake, Physical_Activity, Diet_Quality, Sleep_Hours, Stress_Level)
- Clinical measurements (Diabetes, Hypertension, Breast_Density, Previous_Radiation_Exposure, Age_Menopause, Estrogen_Level, Progesterone_Level, CA15_3, CEA_Level, Tumor_Size_mm, Tumor_Location, Tumor_Shape, Tumor_Margin, Calcification, Lymph_Node_Involvement, Mammography_BIRADS, Ultrasound_Result, MRI_Result, Ki67_Index, HER2_Status, ER_Status, PR_Status, Genomic_Risk_Score)

All these variables are collected **before** any breast‑cancer diagnosis and are therefore legitimate predictors for the synthetic risk label.

## Leakage Assessment
- **`Cancer_Severity`** directly mirrors the target (synthetic stage label) and was removed from the feature set.
- No other column encodes post‑diagnosis information (e.g., treatment or outcome). Thus the risk of target leakage is minimal.

## Modeling Choices
- **Baseline**: Multinomial Logistic Regression (simple linear model).
- **Strong model**: Random Forest (captures non‑linear interactions, handles mixed data types).
- Both models share the same preprocessing pipeline (median imputation + scaling for numeric features, most‑frequent imputation + one‑hot encoding for categoricals).
- Model selection is based on **macro‑F1**; the higher‑scoring model is persisted.

## Reproducibility
- Fixed random seed = 42 for train/test split and model initialization.
- Stratified split (80 % train / 20 % test) preserves class distribution.
- Entire preprocessing + classifier pipeline is saved as `models/risk_pipeline.pkl`.
- All relevant metadata (feature list, numeric/categorical splits, random seed, split parameters, evaluation metrics) are stored in `models/risk_metrics.json`.

## Usage Disclaimer
> **Research prototype:** The `Breast_Cancer_Risk` labels are synthetic. This tool **must not** be used for clinical screening, diagnosis, treatment, or decision‑making.
