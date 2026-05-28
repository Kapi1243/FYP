# Heart Disease Prediction and Explainability

**Can machine learning models predict heart disease while remaining interpretable and trustworthy?**

This project develops and evaluates an explainable machine learning system for heart disease prediction using the UCI Cleveland Heart Disease Dataset. Rather than optimising for predictive accuracy alone, it empirically evaluates whether explanation methods are consistent, stable, and usable by non-technical users — a gap in much of the existing literature.

Built as a BSc dissertation at Bournemouth University (2025/2026), supervised by Dr Emili Balaguer-Ballester.

---

## Research questions

**RQ1** — Do different machine learning models produce consistent feature importance rankings when analysed using SHAP?

**RQ2** — Do SHAP and LIME produce consistent explanations for the same model at both global and local levels?

**RQ3** — How can XAI outputs be translated into plain English explanations for non-technical users?

---

## Key findings

**On model performance:**
Logistic Regression achieved the highest ROC-AUC (0.9498), while XGBoost achieved the strongest fixed-threshold classification performance (accuracy 0.85, F1 0.8302). This suggests that simpler, more interpretable models remain competitive with ensemble methods on this dataset — explainability does not require sacrificing predictive power.

**On cross-model SHAP consistency (RQ1):**
Feature importance rankings were broadly consistent across models. The strongest agreement was between Random Forest and XGBoost (Spearman r = 0.835, p < 0.001). Agreement involving Logistic Regression was moderate (r ≈ 0.63), reflecting the difference between linear and non-linear representations. The most consistently important features across all models were chest pain type, thalassemia, number of major vessels (ca), and ST depression (oldpeak).

**On SHAP vs LIME agreement (RQ2):**
Global agreement was very high across all three models, with Spearman correlations above 0.93 (p < 0.001). Local agreement was more variable — Random Forest and XGBoost maintained high local agreement (r > 0.90), while Logistic Regression dropped to moderate local agreement (r = 0.626). This matters: explanation methods may appear consistent at the population level but diverge for individual patients, which is where clinical decisions are actually made.

**On plain English explanations (RQ3):**
SHAP values were translated into natural language risk statements (e.g. "Chest pain type strongly increases predicted risk") using a threshold-based system relative to each prediction's maximum SHAP value. Categorical features are described in human-readable terms rather than numeric codes. The system includes a disclaimer that outputs support clinical judgement rather than replace it.

---

## Models implemented

| Model | ROC-AUC | Accuracy | Notes |
|---|---|---|---|
| Logistic Regression | 0.9498 | 0.8333 | Highest ranking ability; LinearSHAP |
| Random Forest | 0.9464 | 0.8333 | Best cross-validation stability (CV AUC 0.902) |
| XGBoost | 0.9263 | 0.8500 | Best fixed-threshold performance |
| Decision Tree | 0.8220 | 0.8000 | Fully transparent baseline; Gini-SHAP agreement r = 0.9929 |

All models evaluated using accuracy, precision, recall, F1, ROC-AUC, and five-fold cross-validation on the UCI Cleveland dataset (297 instances after preprocessing).

---

## Dashboard

An interactive Streamlit dashboard allows a user to:

- Enter patient data via human-readable input controls (e.g. thalassemia shown as "Normal", "Fixed Defect", "Reversible Defect" rather than numeric codes)
- Run one or more models simultaneously and compare predictions
- View SHAP waterfall plots showing per-feature contribution to each prediction
- View LIME bar charts showing local linear approximations
- Read plain English risk summaries generated from SHAP values
- Follow the Decision Tree's full decision path in step-by-step readable rules

---

## XAI methodology

SHAP was applied using model-appropriate explainers:
- **TreeSHAP** (exact) for Decision Tree, Random Forest, and XGBoost
- **LinearSHAP** (exact) for Logistic Regression
- Class 1 (disease) SHAP values extracted consistently across all models

LIME was applied using `LimeTabularExplainer` with `discretize_continuous=True` for dashboard outputs (readable range statements) and `discretize_continuous=False` for quantitative comparison (raw importance values for Spearman correlation).

SHAP vs LIME agreement was evaluated at:
- **Global level:** mean absolute SHAP values across all 60 test instances vs mean absolute LIME importances across 50 sampled test instances
- **Local level:** both methods applied to the same representative true positive patient

LIME stability was tested across five repeated runs on the same instance. Mean standard deviation across features was 0.0026, suggesting instability was not a significant concern on this dataset.

---

## Dataset

UCI Cleveland Heart Disease Dataset (Detrano et al., 1989)
- 303 original records; 297 after removing 6 records with missing values
- 13 clinical features; binary target (0 = no disease, 1 = disease present)
- Class distribution: 160 negative (53.9%), 137 positive (46.1%)
- 80/20 stratified train/test split; random seed 42 for reproducibility

Full feature descriptions are provided in the dissertation appendix.

---

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1       # Windows
source .venv/bin/activate           # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

Train all models and generate the performance table:

```bash
python train_models.py
```

This saves trained models to `Models/` and writes `Models/training_results.csv`.

Launch the dashboard:

```bash
streamlit run app.py
```

---

## Project structure

| File | Purpose |
|---|---|
| `setup.py` | Loads, cleans, splits, and scales the dataset |
| `train_models.py` | Trains all four models and writes consolidated results |
| `LogisticRegression.py` | Trains and evaluates Logistic Regression |
| `decision_tree.py` | Trains Decision Tree and exports readable rules |
| `RandomForest.py` | Trains and evaluates Random Forest |
| `xgboost_model.py` | Trains and evaluates XGBoost |
| `shap_analysis.py` | Generates SHAP explanations for all models |
| `lime_analysis.py` | Generates LIME explanations |
| `shap_comparison.py` | Cross-model SHAP feature ranking comparison |
| `xai_comparison.py` | SHAP vs LIME global and local agreement analysis |
| `app.py` | Streamlit dashboard |

Outputs:
- `Models/*.pkl` — trained models
- `scaler.pkl` — fitted StandardScaler (Logistic Regression only)
- `Models/training_results.csv` — model evaluation summary
- `Figures/` — EDA, evaluation, SHAP, LIME, and comparison plots

---

## Limitations

This system is an academic research artefact. It has not been clinically validated and should not be used to make medical decisions. SHAP and LIME explain model behaviour, not medical causality — a feature increasing predicted risk does not mean it causes heart disease. The UCI Cleveland dataset is a widely used benchmark but does not represent modern clinical populations.

---

## Ethics paper

A companion 5,000-word ethics paper examines the implications of deploying XAI systems in healthcare, including the risks of over-reliance on post-hoc explanations, the gap between model behaviour and medical causality, and the importance of user testing with clinicians before clinical adoption.

---

*Bournemouth University · BSc (Hons) Artificial Intelligence with Data Science · 2025/2026*
