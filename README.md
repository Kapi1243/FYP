# FYP Heart Disease XAI Project

This project is a heart disease prediction and explainability artefact. It trains several machine learning models on the Cleveland heart disease dataset and provides a Streamlit dashboard for patient-level prediction with SHAP and LIME explanations.

## Requirements

- Python 3.10 or newer
- pip

## Setup

Create a virtual environment from a working Python installation:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python` is not recognised, install Python from python.org and tick **Add Python to PATH** during installation.

## Main Workflow

Train all models and generate the consolidated performance table:

```powershell
python train_models.py
```

This saves trained models to `Models/` and writes:

```text
Models/training_results.csv
```

Start the dashboard:

```powershell
streamlit run app.py
```

The dashboard allows a user to enter patient data, compare model predictions, and view plain-English, SHAP, LIME, and decision-tree explanations.

## Optional Individual Scripts

These scripts can still be run individually if separate figures or analysis outputs are needed:

```powershell
python LogisticRegression.py
python RandomForest.py
python decision_tree.py
python xgboost_model.py
python eda.py
python shap_analysis.py
python lime_analysis.py
python shap_comparison.py
python xai_comparison.py
```

For normal use, `train_models.py` and `app.py` are the recommended route.

## Outputs

- `Models/*.pkl` - trained models
- `scaler.pkl` - fitted scaler for Logistic Regression
- `Models/training_results.csv` - model evaluation summary
- `Figures/` - EDA, model evaluation, SHAP, LIME, and XAI comparison figures

## Disclaimer

This artefact is for academic demonstration only. It is not a clinical diagnostic tool and should not be used to make medical decisions without professional validation.
