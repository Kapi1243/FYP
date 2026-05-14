# FYP Heart Disease XAI Project (Quick Run Guide)

This is a quick setup guide for the final project so anyone can run the project locally.

## 1. Requirements

- Python 3.10+ recommended
- `pip`

## 2. Create and activate a virtual environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib shap lime xgboost scipy streamlit
```

## 4. Run in this order

From the project root, run:

```bash
python LogisticRegression.py
python RandomForest.py
python decision_tree.py
python xgboost_model.py
```

These scripts train models and save `.pkl` files.

## 5. Start the dashboard

```bash
streamlit run app.py
```

Open the local URL shown in terminal (usually `http://localhost:8501`).

## 6. Optional analysis scripts

Run these only if you want extra figures/reports:

```bash
python eda.py
python shap_analysis.py
python lime_analysis.py
python shap_comparison.py
python xai_comparison.py
```

## Quick troubleshooting

- If `streamlit` command is not found, run:

```bash
python -m streamlit run app.py
```
