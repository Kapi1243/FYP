"""
Unified training pipeline: trains Logistic Regression, Random Forest,
Decision Tree, and XGBoost (if available). Saves fitted models to
`Models/` and writes a single results CSV at `Models/training_results.csv`.

Usage: python train_models.py
"""
import os
import joblib
import warnings
from pprint import pformat

import numpy as np
import pandas as pd

from data_preparation import prepare_data, RANDOM_STATE

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

OUTPUT_DIR_MODELS = "Models"
OUTPUT_DIR_FIGURES = "Figures"
RESULTS_CSV = os.path.join(OUTPUT_DIR_MODELS, "training_results.csv")

os.makedirs(OUTPUT_DIR_MODELS, exist_ok=True)
os.makedirs(OUTPUT_DIR_FIGURES, exist_ok=True)


def evaluate_model(name, estimator, X_test, y_test, X_train, y_train):
    """Fit already-fit estimator and evaluate on test set.
    For GridSearchCV objects we assume `.best_estimator_` is available.
    """
    # If GridSearchCV provided, get best estimator
    if hasattr(estimator, "best_estimator_"):
        model = estimator.best_estimator_
    else:
        model = estimator

    # Ensure estimator is fitted (GridSearchCV/estimator should be fit already)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        # some estimators might not support predict_proba
        # fall back to decision_function if available
        try:
            y_prob = model.decision_function(X_test)
            # scale to 0-1 using logistic sigmoid
            y_prob = 1 / (1 + np.exp(-y_prob))
        except Exception:
            y_prob = None

    y_pred = model.predict(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        except Exception:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    # 5-fold CV AUC on training set (if possible)
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        metrics["cv_auc_mean"] = float(cv_scores.mean())
        metrics["cv_auc_std"] = float(cv_scores.std())
    except Exception:
        metrics["cv_auc_mean"] = float("nan")
        metrics["cv_auc_std"] = float("nan")

    return model, metrics


def run():
    data = prepare_data()

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    X_train_scaled = data.get("X_train_scaled")
    X_test_scaled = data.get("X_test_scaled")

    results = []

    # -----------------------
    # 1) Logistic Regression
    # -----------------------
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
    # use scaled features if available
    if X_train_scaled is None:
        print("Scaler not found; using raw features for Logistic Regression")
        lr.fit(X_train, y_train)
        model_lr = lr
        Xtr, Xte = X_train, X_test
    else:
        lr.fit(X_train_scaled, y_train)
        model_lr = lr
        Xtr, Xte = X_train_scaled, X_test_scaled

    joblib.dump(model_lr, os.path.join(OUTPUT_DIR_MODELS, "logistic_regression.pkl"))
    print("Saved: Models/logistic_regression.pkl")

    model_lr, metrics_lr = evaluate_model("LogisticRegression", model_lr, Xte, y_test, Xtr, y_train)
    results.append(metrics_lr)

    # -----------------------
    # 2) Random Forest
    # -----------------------
    print("\nTraining Random Forest (GridSearchCV)...")
    base_rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    rf_param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
    }

    grid_rf = GridSearchCV(base_rf, rf_param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1)
    grid_rf.fit(X_train, y_train)

    best_rf = grid_rf.best_estimator_
    joblib.dump(best_rf, os.path.join(OUTPUT_DIR_MODELS, "random_forest.pkl"))
    print("Saved: Models/random_forest.pkl")

    best_rf, metrics_rf = evaluate_model("RandomForest", grid_rf, X_test, y_test, X_train, y_train)
    metrics_rf["best_params"] = pformat(grid_rf.best_params_)
    results.append(metrics_rf)

    # -----------------------
    # 3) Decision Tree
    # -----------------------
    print("\nTraining Decision Tree (GridSearchCV)...")
    base_dt = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    dt_param_grid = {
        "max_depth": [3, 4, 5, 6, None],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
    }

    grid_dt = GridSearchCV(base_dt, dt_param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1)
    grid_dt.fit(X_train, y_train)

    best_dt = grid_dt.best_estimator_
    joblib.dump(best_dt, os.path.join(OUTPUT_DIR_MODELS, "decision_tree.pkl"))
    print("Saved: Models/decision_tree.pkl")

    best_dt, metrics_dt = evaluate_model("DecisionTree", grid_dt, X_test, y_test, X_train, y_train)
    metrics_dt["best_params"] = pformat(grid_dt.best_params_)
    results.append(metrics_dt)

    # -----------------------
    # 4) XGBoost (if available)
    # -----------------------
    try:
        import xgboost as xgb  # type: ignore
        print("\nTraining XGBoost (GridSearchCV)...")

        # compute scale_pos_weight
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        base_xgb = xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False,
        )

        xgb_param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1],
        }

        grid_xgb = GridSearchCV(base_xgb, xgb_param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1)
        grid_xgb.fit(X_train, y_train)

        best_xgb = grid_xgb.best_estimator_
        joblib.dump(best_xgb, os.path.join(OUTPUT_DIR_MODELS, "xgboost.pkl"))
        print("Saved: Models/xgboost.pkl")

        best_xgb, metrics_xgb = evaluate_model("XGBoost", grid_xgb, X_test, y_test, X_train, y_train)
        metrics_xgb["best_params"] = pformat(grid_xgb.best_params_)
        results.append(metrics_xgb)

    except Exception as e:  # xgboost not installed
        print("XGBoost training skipped (xgboost not available):", e)

    # -----------------------
    # Finalize results
    # -----------------------
    results_df = pd.DataFrame(results)
    # order columns
    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "cv_auc_mean", "cv_auc_std", "best_params"]
    for c in cols:
        if c not in results_df.columns:
            results_df[c] = ""

    results_df = results_df[cols]

    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved consolidated results: {RESULTS_CSV}")
    print("\nResults summary:\n", results_df)


if __name__ == "__main__":
    run()
