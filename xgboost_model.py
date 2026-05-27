import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb

from data_preparation import RANDOM_STATE, prepare_data

data = prepare_data()
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

# -------------------------------
# 0. Setup
# -------------------------------
os.makedirs("Figures", exist_ok=True)
os.makedirs("Models", exist_ok=True)

# -------------------------------
# 1. Class Imbalance Weight
# -------------------------------
# XGBoost uses scale_pos_weight instead of class_weight
# Calculated as: number of negative cases / number of positive cases
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.4f}")

# -------------------------------
# 2. Hyperparameter Tuning
# -------------------------------
base_xgb = xgb.XGBClassifier(
    random_state=RANDOM_STATE,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    verbosity=0
)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1]
}

print("Running GridSearchCV — this may take a moment...")
grid_search = GridSearchCV(
    base_xgb,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV AUC:     {grid_search.best_score_:.4f}")

# -------------------------------
# 3. Train Best Model
# -------------------------------
model = grid_search.best_estimator_

# Save model for reuse in XAI scripts
joblib.dump(model, "Models/xgboost.pkl")
print("Model saved to Models/xgboost.pkl")

# -------------------------------
# 4. Cross-Validation
# -------------------------------
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
print(f"\n5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# -------------------------------
# 5. Predictions
# -------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -------------------------------
# 6. Evaluation Metrics
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n--- XGBoost ---")
print(f"Accuracy:      {accuracy:.4f}")
print(f"Precision:     {precision:.4f}")
print(f"Recall:        {recall:.4f}")
print(f"F1 Score:      {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

# -------------------------------
# 7. Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d",
    cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("XGBoost — Confusion Matrix")
plt.tight_layout()
plt.savefig("Figures/xgb_confusion_matrix.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 8. ROC Curve
# -------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="steelblue", label=f"XGBoost (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("XGBoost — ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("Figures/xgb_roc_curve.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 9. Feature Importance
# -------------------------------
feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
feature_importance_sorted = feature_importance.sort_values()

plt.figure(figsize=(8, 6))
feature_importance_sorted.plot(
    kind="barh",
    color=["steelblue" if v >= feature_importance.mean() else "salmon" for v in feature_importance_sorted]
)
plt.axvline(feature_importance.mean(), color="black", linewidth=0.8, linestyle="--", label="Mean Importance")
plt.title("XGBoost — Feature Importance")
plt.xlabel("Importance Score (F-score)")
plt.legend()
plt.tight_layout()
plt.savefig("Figures/xgb_feature_importance.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 10. Summary
# -------------------------------
print("\n--- XGBoost Summary ---")
print(f"Best Parameters:   {grid_search.best_params_}")
print(f"Accuracy:          {accuracy:.4f}")
print(f"Recall (Disease):  {recall:.4f}")
print(f"AUC:               {roc_auc:.4f}")
print(f"5-Fold CV AUC:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
