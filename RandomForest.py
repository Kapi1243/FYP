import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from data_preparation import RANDOM_STATE, prepare_data

data = prepare_data()
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

from sklearn.ensemble import RandomForestClassifier
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
# 1. Hyperparameter Tuning
# -------------------------------
base_rf = RandomForestClassifier(
    random_state=RANDOM_STATE,
    class_weight="balanced"
)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10]
}

print("Running GridSearchCV — this may take a moment...")
grid_search = GridSearchCV(
    base_rf,
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
# 2. Train Best Model
# -------------------------------
rf = grid_search.best_estimator_

# Save model for reuse in XAI scripts
joblib.dump(rf, "Models/random_forest.pkl")
print("Model saved to Models/random_forest.pkl")

# -------------------------------
# 3. Cross-Validation
# -------------------------------
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="roc_auc")
print(f"\n5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# -------------------------------
# 4. Predictions
# -------------------------------
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

# -------------------------------
# 5. Evaluation Metrics
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n--- Random Forest ---")
print(f"Accuracy:      {accuracy:.4f}")
print(f"Precision:     {precision:.4f}")
print(f"Recall:        {recall:.4f}")
print(f"F1 Score:      {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

# -------------------------------
# 6. Confusion Matrix
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
plt.title("Random Forest — Confusion Matrix")
plt.tight_layout()
plt.savefig("Figures/rf_confusion_matrix.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 7. ROC Curve
# -------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="steelblue", label=f"Random Forest (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest — ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("Figures/rf_roc_curve.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 8. Feature Importance
# -------------------------------
feature_importance = pd.Series(rf.feature_importances_, index=X_train.columns)
feature_importance_sorted = feature_importance.sort_values()

plt.figure(figsize=(8, 6))
feature_importance_sorted.plot(
    kind="barh",
    color=["steelblue" if v >= feature_importance.mean() else "salmon" for v in feature_importance_sorted]
)
plt.axvline(feature_importance.mean(), color="black", linewidth=0.8, linestyle="--", label="Mean Importance")
plt.title("Random Forest — Feature Importance")
plt.xlabel("Importance Score (Gini)")
plt.legend()
plt.tight_layout()
plt.savefig("Figures/rf_feature_importance.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 9. Summary
# -------------------------------
print("\n--- Random Forest Summary ---")
print(f"Best Parameters:   {grid_search.best_params_}")
print(f"Accuracy:          {accuracy:.4f}")
print(f"Recall (Disease):  {recall:.4f}")
print(f"AUC:               {roc_auc:.4f}")
print(f"5-Fold CV AUC:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
