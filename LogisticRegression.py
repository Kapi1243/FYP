import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from setup import X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, RANDOM_STATE

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
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
# 1. Train Model
# -------------------------------
model = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

# Save model for reuse in XAI scripts
joblib.dump(model, "Models/logistic_regression.pkl")
print("Model saved to Models/logistic_regression.pkl")

# -------------------------------
# 2. Cross-Validation
# -------------------------------
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="roc_auc")
print(f"\n5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# -------------------------------
# 3. Predictions
# -------------------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# -------------------------------
# 4. Evaluation Metrics
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n--- Logistic Regression ---")
print(f"Accuracy:      {accuracy:.4f}")
print(f"Precision:     {precision:.4f}")
print(f"Recall:        {recall:.4f}")
print(f"F1 Score:      {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

# -------------------------------
# 5. Confusion Matrix
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
plt.title("Logistic Regression — Confusion Matrix")
plt.tight_layout()
plt.savefig("Figures/lr_confusion_matrix.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 6. ROC Curve
# -------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="steelblue", label=f"Logistic Regression (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression — ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("Figures/lr_roc_curve.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 7. Feature Importance (Coefficients)
# -------------------------------
coefficients = pd.Series(model.coef_[0], index=X_train.columns)
coefficients_sorted = coefficients.sort_values()

plt.figure(figsize=(8, 6))
coefficients_sorted.plot(kind="barh", color=["salmon" if c < 0 else "steelblue" for c in coefficients_sorted])
plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
plt.title("Logistic Regression — Feature Coefficients")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig("Figures/lr_feature_importance.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 8. Summary
# -------------------------------
print("\n--- Logistic Regression Summary ---")
print(f"Accuracy:          {accuracy:.4f}")
print(f"Recall (Disease):  {recall:.4f}")
print(f"AUC:               {roc_auc:.4f}")
print(f"5-Fold CV AUC:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")