import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from setup import X_train, X_test, y_train, y_test, RANDOM_STATE

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
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
import shap

# -------------------------------
# 0. Setup
# -------------------------------
os.makedirs("Figures", exist_ok=True)
os.makedirs("Models", exist_ok=True)

# -------------------------------
# 1. Hyperparameter Tuning
# -------------------------------
base_dt = DecisionTreeClassifier(
    random_state=RANDOM_STATE,
    class_weight="balanced"
)

param_grid = {
    "max_depth":        [3, 4, 5, 6, None],
    "min_samples_split": [2, 5, 10],
    "criterion":        ["gini", "entropy"]
}

print("Running GridSearchCV for Decision Tree...")
grid_search = GridSearchCV(
    base_dt,
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
dt = grid_search.best_estimator_

joblib.dump(dt, "Models/decision_tree.pkl")
print("Model saved to Models/decision_tree.pkl")

# -------------------------------
# 3. Cross Validation
# -------------------------------
cv_scores = cross_val_score(dt, X_train, y_train, cv=5, scoring="roc_auc")
print(f"\n5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# -------------------------------
# 4. Predictions
# -------------------------------
y_pred = dt.predict(X_test)
y_prob = dt.predict_proba(X_test)[:, 1]

# -------------------------------
# 5. Evaluation Metrics
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n--- Decision Tree ---")
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
plt.title("Decision Tree — Confusion Matrix")
plt.tight_layout()
plt.savefig("Figures/dt_confusion_matrix.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 7. ROC Curve
# -------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="steelblue", label=f"Decision Tree (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Decision Tree — ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("Figures/dt_roc_curve.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 8. Feature Importance
# -------------------------------
feature_importance = pd.Series(dt.feature_importances_, index=X_train.columns)
feature_importance_sorted = feature_importance.sort_values()

plt.figure(figsize=(8, 6))
feature_importance_sorted.plot(
    kind="barh",
    color=["steelblue" if v >= feature_importance.mean() else "salmon" for v in feature_importance_sorted]
)
plt.axvline(feature_importance.mean(), color="black", linewidth=0.8, linestyle="--", label="Mean Importance")
plt.title("Decision Tree — Feature Importance (Gini)")
plt.xlabel("Importance Score")
plt.legend()
plt.tight_layout()
plt.savefig("Figures/dt_feature_importance.png", bbox_inches="tight")
plt.show()

# -------------------------------
# 9. Tree Visualisation
# -------------------------------
# Full tree diagram
plt.figure(figsize=(24, 10))
plot_tree(
    dt,
    feature_names=list(X_train.columns),
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    fontsize=9,
    impurity=True,
    proportion=False
)
plt.title("Decision Tree — Full Structure")
plt.tight_layout()
plt.savefig("Figures/dt_tree_full.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: Figures/dt_tree_full.png")

# Text representation — useful for dissertation appendix
tree_text = export_text(dt, feature_names=list(X_train.columns))
print("\n--- Decision Tree Text Rules ---")
print(tree_text)

with open("Figures/dt_rules.txt", "w") as f:
    f.write(tree_text)
print("Saved: Figures/dt_rules.txt")

# -------------------------------
# 10. SHAP Analysis
# -------------------------------
print("\nComputing SHAP values for Decision Tree...")
shap_explainer = shap.TreeExplainer(dt)
shap_values    = shap_explainer(X_test)

if shap_values.values.ndim == 3:
    shap_values_class1 = shap.Explanation(
        values=shap_values.values[:, :, 1],
        base_values=shap_values.base_values[:, 1],
        data=shap_values.data,
        feature_names=list(X_train.columns)
    )
else:
    shap_values_class1 = shap_values

# SHAP Summary
plt.figure()
shap.summary_plot(shap_values_class1, X_test, show=False)
plt.title("Decision Tree — SHAP Summary")
plt.tight_layout()
plt.savefig("Figures/dt_shap_summary.png", bbox_inches="tight")
plt.show()
print("Saved: Figures/dt_shap_summary.png")

# SHAP vs Built-in Feature Importance Comparison
shap_mean = np.abs(shap_values_class1.values).mean(axis=0)
shap_importance = pd.Series(shap_mean, index=X_train.columns)

comparison_df = pd.DataFrame({
    "Gini Importance": feature_importance,
    "SHAP Importance": shap_importance
}).sort_values("Gini Importance", ascending=False)

# Normalise for fair comparison
comparison_df["Gini Norm"] = comparison_df["Gini Importance"] / comparison_df["Gini Importance"].sum()
comparison_df["SHAP Norm"] = comparison_df["SHAP Importance"] / comparison_df["SHAP Importance"].sum()

fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(len(comparison_df))
width = 0.35

ax.barh(x + width / 2, comparison_df["Gini Norm"], width, label="Built-in Gini", color="steelblue", edgecolor="black")
ax.barh(x - width / 2, comparison_df["SHAP Norm"],  width, label="SHAP",          color="salmon",    edgecolor="black")
ax.set_yticks(x)
ax.set_yticklabels(comparison_df.index)
ax.set_xlabel("Normalised Importance")
ax.set_title("Decision Tree — Built-in Gini vs SHAP Feature Importance")
ax.legend()
plt.tight_layout()
plt.savefig("Figures/dt_gini_vs_shap.png", bbox_inches="tight")
plt.show()
print("Saved: Figures/dt_gini_vs_shap.png")

# Spearman agreement between Gini and SHAP
from scipy.stats import spearmanr
gini_ranks = comparison_df["Gini Norm"].rank(ascending=False).values
shap_ranks = comparison_df["SHAP Norm"].rank(ascending=False).values
r, p = spearmanr(gini_ranks, shap_ranks)
print(f"\nSpearman Agreement (Gini vs SHAP): r = {r:.4f} (p = {p:.4f})")

# -------------------------------
# 11. Summary
# -------------------------------
print("\n--- Decision Tree Summary ---")
print(f"Best Parameters:        {grid_search.best_params_}")
print(f"Accuracy:               {accuracy:.4f}")
print(f"Recall (Disease):       {recall:.4f}")
print(f"AUC:                    {roc_auc:.4f}")
print(f"5-Fold CV AUC:          {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Gini vs SHAP Agreement: r = {r:.4f} (p = {p:.4f})")
