import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr

from setup import X_train, X_test, y_train, y_test

# -------------------------------
# 0. Setup
# -------------------------------
os.makedirs("Figures", exist_ok=True)

# -------------------------------
# 1. Load Models & Scaler
# -------------------------------
lr_model  = joblib.load("Models/logistic_regression.pkl")
rf_model  = joblib.load("Models/random_forest.pkl")
xgb_model = joblib.load("Models/xgboost.pkl")
scaler    = joblib.load("scaler.pkl")

feature_names = list(X_train.columns)

print("Models and scaler loaded successfully.")

# -------------------------------
# 2. Compute SHAP Values
# -------------------------------

# --- Logistic Regression (Linear Explainer) ---
print("\nComputing SHAP values for Logistic Regression...")
lr_explainer  = shap.LinearExplainer(lr_model, scaler.transform(X_train))
lr_shap_raw   = lr_explainer.shap_values(scaler.transform(X_test))
lr_shap_mean  = np.abs(lr_shap_raw).mean(axis=0)

# --- Random Forest (Tree Explainer) ---
print("Computing SHAP values for Random Forest...")
rf_explainer  = shap.TreeExplainer(rf_model)
rf_shap_raw   = rf_explainer(X_test)
rf_shap_mean  = np.abs(rf_shap_raw.values[:, :, 1]).mean(axis=0)

# --- XGBoost (Tree Explainer) ---
print("Computing SHAP values for XGBoost...")
xgb_explainer = shap.TreeExplainer(xgb_model)
xgb_shap_raw  = xgb_explainer(X_test)

if xgb_shap_raw.values.ndim == 3:
    xgb_shap_mean = np.abs(xgb_shap_raw.values[:, :, 1]).mean(axis=0)
else:
    xgb_shap_mean = np.abs(xgb_shap_raw.values).mean(axis=0)

# -------------------------------
# 3. Feature Ranking Table
# -------------------------------

rankings = pd.DataFrame({
    "Feature": feature_names,
    "LR Mean SHAP": lr_shap_mean,
    "RF Mean SHAP": rf_shap_mean,
    "XGB Mean SHAP": xgb_shap_mean
})

rankings["LR Rank"]  = rankings["LR Mean SHAP"].rank(ascending=False).astype(int)
rankings["RF Rank"]  = rankings["RF Mean SHAP"].rank(ascending=False).astype(int)
rankings["XGB Rank"] = rankings["XGB Mean SHAP"].rank(ascending=False).astype(int)

rankings["Avg Rank"] = rankings[["LR Rank", "RF Rank", "XGB Rank"]].mean(axis=1)
rankings.sort_values("Avg Rank", inplace=True)

print("\n--- Cross-Model SHAP Feature Rankings ---")
print(rankings[["Feature", "LR Rank", "RF Rank", "XGB Rank", "Avg Rank"]])

rankings.to_csv("Figures/shap_feature_rankings.csv", index=False)
print("\nFeature rankings saved to Figures/shap_feature_rankings.csv")

# -------------------------------
# 4. Combined Bar Chart
# -------------------------------

fig, ax = plt.subplots(figsize=(10, 7))

x = np.arange(len(feature_names))
width = 0.25

# Sort features by average SHAP importance for better visualization
sort_idx = rankings.index.tolist()
sorted_features = [feature_names[i] for i in sort_idx]
sorted_lr = lr_shap_mean[sort_idx]
sorted_rf = rf_shap_mean[sort_idx]
sorted_xgb = xgb_shap_mean[sort_idx]

bars1 = ax.barh(x + width, sorted_lr, width, label="Logistic Regression", color="steelblue", edgecolor="black")
bars2 = ax.barh(x, sorted_rf, width, label="Random Forest", color="salmon", edgecolor="black")
bars3 = ax.barh(x - width, sorted_xgb, width, label="XGBoost", color="mediumseagreen", edgecolor="black")

ax.set_yticks(x)
ax.set_yticklabels(sorted_features)
ax.set_xlabel("Mean Absolute SHAP Value")
ax.set_title("Cross-Model SHAP Feature Importance Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("Figures/shap_feature_importance_comparison.png", bbox_inches="tight")
plt.show()
print("\nCombined SHAP feature importance chart saved to Figures/shap_feature_importance_comparison.png")

# -------------------------------
# 5. Spearman Rank Correlation (Agreement Score)
# -------------------------------

lr_ranks = rankings["LR Rank"].values
rf_ranks = rankings["RF Rank"].values
xgb_ranks = rankings["XGB Rank"].values

corr_lr_rf,   p_lr_rf   = spearmanr(lr_ranks,  rf_ranks)
corr_lr_xgb,  p_lr_xgb  = spearmanr(lr_ranks,  xgb_ranks)
corr_rf_xgb,  p_rf_xgb  = spearmanr(rf_ranks,  xgb_ranks)
avg_agreement = np.mean([corr_lr_rf, corr_lr_xgb, corr_rf_xgb])

print("\n--- Spearman Rank Correlation (Feature Ranking Agreement) ---")
print(f"LR  vs RF:      r = {corr_lr_rf:.4f}  (p = {p_lr_rf:.4f})")
print(f"LR  vs XGBoost: r = {corr_lr_xgb:.4f}  (p = {p_lr_xgb:.4f})")
print(f"RF  vs XGBoost: r = {corr_rf_xgb:.4f}  (p = {p_rf_xgb:.4f})")
print(f"Average Agreement Score: {avg_agreement:.4f}")

# -------------------------------
# 6. Heatmap of Rankings
# -------------------------------

rank_matrix = rankings[["LR Rank", "RF Rank", "XGB Rank"]].set_index(pd.Index(sorted_features))

fig, ax = plt.subplots(figsize=(7, 8))
import seaborn as sns
sns.heatmap(
    rank_matrix,
    annot=True,
    fmt="d",
    cmap="YlOrRd_r",
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "Feature Rank (1 = Most Important)"}
)
ax.set_title("Feature Importance Ranking Heatmap")
ax.set_xlabel("Model")
ax.set_ylabel("Feature")
plt.tight_layout()
plt.savefig("Figures/shap_feature_ranking_heatmap.png", bbox_inches="tight")
plt.show()
print("\nFeature ranking heatmap saved to Figures/shap_feature_ranking_heatmap.png")

# -------------------------------
# 7. Summary
# -------------------------------
print("\n--- Cross-Model SHAP Comparison Summary ---")
print(f"Top feature (LR):      {rankings.iloc[0]['Feature']} (Rank {rankings.iloc[0]['LR Rank']})")
print(f"Top feature (RF):      {rankings.iloc[0]['Feature']} (Rank {rankings.iloc[0]['RF Rank']})")
print(f"Top feature (XGBoost): {rankings.iloc[0]['Feature']} (Rank {rankings.iloc[0]['XGB Rank']})")
print(f"Average Spearman Agreement: {avg_agreement:.4f}")



