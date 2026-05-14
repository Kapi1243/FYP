import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from scipy.stats import spearmanr

from setup import X_train, X_test, y_train, y_test, RANDOM_STATE

# -------------------------------
# 0. Setup
# -------------------------------
os.makedirs("Figures", exist_ok=True)

# -------------------------------
# 1. Load Models
# -------------------------------
lr_model  = joblib.load("Models/logistic_regression.pkl")
rf_model  = joblib.load("Models/random_forest.pkl")
xgb_model = joblib.load("Models/xgboost.pkl")
scaler    = joblib.load("scaler.pkl")

feature_names = list(X_train.columns)
models = {
    "Logistic Regression": lr_model,
    "Random Forest":       rf_model,
    "XGBoost":             xgb_model,
}

print("Models loaded successfully.")

# -------------------------------
# 2. Helper — LIME importance averaged across test set
# -------------------------------
def get_lime_importance(model, X_train, X_test, feature_names, predict_fn, n_samples=50):
    """Average LIME feature importance across n_samples from the test set."""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=["No Disease", "Disease"],
        mode="classification",
        discretize_continuous=False,
        random_state=RANDOM_STATE
    )

    importance_matrix = np.zeros((n_samples, len(feature_names)))

    for i in range(n_samples):
        exp = explainer.explain_instance(
            X_test.iloc[i].values,
            predict_fn,
            num_features=len(feature_names)
        )
        exp_map = dict(exp.as_list())
        for j, feat in enumerate(feature_names):
            importance_matrix[i, j] = abs(exp_map.get(feat, 0.0))

    return importance_matrix.mean(axis=0)


# -------------------------------
# 3. Helper — LIME importance for single patient
# -------------------------------
def get_lime_local(model, X_train, instance, feature_names, predict_fn):
    """LIME feature importance for a single patient."""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=["No Disease", "Disease"],
        mode="classification",
        discretize_continuous=False,
        random_state=RANDOM_STATE
    )

    exp = explainer.explain_instance(
        instance.values,
        predict_fn,
        num_features=len(feature_names)
    )

    exp_map = dict(exp.as_list())
    return np.array([exp_map.get(feat, 0.0) for feat in feature_names])


# -------------------------------
# 4. Find Representative Patient
# -------------------------------
y_pred_rf = rf_model.predict(X_test)
y_test_arr = y_test.values

true_positive_idx = np.where((y_pred_rf == 1) & (y_test_arr == 1))[0]
representative_idx = true_positive_idx[0]

print(f"\nRepresentative patient: index {representative_idx}")
print(f"Actual: {int(y_test_arr[representative_idx])} | Predicted: {int(y_pred_rf[representative_idx])}")

# -------------------------------
# 5. Run Comparison Per Model
# -------------------------------
all_results = {}

for name, model in models.items():
    print(f"\n--- Processing {name} ---")

    # Define predict function (handle LR scaling)
    if name == "Logistic Regression":
        predict_fn     = lambda x: model.predict_proba(scaler.transform(x))
        X_test_model   = scaler.transform(X_test)
        X_train_model  = scaler.transform(X_train)
    else:
        predict_fn     = model.predict_proba
        X_test_model   = X_test.values
        X_train_model  = X_train.values

    # --- SHAP Global (averaged) ---
    print(f"  Computing SHAP (global)...")
    if name == "Logistic Regression":
        shap_explainer = shap.LinearExplainer(model, X_train_model)
        shap_raw       = shap_explainer.shap_values(X_test_model)
        shap_global    = np.abs(shap_raw).mean(axis=0)
    else:
        shap_explainer = shap.TreeExplainer(model)
        shap_raw       = shap_explainer(X_test)
        if shap_raw.values.ndim == 3:
            shap_global = np.abs(shap_raw.values[:, :, 1]).mean(axis=0)
        else:
            shap_global = np.abs(shap_raw.values).mean(axis=0)

    # --- SHAP Local (single patient) ---
    print(f"  Computing SHAP (local)...")
    if name == "Logistic Regression":
        shap_local_raw = shap_explainer.shap_values(
            X_test_model[representative_idx].reshape(1, -1)
        )
        shap_local = shap_local_raw[0]
    else:
        shap_local_raw = shap_explainer(X_test.iloc[[representative_idx]])
        if shap_local_raw.values.ndim == 3:
            shap_local = shap_local_raw.values[0, :, 1]
        else:
            shap_local = shap_local_raw.values[0]

    # --- LIME Global (averaged) ---
    print(f"  Computing LIME (global, 50 samples)...")
    lime_global = get_lime_importance(
        model, X_train, X_test, feature_names, predict_fn, n_samples=50
    )

    # --- LIME Local (single patient) ---
    print(f"  Computing LIME (local)...")
    lime_local = get_lime_local(
        model, X_train, X_test.iloc[representative_idx], feature_names, predict_fn
    )

    all_results[name] = {
        "shap_global": shap_global,
        "lime_global": lime_global,
        "shap_local":  shap_local,
        "lime_local":  lime_local,
    }

# -------------------------------
# 6. Global Comparison — Bar Charts + Spearman
# -------------------------------
print("\n--- Global SHAP vs LIME Comparison ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
global_agreements = {}

for ax, (name, res) in zip(axes, all_results.items()):
    shap_vals = np.abs(res["shap_global"])
    lime_vals = np.abs(res["lime_global"])

    # Normalise for fair comparison
    shap_norm = shap_vals / shap_vals.sum()
    lime_norm = lime_vals / lime_vals.sum()

    # Sort by SHAP importance
    sort_idx = np.argsort(shap_norm)
    sorted_features = [feature_names[i] for i in sort_idx]

    x = np.arange(len(feature_names))
    width = 0.35

    ax.barh(x + width / 2, shap_norm[sort_idx], width, label="SHAP", color="steelblue",  edgecolor="black")
    ax.barh(x - width / 2, lime_norm[sort_idx], width, label="LIME", color="salmon",      edgecolor="black")
    ax.set_yticks(x)
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel("Normalised Importance")
    ax.set_title(f"{name}")
    ax.legend()

    # Spearman agreement
    shap_ranks = pd.Series(shap_norm).rank(ascending=False).values
    lime_ranks = pd.Series(lime_norm).rank(ascending=False).values
    r, p = spearmanr(shap_ranks, lime_ranks)
    global_agreements[name] = {"r": r, "p": p}

    ax.text(
        0.98, 0.02, f"r = {r:.3f}\np = {p:.3f}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

plt.suptitle("Global SHAP vs LIME Feature Importance Comparison", fontsize=14)
plt.tight_layout()
plt.savefig("Figures/xai_comparison_global.png", bbox_inches="tight")
plt.show()
print("Saved: Figures/xai_comparison_global.png")

for name, scores in global_agreements.items():
    print(f"{name}: r = {scores['r']:.4f} (p = {scores['p']:.4f})")

# -------------------------------
# 7. Local Comparison — Single Patient
# -------------------------------
print("\n--- Local SHAP vs LIME Comparison (Single Patient) ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
local_agreements = {}

for ax, (name, res) in zip(axes, all_results.items()):
    shap_vals = res["shap_local"]
    lime_vals = res["lime_local"]

    # Sort by absolute SHAP value
    sort_idx = np.argsort(np.abs(shap_vals))
    sorted_features = [feature_names[i] for i in sort_idx]

    x = np.arange(len(feature_names))
    width = 0.35

    ax.barh(x + width / 2, shap_vals[sort_idx], width, label="SHAP", color="steelblue",  edgecolor="black")
    ax.barh(x - width / 2, lime_vals[sort_idx], width, label="LIME", color="salmon",      edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(x)
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel("Feature Contribution")
    ax.set_title(f"{name}")
    ax.legend()

    # Spearman agreement
    shap_ranks = pd.Series(np.abs(shap_vals)).rank(ascending=False).values
    lime_ranks = pd.Series(np.abs(lime_vals)).rank(ascending=False).values
    r, p = spearmanr(shap_ranks, lime_ranks)
    local_agreements[name] = {"r": r, "p": p}

    ax.text(
        0.98, 0.02, f"r = {r:.3f}\np = {p:.3f}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

plt.suptitle(
    f"Local SHAP vs LIME Comparison — Patient {representative_idx} "
    f"(Actual: Disease, Predicted: Disease)",
    fontsize=13
)
plt.tight_layout()
plt.savefig("Figures/xai_comparison_local.png", bbox_inches="tight")
plt.show()
print("Saved: Figures/xai_comparison_local.png")

for name, scores in local_agreements.items():
    print(f"{name}: r = {scores['r']:.4f} (p = {scores['p']:.4f})")

# -------------------------------
# 8. Summary Table
# -------------------------------
print("\n--- XAI Agreement Summary ---")
summary = []
for name in models.keys():
    summary.append({
        "Model":             name,
        "Global r":          f"{global_agreements[name]['r']:.4f}",
        "Global p":          f"{global_agreements[name]['p']:.4f}",
        "Local r":           f"{local_agreements[name]['r']:.4f}",
        "Local p":           f"{local_agreements[name]['p']:.4f}",
    })

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
summary_df.to_csv("Figures/xai_comparison_summary.csv", index=False)
print("\nSaved: Figures/xai_comparison_summary.csv")
