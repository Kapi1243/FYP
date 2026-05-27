import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap

from data_preparation import RANDOM_STATE, prepare_data

data = prepare_data(save_scaler=False)
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

# -------------------------------
# 0. Setup
# -------------------------------
os.makedirs("Figures", exist_ok=True)

# -------------------------------
# 1. Load Saved Model
# -------------------------------
rf = joblib.load("Models/random_forest.pkl")
print("Model loaded from Models/random_forest.pkl")

# -------------------------------
# 2. SHAP Explainer
# -------------------------------
explainer = shap.TreeExplainer(rf)
shap_values = explainer(X_test)

# Class 1 = Disease
shap_values_class1 = shap_values[:, :, 1]

# -------------------------------
# 3. Global Explanation — Beeswarm
# -------------------------------
plt.figure()
shap.summary_plot(shap_values_class1, X_test, show=False)
plt.title("SHAP Summary Plot (Beeswarm) — Random Forest")
plt.tight_layout()
plt.savefig("Figures/shap_summary_beeswarm.png", bbox_inches="tight")
plt.show()
print("Saved: Figures/shap_summary_beeswarm.png")

# -------------------------------
# 4. Global Explanation — Bar Plot
# -------------------------------
plt.figure()
shap.summary_plot(shap_values_class1, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Bar) — Random Forest")
plt.tight_layout()
plt.savefig("Figures/shap_summary_bar.png", bbox_inches="tight")
plt.show()
print("Saved: Figures/shap_summary_bar.png")

# -------------------------------
# 5. Dependence Plots — Key Features
# -------------------------------
for feature in ["thalach", "cp", "oldpeak", "ca"]:
    plt.figure()
    shap.dependence_plot(feature, shap_values_class1.values, X_test, show=False)
    plt.title(f"SHAP Dependence Plot — {feature}")
    plt.tight_layout()
    plt.savefig(f"Figures/shap_dependence_{feature}.png", bbox_inches="tight")
    plt.show()
    print(f"Saved: Figures/shap_dependence_{feature}.png")

# -------------------------------
# 6. Local Explanations — Interesting Cases
# -------------------------------
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

# Find interesting cases
y_test_arr = y_test.values

true_positive_idx  = np.where((y_pred == 1) & (y_test_arr == 1))[0]
false_positive_idx = np.where((y_pred == 1) & (y_test_arr == 0))[0]
false_negative_idx = np.where((y_pred == 0) & (y_test_arr == 1))[0]

cases = {}
if len(true_positive_idx)  > 0: cases["true_positive"]  = true_positive_idx[0]
if len(false_positive_idx) > 0: cases["false_positive"] = false_positive_idx[0]
if len(false_negative_idx) > 0: cases["false_negative"] = false_negative_idx[0]

for case_name, idx in cases.items():
    print(f"\n--- Local Explanation: {case_name} (index {idx}) ---")
    print(f"Actual: {int(y_test_arr[idx])} | Predicted: {int(y_pred[idx])} | Probability: {y_prob[idx]:.4f}")

    # Waterfall plot
    plt.figure()
    shap.plots.waterfall(shap_values_class1[idx], show=False)
    plt.title(f"SHAP Waterfall — {case_name}")
    plt.tight_layout()
    plt.savefig(f"Figures/shap_waterfall_{case_name}.png", bbox_inches="tight")
    plt.show()
    print(f"Saved: Figures/shap_waterfall_{case_name}.png")

# -------------------------------
# 7. Summary
# -------------------------------
print("\n--- SHAP Analysis Complete ---")
print(f"Plots saved to Figures/")
