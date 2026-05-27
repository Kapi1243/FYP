import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular

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
# 2. LIME Explainer
# -------------------------------
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=list(X_train.columns),
    class_names=["No Disease", "Disease"],
    mode="classification",
    discretize_continuous=True,
    random_state=RANDOM_STATE
)

# -------------------------------
# 3. Find Interesting Cases
# -------------------------------
y_pred = rf.predict(X_test)
y_test_arr = y_test.values

true_positive_idx  = np.where((y_pred == 1) & (y_test_arr == 1))[0]
false_positive_idx = np.where((y_pred == 1) & (y_test_arr == 0))[0]
false_negative_idx = np.where((y_pred == 0) & (y_test_arr == 1))[0]

cases = {}
if len(true_positive_idx)  > 0: cases["true_positive"]  = true_positive_idx[0]
if len(false_positive_idx) > 0: cases["false_positive"] = false_positive_idx[0]
if len(false_negative_idx) > 0: cases["false_negative"] = false_negative_idx[0]

# -------------------------------
# 4. Generate LIME Explanations
# -------------------------------
for case_name, idx in cases.items():
    print(f"\n--- LIME Explanation: {case_name} (index {idx}) ---")
    print(f"Actual: {int(y_test_arr[idx])} | Predicted: {int(y_pred[idx])}")

    exp = explainer.explain_instance(
        X_test.iloc[idx].values,
        rf.predict_proba,
        num_features=10
    )

    # Save as HTML
    exp.save_to_file(f"Figures/lime_{case_name}.html")
    print(f"Saved: Figures/lime_{case_name}.html")

    # Save as PNG
    fig = exp.as_pyplot_figure()
    plt.title(f"LIME Explanation — {case_name}")
    plt.tight_layout()
    plt.savefig(f"Figures/lime_{case_name}.png", bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: Figures/lime_{case_name}.png")

# -------------------------------
# 5. LIME Stability Check
# -------------------------------
# Run LIME multiple times on the same instance to show consistency
# This demonstrates LIME's stochastic nature — worth discussing in your write-up
if len(true_positive_idx) > 0:
    idx = true_positive_idx[0]
    print(f"\n--- LIME Stability Check (true positive, index {idx}) ---")

    feature_weights = {}

    for run in range(5):
        exp = explainer.explain_instance(
            X_test.iloc[idx].values,
            rf.predict_proba,
            num_features=10
        )
        for feature, weight in exp.as_list():
            if feature not in feature_weights:
                feature_weights[feature] = []
            feature_weights[feature].append(weight)

    print("Feature weight variance across 5 runs:")
    feature_stds = []
    for feature, weights in feature_weights.items():
        feature_std = np.std(weights)
        feature_stds.append(feature_std)
        print(f"  {feature}: mean={np.mean(weights):.4f}, std={feature_std:.4f}")

    if feature_stds:
        print(f"Mean standard deviation across features: {np.mean(feature_stds):.4f}")

# -------------------------------
# 6. Summary
# -------------------------------
print("\n--- LIME Analysis Complete ---")
print("Plots saved to Figures/")
