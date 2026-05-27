import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import lime
import lime.lime_tabular

from data_preparation import FEATURE_COLUMNS, RANDOM_STATE, prepare_data

MODEL_DIR = "Models"

# -------------------------------
# 0. Page Config
# -------------------------------
st.set_page_config(
    page_title="Heart Disease XAI Dashboard",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 Heart Disease Prediction & Explainability Dashboard")
st.markdown("Enter patient data in the sidebar to generate a prediction with SHAP and LIME explanations.")

# -------------------------------
# 1. Load Models & Scaler
# -------------------------------
@st.cache_resource
def load_models():
    lr  = joblib.load(f"{MODEL_DIR}/logistic_regression.pkl")
    rf  = joblib.load(f"{MODEL_DIR}/random_forest.pkl")
    xgb = joblib.load(f"{MODEL_DIR}/xgboost.pkl")
    dt  = joblib.load(f"{MODEL_DIR}/decision_tree.pkl")
    scaler = joblib.load("scaler.pkl")
    return lr, rf, xgb, dt, scaler


@st.cache_data(show_spinner=False)
def load_prepared_data():
    return prepare_data(save_scaler=False, verbose=False)


lr_model, rf_model, xgb_model, dt_model, scaler = load_models()

models = {
    "Logistic Regression": lr_model,
    "Random Forest":       rf_model,
    "XGBoost":             xgb_model,
    "Decision Tree":       dt_model,
}

feature_names = FEATURE_COLUMNS

# -------------------------------
# 2. Plain English Helper
# -------------------------------
FEATURE_LABELS = {
    "age":      "Age",
    "sex":      "Sex",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol":     "Cholesterol",
    "fbs":      "Fasting Blood Sugar",
    "restecg":  "Resting ECG",
    "thalach":  "Maximum Heart Rate",
    "exang":    "Exercise Induced Angina",
    "oldpeak":  "ST Depression",
    "slope":    "ST Slope",
    "ca":       "Number of Blocked Vessels",
    "thal":     "Thalassemia Type",
}

def generate_plain_english(shap_exp, pred, prob, model_name):
    values   = shap_exp.values
    features = shap_exp.feature_names
    data     = shap_exp.data

    sorted_idx   = np.argsort(np.abs(values))[::-1]
    top_features = [(features[i], values[i], data[i]) for i in sorted_idx[:5]]

    # Normalise impact for relative comparison
    max_impact = max(abs(values)) if max(abs(values)) > 0 else 1

    def impact_label(val):
        ratio = abs(val) / max_impact
        if ratio > 0.7:   return "strongly"
        elif ratio > 0.4: return "moderately"
        elif ratio > 0.2: return "slightly"
        else:             return "marginally"

    prediction_text = "**heart disease**" if pred == 1 else "**no heart disease**"
    confidence_text = f"{max(prob, 1 - prob) * 100:.1f}%"

    lines = []
    lines.append(f"The {model_name} model predicts {prediction_text} with {confidence_text} confidence.")
    lines.append("")
    lines.append("**The top factors influencing this prediction are:**")
    lines.append("")

    for feat, val, feat_val in top_features:
        label     = FEATURE_LABELS.get(feat, feat)
        direction = "raises" if val > 0 else "lowers"
        strength  = impact_label(val)

        if feat == "thal":
            val_desc = {3.0: "Normal", 6.0: "Fixed Defect", 7.0: "Reversible Defect"}.get(feat_val, f"{feat_val:.0f}")
        elif feat == "cp":
            val_desc = {1.0: "Typical Angina", 2.0: "Atypical Angina", 3.0: "Non-Anginal Pain", 4.0: "Asymptomatic"}.get(feat_val, f"{feat_val:.0f}")
        elif feat == "sex":
            val_desc = "Male" if feat_val == 1.0 else "Female"
        elif feat == "fbs":
            val_desc = "Yes" if feat_val == 1.0 else "No"
        elif feat == "exang":
            val_desc = "Yes" if feat_val == 1.0 else "No"
        elif feat == "restecg":
            val_desc = {0.0: "Normal", 1.0: "ST-T Abnormality", 2.0: "Left Ventricular Hypertrophy"}.get(feat_val, f"{feat_val:.0f}")
        elif feat == "slope":
            val_desc = {1.0: "Upsloping", 2.0: "Flat", 3.0: "Downsloping"}.get(feat_val, f"{feat_val:.0f}")
        elif feat == "ca":
            val_desc = f"{int(feat_val)} vessel{'s' if feat_val != 1 else ''}"
        else:
            val_desc = f"{feat_val:.1f}"

        lines.append(f"- **{label}** ({val_desc}) {strength} {direction} the risk of heart disease.")

    lines.append("")

    if pred == 1:
        if prob > 0.85:
            lines.append("⚠️ **High Risk:** The model is highly confident this patient has heart disease. Clinical review is strongly recommended.")
        else:
            lines.append("⚠️ **Moderate-High Risk:** The model predicts heart disease but with some uncertainty. Further clinical assessment is advised.")
    else:
        if prob < 0.15:
            lines.append("✅ **Low Risk:** The model is highly confident this patient does not have heart disease.")
        else:
            lines.append("✅ **Low-Moderate Risk:** The model predicts no heart disease but with some uncertainty. Routine monitoring is advised.")

    lines.append("")
    lines.append("*This explanation is generated by an AI system and should be used to support, not replace, clinical judgement.*")

    return "\n".join(lines)


# -------------------------------
# 3. Decision Tree Path Helper
# -------------------------------
def get_decision_path(model, input_df, feature_names):
    """Extract human readable decision path from Decision Tree."""
    feature  = model.tree_.feature
    threshold = model.tree_.threshold

    node_indicator = model.decision_path(input_df)
    node_ids       = node_indicator.indices

    path_lines = []
    for node_id in node_ids:
        if feature[node_id] == -2:
            # Leaf node
            leaf_val   = model.tree_.value[node_id][0]
            pred_class = "Disease" if leaf_val[1] > leaf_val[0] else "No Disease"
            path_lines.append(f"→ **Decision: {pred_class}**")
        else:
            feat_name  = feature_names[feature[node_id]]
            feat_label = FEATURE_LABELS.get(feat_name, feat_name)
            thresh     = threshold[node_id]
            feat_val   = input_df[feat_name].values[0]

            if feat_val <= thresh:
                path_lines.append(f"✅ **{feat_label}** = {feat_val:.1f} ≤ {thresh:.2f} → Go left")
            else:
                path_lines.append(f"❌ **{feat_label}** = {feat_val:.1f} > {thresh:.2f} → Go right")

    return path_lines


# -------------------------------
# 4. Sidebar — Patient Input
# -------------------------------
st.sidebar.header("Patient Data Input")

age      = st.sidebar.slider("Age",                            20,  80,  54)
sex      = st.sidebar.selectbox("Sex",                         options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp       = st.sidebar.selectbox("Chest Pain Type (cp)",        options=[1, 2, 3, 4], format_func=lambda x: {1: "1 - Typical Angina", 2: "2 - Atypical Angina", 3: "3 - Non-Anginal Pain", 4: "4 - Asymptomatic ⚠️"}[x])
trestbps = st.sidebar.slider("Resting Blood Pressure",         80,  200, 130)
chol     = st.sidebar.slider("Cholesterol (mg/dl)",            100, 600, 246)
fbs      = st.sidebar.selectbox("Fasting Blood Sugar > 120",   options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
restecg  = st.sidebar.selectbox("Resting ECG",                 options=[0, 1, 2], format_func=lambda x: {0: "0 - Normal", 1: "1 - ST-T Abnormality", 2: "2 - Left Ventricular Hypertrophy"}[x])
thalach  = st.sidebar.slider("Max Heart Rate Achieved",        60,  220, 150)
exang    = st.sidebar.selectbox("Exercise Induced Angina",     options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak  = st.sidebar.slider("ST Depression (oldpeak)",        0.0, 6.0, 1.0, step=0.1)
slope    = st.sidebar.selectbox("Slope of ST Segment",         options=[1, 2, 3], format_func=lambda x: {1: "1 - Upsloping", 2: "2 - Flat", 3: "3 - Downsloping"}[x])
ca       = st.sidebar.selectbox("Major Vessels Coloured (ca)", options=[0, 1, 2, 3])
thal     = st.sidebar.selectbox("Thalassemia (thal)",          options=[3, 6, 7], format_func=lambda x: {3: "3 - Normal", 6: "6 - Fixed Defect", 7: "7 - Reversible Defect"}[x])

st.sidebar.caption("⚠️ Asymptomatic chest pain does not indicate lower risk.")

input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                         thalach, exang, oldpeak, slope, ca, thal]])
input_df   = pd.DataFrame(input_data, columns=feature_names)

# -------------------------------
# 5. Model Selection
# -------------------------------
st.sidebar.header("Model Selection")
selected_models = st.sidebar.multiselect(
    "Select models to run",
    options=list(models.keys()),
    default=list(models.keys())
)

show_plain_english = st.sidebar.checkbox("Show Plain English Explanations", value=True)
show_technical     = st.sidebar.checkbox("Show Technical Plots (SHAP/LIME)", value=True)

run_button = st.sidebar.button("Run Prediction", type="primary")

# -------------------------------
# 6. Predictions
# -------------------------------
if run_button:
    if not selected_models:
        st.warning("Please select at least one model.")
        st.stop()

    prepared_data = load_prepared_data()
    X_train = prepared_data["X_train"]

    st.header("Prediction Results")

    # --- Model Comparison Table ---
    results = []
    for name in selected_models:
        model = models[name]

        if name == "Logistic Regression":
            input_scaled = scaler.transform(input_df)
            prob = model.predict_proba(input_scaled)[0][1]
            pred = model.predict(input_scaled)[0]
        else:
            prob = model.predict_proba(input_df)[0][1]
            pred = model.predict(input_df)[0]

        results.append({
            "Model":      name,
            "Prediction": "🔴 Disease" if pred == 1 else "🟢 No Disease",
            "Probability": f"{prob:.4f}",
            "Confidence": f"{max(prob, 1 - prob) * 100:.1f}%"
        })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    # -------------------------------
    # 7. Per-Model Explanations
    # -------------------------------
    for name in selected_models:
        model = models[name]

        if name == "Logistic Regression":
            input_for_model = scaler.transform(input_df)
        else:
            input_for_model = input_df

        prob = model.predict_proba(
            input_for_model if name == "Logistic Regression" else input_df
        )[0][1]
        pred = int(model.predict(
            input_for_model if name == "Logistic Regression" else input_df
        )[0])

        st.markdown("---")
        st.subheader(f"🔍 {name} — Explanations")

        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction",  "Disease" if pred == 1 else "No Disease")
        col2.metric("Probability", f"{prob:.4f}")
        col3.metric("Confidence",  f"{max(prob, 1 - prob) * 100:.1f}%")

        # --- Decision Tree Path ---
        if name == "Decision Tree":
            st.markdown("#### 🌳 Decision Path (Transparent Rule Trace)")
            st.caption("This model is fully transparent — you can follow the exact logic it used to reach its decision.")
            path = get_decision_path(model, input_df, feature_names)
            for step in path:
                st.markdown(step)

        # --- Plain English ---
        if show_plain_english:
            st.markdown("#### 💬 Plain English Explanation")

            try:
                if name == "Logistic Regression":
                    pe_explainer = shap.LinearExplainer(model, scaler.transform(X_train))
                    pe_sv        = pe_explainer.shap_values(input_for_model)
                    pe_vals      = pe_sv[0] if isinstance(pe_sv, list) else pe_sv[0]
                    base_val     = pe_explainer.expected_value if np.isscalar(pe_explainer.expected_value) else pe_explainer.expected_value[0]
                    shap_exp_pe  = shap.Explanation(
                        values=pe_vals,
                        base_values=base_val,
                        data=input_df.values[0],
                        feature_names=feature_names
                    )
                else:
                    pe_explainer = shap.TreeExplainer(model)
                    pe_sv        = pe_explainer(input_df)
                    if pe_sv.values.ndim == 3:
                        shap_exp_pe = shap.Explanation(
                            values=pe_sv.values[0, :, 1],
                            base_values=pe_sv.base_values[0, 1],
                            data=input_df.values[0],
                            feature_names=feature_names
                        )
                    else:
                        shap_exp_pe = shap.Explanation(
                            values=pe_sv.values[0],
                            base_values=pe_sv.base_values[0],
                            data=input_df.values[0],
                            feature_names=feature_names
                        )

                plain_text = generate_plain_english(shap_exp_pe, pred, prob, name)
                st.markdown(plain_text)

            except Exception as e:
                st.error(f"Plain English error: {e}")

        # --- Technical Plots ---
        if show_technical and name != "Decision Tree":
            shap_col, lime_col = st.columns(2)

            # SHAP
            with shap_col:
                st.markdown("#### SHAP Explanation")
                try:
                    if name == "Logistic Regression":
                        explainer = shap.LinearExplainer(model, scaler.transform(X_train))
                        sv        = explainer.shap_values(input_for_model)
                        shap_vals = sv[0] if isinstance(sv, list) else sv[0]
                        base_val  = explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[1]
                        shap_exp  = shap.Explanation(
                            values=shap_vals,
                            base_values=base_val,
                            data=input_df.values[0],
                            feature_names=feature_names
                        )
                    else:
                        explainer = shap.TreeExplainer(model)
                        sv        = explainer(input_df)
                        if sv.values.ndim == 3:
                            shap_exp = shap.Explanation(
                                values=sv.values[0, :, 1],
                                base_values=sv.base_values[0, 1],
                                data=input_df.values[0],
                                feature_names=feature_names
                            )
                        else:
                            shap_exp = shap.Explanation(
                                values=sv.values[0],
                                base_values=sv.base_values[0],
                                data=input_df.values[0],
                                feature_names=feature_names
                            )

                    shap.plots.waterfall(shap_exp, max_display=13, show=False)
                    fig = plt.gcf()
                    st.pyplot(fig, use_container_width=False)
                    plt.close()

                except Exception as e:
                    st.error(f"SHAP error: {e}")

            # LIME
            with lime_col:
                st.markdown("#### LIME Explanation")
                try:
                    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                        training_data=X_train.values,
                        feature_names=feature_names,
                        class_names=["No Disease", "Disease"],
                        mode="classification",
                        discretize_continuous=True,
                        random_state=RANDOM_STATE
                    )

                    exp = lime_explainer.explain_instance(
                        input_df.values[0],
                        model.predict_proba if name != "Logistic Regression" else lambda x: lr_model.predict_proba(scaler.transform(x)),
                        num_features=10
                    )

                    fig = exp.as_pyplot_figure()
                    plt.title(f"LIME — {name}")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=False)
                    plt.close()

                except Exception as e:
                    st.error(f"LIME error: {e}")

        # Decision Tree — show SHAP only, no LIME needed
        elif show_technical and name == "Decision Tree":
            st.markdown("#### SHAP Explanation")
            st.caption("SHAP vs built-in Gini agreement: r = 0.993 — SHAP faithfully recovers the tree's own logic.")
            try:
                explainer = shap.TreeExplainer(model)
                sv        = explainer(input_df)

                if sv.values.ndim == 3:
                    shap_exp = shap.Explanation(
                        values=sv.values[0, :, 1],
                        base_values=sv.base_values[0, 1],
                        data=input_df.values[0],
                        feature_names=feature_names
                    )
                else:
                    shap_exp = shap.Explanation(
                        values=sv.values[0],
                        base_values=sv.base_values[0],
                        data=input_df.values[0],
                        feature_names=feature_names
                    )

                shap.plots.waterfall(shap_exp, max_display=13, show=False)
                fig = plt.gcf()
                st.pyplot(fig, use_container_width=False)
                plt.close()

            except Exception as e:
                st.error(f"SHAP error: {e}")

    # -------------------------------
    # 8. Patient Input Summary
    # -------------------------------
    st.markdown("---")
    st.subheader("📋 Patient Input Summary")
    st.dataframe(input_df, use_container_width=True, hide_index=True)

else:
    st.info("👈 Enter patient data in the sidebar and click **Run Prediction** to generate results.")
