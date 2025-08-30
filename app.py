# app.py - Fully Polished Fetal Health Prediction App (Fixed)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model and features
model = joblib.load('voting_ensemble_model.pkl')
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Try to get feature importances (from Random Forest inside ensemble)
try:
    feature_importances = model.estimators_[0].feature_importances_
except:
    feature_importances = np.ones(len(feature_names)) / len(feature_names)  # fallback

# App Layout
st.set_page_config(page_title="Fetal Health Prediction", page_icon="🤰", layout="wide")
st.title("Fetal Health Prediction Portal")
st.write("Predict fetal health from CTG features. You can enter data manually or upload a CSV file for batch predictions.")

# Input Section
st.header("Manual Input")

medical_ranges = {
    "baseline value": (80, 180),
    "accelerations": (0, 10),
    "fetal_movement": (0, 10),
    "uterine_contractions": (0, 10),
    "light_decelerations": (0, 5),
    "severe_decelerations": (0, 5),
    "prolongued_decelerations": (0, 5),
    "abnormal_short_term_variability": (0, 10),
    "mean_value_of_short_term_variability": (0, 10),
    "histogram_width": (0, 200),
    # Add remaining features as needed
}

input_data = {}
for feature in feature_names:
    min_val, max_val = medical_ranges.get(feature, (0, 100))
    col1, col2 = st.columns([3,1])
    with col1:
        slider_val = st.slider(feature, min_value=float(min_val), max_value=float(max_val), value=float((min_val+max_val)/2))
    with col2:
        number_val = st.number_input(f"{feature} (type value)", min_value=float(min_val), max_value=float(max_val), value=slider_val)
    final_val = number_val
    input_data[feature] = final_val
    if feature in medical_ranges:
        if final_val < medical_ranges[feature][0] or final_val > medical_ranges[feature][1]:
            st.warning(f"{feature} is outside normal range!")

input_df = pd.DataFrame([input_data])

# Prediction Section (Manual Input)
st.header("Prediction")
if st.button("Predict Manual Input"):
    pred_class = model.predict(input_df)[0]

    # Color-coded output + medical notes
    if pred_class == 1:
        st.success(f"Predicted Fetal Health Class: {pred_class} → Normal ✅")
        st.info("Medical Note: Fetal health is normal. No immediate risk detected. Routine monitoring recommended.")
    elif pred_class == 2:
        st.warning(f"Predicted Fetal Health Class: {pred_class} → Suspect ⚠️")
        st.info("Medical Note: Fetal health is suspect. Close monitoring is advised, further tests may be required.")
    else:
        st.error(f"Predicted Fetal Health Class: {pred_class} → Pathological ❌")
        st.info("Medical Note: Fetal health is pathological. Immediate medical attention required!")

# Batch Prediction Section
st.header("Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV with CTG features", type=["csv"])
if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    missing_cols = [col for col in feature_names if col not in batch_df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        batch_input = batch_df[feature_names]
        batch_preds = model.predict(batch_input)
        batch_df['Predicted_Class'] = batch_preds

        # Add medical notes for batch predictions
        def add_medical_notes(row):
            if row['Predicted_Class'] == 1:
                return "Normal: Routine monitoring"
            elif row['Predicted_Class'] == 2:
                return "Suspect: Close monitoring advised"
            else:
                return "Pathological: Immediate attention required"
        batch_df['Medical_Note'] = batch_df.apply(add_medical_notes, axis=1)

        st.subheader("Batch Predictions")
        st.dataframe(batch_df)

        # Pie chart for class distribution
        st.subheader("Predicted Class Distribution")
        dist = batch_df['Predicted_Class'].value_counts().sort_index()
        fig, ax = plt.subplots()
        colors = ['green','orange','red']
        ax.pie(dist, labels=[f"{i} ({dist[i]})" for i in dist.index], colors=colors[:len(dist)], autopct='%1.1f%%')
        st.pyplot(fig)

        # Download button
        st.download_button(
            label="Download Predictions as CSV",
            data=batch_df.to_csv(index=False),
            file_name="fetal_health_predictions.csv",
            mime="text/csv"
        )

# Feature Importance Visualization
st.header("Feature Importance")
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10)
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=fi_df, palette='Set2', ax=ax)
ax.set_title("Top 10 Feature Importance")
st.pyplot(fig)
