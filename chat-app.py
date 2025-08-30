# chat-app.py - Fetal Health Prediction + Advanced Chatbot

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import requests
import openai

# -------------------------------
# OpenAI GPT-3 Setup
# -------------------------------
openai.api_key = "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"

def gpt3_response(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return None

# -------------------------------
# Web Search Fallback
# -------------------------------
def search_web(query):
    try:
        # Simple Google search scraping example (limited)
        # For production use, use SerpAPI or Bing Search API
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&key=YOUR_GOOGLE_API_KEY&cx=YOUR_CX_ID"
        res = requests.get(url).json()
        snippet = res['items'][0]['snippet']
        return snippet
    except:
        return "Sorry, I could not find an answer online."

# -------------------------------
# Load Model and Features
# -------------------------------
model = joblib.load('voting_ensemble_model.pkl')
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Feature importance fallback
try:
    feature_importances = model.estimators_[0].feature_importances_
except:
    feature_importances = np.ones(len(feature_names))/len(feature_names)

# -------------------------------
# Prediction Functions
# -------------------------------
def predict_fetal_health(df):
    X = df[feature_names]
    prediction = model.predict(X)
    probability = model.predict_proba(X).max()
    classes = {1: "Normal", 2: "Suspect", 3: "Pathological"}
    return [classes[p] for p in prediction], probability

def interpret_prediction(pred_class, prob):
    msg = f"The predicted fetal health status is **{pred_class}** with confidence {prob*100:.1f}%.\n"
    if pred_class == "Normal":
        msg += "Fetus appears healthy. Routine monitoring recommended."
    elif pred_class == "Suspect":
        msg += "Some unusual patterns detected. Close monitoring advised."
    else:
        msg += "High risk detected! Immediate medical attention required."
    return msg

def add_medical_notes(pred_class):
    if pred_class == 1:
        return "Normal: Routine monitoring"
    elif pred_class == 2:
        return "Suspect: Close monitoring advised"
    else:
        return "Pathological: Immediate attention required"

# -------------------------------
# Chatbot Response Logic
# -------------------------------
def chatbot_response(user_input, uploaded_data=None, manual_input_df=None):
    user_input_lower = user_input.lower()
    
    # Basic app-specific responses (manual input guidance, prediction interpretation)
    if "hello" in user_input_lower or "hi" in user_input_lower:
        return "Hello! I'm your Fetal Health Assistant Bot. You can upload CTG data or enter manual inputs for predictions."
    
    # Prediction related queries
    if "predict" in user_input_lower or "analyze" in user_input_lower:
        if manual_input_df is not None:
            pred_class, prob = predict_fetal_health(manual_input_df)
            return interpret_prediction(pred_class[0], prob)
        elif uploaded_data is not None:
            pred_class, prob = predict_fetal_health(uploaded_data)
            return interpret_prediction(pred_class[0], prob)
        else:
            return "Upload CSV or enter manual data first for prediction."
    
    # For any other question, ask GPT-3
    gpt_answer = gpt3_response(user_input)
    if gpt_answer:
        return gpt_answer
    else:
        return "Sorry, I could not find an answer."

# -------------------------------
# Streamlit Layout
# -------------------------------
st.set_page_config(page_title="Fetal Health Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Fetal Health Prediction + Advanced Chatbot")

# Sidebar: CSV Upload
st.sidebar.header("Upload CTG Data (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload CSV with CTG features", type=["csv"])
uploaded_data = None
if uploaded_file is not None:
    uploaded_data = pd.read_csv(uploaded_file)
    missing_cols = [col for col in feature_names if col not in uploaded_data.columns]
    if missing_cols:
        st.sidebar.error(f"Missing columns: {missing_cols}")
        uploaded_data = None
    else:
        st.sidebar.success("File uploaded successfully!")
        st.sidebar.write(uploaded_data.head())

# Manual Input
st.header("Manual Input for Single Prediction")
input_data = {}
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
}

for feature in feature_names:
    min_val, max_val = medical_ranges.get(feature, (0, 100))
    col1, col2 = st.columns([3,1])
    with col1:
        slider_val = st.slider(feature, float(min_val), float(max_val), float((min_val+max_val)/2))
    with col2:
        number_val = st.number_input(f"{feature} value", float(min_val), float(max_val), slider_val)
    input_data[feature] = number_val
input_df = pd.DataFrame([input_data])

# Prediction Buttons
st.header("Predictions")
col1, col2 = st.columns(2)

with col1:
    if st.button("Predict Manual Input"):
        pred_class, prob = predict_fetal_health(input_df)
        st.markdown(interpret_prediction(pred_class[0], prob))
        st.info(f"Medical Note: {add_medical_notes(pred_class[0])}")

with col2:
    if uploaded_data is not None and st.button("Predict CSV Upload"):
        preds, prob = predict_fetal_health(uploaded_data)
        uploaded_data['Predicted_Class'] = preds
        uploaded_data['Medical_Note'] = uploaded_data['Predicted_Class'].apply(
            lambda x: add_medical_notes({"Normal":1,"Suspect":2,"Pathological":3}[x])
        )
        st.dataframe(uploaded_data)

        # Pie chart
        st.subheader("Predicted Class Distribution")
        dist = uploaded_data['Predicted_Class'].value_counts().sort_index()
        fig, ax = plt.subplots()
        colors = ['green','orange','red']
        ax.pie(dist, labels=[f"{i} ({dist[i]})" for i in dist.index], colors=colors[:len(dist)], autopct='%1.1f%%')
        st.pyplot(fig)

        # Download predictions
        st.download_button("Download Predictions as CSV", uploaded_data.to_csv(index=False), "fetal_health_predictions.csv", "text/csv")

# Chatbot Section
st.header("Advanced Chatbot Assistant")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Type your message here...")
if st.button("Send") and user_input:
    response = chatbot_response(user_input, uploaded_data=uploaded_data, manual_input_df=input_df)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# Display chat history
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")
