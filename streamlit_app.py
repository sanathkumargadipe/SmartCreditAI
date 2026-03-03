import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="SmartCreditAI",
    page_icon="💳",
    layout="centered"
)

st.markdown("""
    <style>
        body {
            background-color: #f4f6f9;
        }
        .main {
            background: linear-gradient(to right, #f8fbff, #eef3f9);
            padding: 20px;
            border-radius: 15px;
        }
        .stButton>button {
            background-color: #1f4e79;
            color: white;
            border-radius: 8px;
            height: 3em;
            width: 100%;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #163a5f;
            color: white;
        }
        .risk-high {
            color: white;
            background-color: #c0392b;
            padding: 15px;
            border-radius: 10px;
            font-weight: bold;
            text-align: center;
        }
        .risk-low {
            color: white;
            background-color: #27ae60;
            padding: 15px;
            border-radius: 10px;
            font-weight: bold;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

model_data = joblib.load("credit_risk_research_model.pkl")
model = model_data["model"]
threshold = model_data["threshold"]

st.markdown("<h1 style='text-align: center; color: #1f4e79;'>SmartCreditAI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>Advanced Credit Risk Assessment System</h4>", unsafe_allow_html=True)
st.markdown("---")

st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    duration = st.number_input("Loan Duration (months)", min_value=1)
    credit_amount = st.number_input("Loan Amount")
    age = st.number_input("Age")

with col2:
    housing = st.selectbox("Housing", ["own", "rent", "free"])
    job = st.selectbox("Job", ["unskilled", "skilled", "management"])
    saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich"])
    checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich"])
    purpose = st.selectbox("Loan Purpose", ["car", "education", "furniture", "business"])

# ⚠ Add remaining features exactly as in dataset
# Make sure column names match training dataset

input_data = {
    "duration": duration,
    "credit_amount": credit_amount,
    "age": age,
    "housing": housing,
    "job": job,
    "saving_accounts": saving_accounts,
    "checking_account": checking_account,
    "purpose": purpose
}

input_df = pd.DataFrame([input_data])

st.markdown("---")

if st.button("Predict Credit Risk"):

    probs = model.predict_proba(input_df)[:, 1]
    probability = probs[0]
    prediction = (probability > threshold)

    st.subheader("Prediction Results")

    # Probability Progress Bar
    st.progress(float(probability))

    st.write(f"### Risk Probability: {round(probability*100, 2)}%")
    st.write(f"Model Threshold: {threshold}")

    if prediction:
        st.markdown("<div class='risk-high'>⚠ HIGH RISK CUSTOMER</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='risk-low'>✅ LOW RISK CUSTOMER</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Model: Ensemble (Random Forest + XGBoost) | SMOTE Balanced | Cross-Validated")

