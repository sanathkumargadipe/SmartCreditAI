import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="SmartCreditAI", page_icon="💳", layout="centered")

st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #f8fbff, #eef3f9);
}
.stButton>button {
    background-color: #1f4e79;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-weight: bold;
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

# ===============================
# Load Model
# ===============================

model_data = joblib.load("credit_risk_top8_model.pkl")
model = model_data["model"]
threshold = model_data["threshold"]

# ===============================
# Header
# ===============================

st.title("💳 SmartCreditAI")
st.subheader("Optimized Credit Risk Prediction (Top 8 Features)")
st.markdown("---")

# ===============================
# Input Section
# ===============================

col1, col2 = st.columns(2)

with col1:
    duration = st.number_input("Loan Duration (months)", min_value=1)
    amount = st.number_input("Loan Amount")
    age = st.number_input("Age", min_value=18)
    status = st.selectbox("Account Status", 
                          ["no checking", "little", "moderate", "rich"])

with col2:
    savings = st.selectbox("Savings", 
                           ["no savings", "little", "moderate", "rich"])
    purpose = st.selectbox("Loan Purpose", 
                           ["car", "education", "furniture", "business", "radio/TV", "appliances"])
    property = st.selectbox("Property", 
                            ["real estate", "car", "savings", "other"])
    job = st.selectbox("Job Type", 
                       ["unskilled", "skilled", "highly skilled", "management"])



input_data = pd.DataFrame([{
    "status": status,
    "duration": duration,
    "purpose": purpose,
    "amount": amount,
    "savings": savings,
    "age": age,
    "property": property,
    "job": job
}])

st.markdown("---")


if st.button("Predict Credit Risk"):

    probs = model.predict_proba(input_data)[:, 1]
    probability = probs[0]
    prediction = probability > threshold

    st.subheader("Prediction Result")

    st.progress(float(probability))
    st.write(f"Risk Probability: {round(probability * 100, 2)}%")
    st.write(f"Decision Threshold: {threshold}")

    if prediction:
        st.markdown("<div class='risk-high'>⚠ HIGH RISK CUSTOMER</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='risk-low'>✅ LOW RISK CUSTOMER</div>", unsafe_allow_html=True)

    st.caption("Model: Random Forest | SMOTE Balanced | Auto Feature Selection")
