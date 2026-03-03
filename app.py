import streamlit as st
import numpy as np
import joblib

# Load saved model
model = joblib.load("credit_risk_model.pkl")

st.title("Credit Risk Prediction System")

st.write("Enter customer details below:")

# Example input fields (simplified demo)
duration = st.number_input("Loan Duration (months)", min_value=4, max_value=72)
amount = st.number_input("Loan Amount")
age = st.number_input("Age")

# NOTE:
# For full version, you must include all 20 features.
# For demo mini-project, you can simplify OR use default values for others.

if st.button("Predict Risk"):
    
    # Create dummy full feature array (replace properly if using full features)
    sample = np.zeros((1, 20))
    
    # Assign important fields
    sample[0][0] = duration
    sample[0][1] = amount
    sample[0][2] = age

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.success("Low Risk Customer ✅")
    else:
        st.error("High Risk Customer ⚠️")