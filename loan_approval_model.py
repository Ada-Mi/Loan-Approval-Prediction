# Deploying the model

#Import libs
import streamlit as st
import pickle
import numpy as np


# 📌 Load the saved model & scaler

with open("decisiontree.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 📌 Streamlit UI
st.title("🏦 Loan Approval Prediction App")

st.sidebar.header("Enter Loan Applicant Details")

# 📌 User inputs
income = st.sidebar.number_input("Annual Income ($)", min_value=1000, max_value=500000, step=1000)
cibil_score = st.sidebar.number_input("Cibil Score (300-850)", min_value=300, max_value=850, step=10)
loan_amount = st.sidebar.number_input("Loan Amount ($)", min_value=500, max_value=100000, step=500)
loan_term = st.sidebar.selectbox("Loan Term (Years)", [5, 10, 15, 20, 30])
movable_assets = st.sidebar.selectbox("movable_assets", [0, 1, 2, 3, 4])
immovable_assets = st.sidebar.selectbox("immovable_assets", [0, 1, 2, 3, 4])
selfemployed = st.sidebar.selectbox("selfemployed", ["Yes", "No"])
selfemployed = 1 if selfemployed == "yes" else 0
No_of_dependents = st.sidebar.selectbox("no_of_dependents", [0, 1, 2, 3, 4, 5])
education = st.sidebar.selectbox("education", ["yes", "No"])
education = 1 if education == "yes" else 0

                     


# 📌 Make prediction when button is clicked
if st.sidebar.button("Check Loan Approval"):
    # Prepare input data
    user_input = np.array([[income, cibil_score, loan_amount, loan_term, movable_assets, immovable_assets, selfemployed, No_of_dependents, education]])
    user_input_scaled = scaler.transform(user_input)

    # Make prediction
    prediction = model.predict(user_input_scaled)[0]
    probability = model.predict_proba(user_input_scaled)[0][1]  # Probability of approval

    # Display result
if prediction == 1:
    st.success(f"✅ Loan Approved! Probability: {probability:.2f}")
else:
    st.error(f"❌ Loan Rejected. Probability: {probability:.2f}")
