import streamlit as st
import pickle
import numpy as np
import os

# Load model and scaler
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "svm_model.pkl")
scaler_path = os.path.join(current_dir, "scaler.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# App title
st.title(" Employee Attrition Prediction")
st.markdown("Enter employee details in the sidebar to predict whether the employee is likely to **stay or leave**.")

# Sidebar for user input
st.sidebar.header(" Enter Employee Details")

# Widgets for input features
Age = st.sidebar.slider("Age", 18, 60, 30)
DistanceFromHome = st.sidebar.slider("Distance From Home (km)", 1, 30, 10)
MonthlyIncome = st.sidebar.number_input("Monthly Income (USD)", min_value=1000, max_value=20000, value=5000, step=500)
JobSatisfaction = st.sidebar.slider("Job Satisfaction (1 = Low, 4 = High)", 1, 4, 3)
YearsAtCompany = st.sidebar.slider("Years at Company", 0, 40, 5)

# Format input for model
input_data = np.array([[Age, DistanceFromHome, MonthlyIncome, JobSatisfaction, YearsAtCompany]])
input_scaled = scaler.transform(input_data)

# Predict on button click
if st.button(" Predict Attrition"):
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1] * 100  # Likelihood of leaving

    if prediction[0] == 1:
        st.error(f" This employee is **likely to leave** the company.\n\nAttrition Probability: **{probability:.2f}%**")
    else:
        st.success(f" This employee is **likely to stay** at the company.\n\nAttrition Probability: **{100 - probability:.2f}%**")
