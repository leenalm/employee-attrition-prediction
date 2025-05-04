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
st.markdown("Enter employee details to predict whether the employee is likely to **stay or leave**.")

# Sidebar for user input
st.sidebar.header(" Enter Employee Details")

# Widgets for 10 input features
Age = st.sidebar.slider("Age", 18, 60, 30)
DistanceFromHome = st.sidebar.slider("Distance From Home (km)", 1, 30, 10)
MonthlyIncome = st.sidebar.number_input("Monthly Income (USD)", 1000, 20000, 5000, step=500)
JobSatisfaction = st.sidebar.slider("Job Satisfaction (1 = Low, 4 = High)", 1, 4, 3)
YearsAtCompany = st.sidebar.slider("Years at Company", 0, 40, 5)
TotalWorkingYears = st.sidebar.slider("Total Working Years", 0, 40, 10)
NumCompaniesWorked = st.sidebar.slider("Number of Companies Worked", 0, 10, 2)
OverTime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
EnvironmentSatisfaction = st.sidebar.slider("Environment Satisfaction (1 = Low, 4 = High)", 1, 4, 3)
WorkLifeBalance = st.sidebar.slider("Work-Life Balance (1 = Low, 4 = High)", 1, 4, 3)

# Convert categorical value to numeric
OverTime_binary = 1 if OverTime == "Yes" else 0

# Combine into array
input_data = np.array([[Age, DistanceFromHome, MonthlyIncome, JobSatisfaction,
                        YearsAtCompany, TotalWorkingYears, NumCompaniesWorked,
                        OverTime_binary, EnvironmentSatisfaction, WorkLifeBalance]])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict on button click
if st.button(" Predict Attrition"):
    prediction = model.predict(input_scaled)
    
    # Optional: show confidence if model supports probability
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_scaled)[0][1] * 100  # probability of leaving
    else:
        probability = None

    if prediction[0] == 1:
        st.error(" This employee is **likely to leave** the company.")
        if probability:
            st.write(f"Attrition Probability: **{probability:.2f}%**")
    else:
        st.success(" This employee is **likely to stay** at the company.")
        if probability:
            st.write(f"Attrition Probability: **{100 - probability:.2f}%**")
