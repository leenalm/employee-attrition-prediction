import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Title
st.title("Employee Attrition Prediction")
st.subheader("Powered by Tuned SVM Model")

# Sidebar inputs
st.sidebar.header("Enter Employee Details")

# Example input fields (use same order and types as training)
Age = st.sidebar.slider("Age", 18, 60, 30)
DistanceFromHome = st.sidebar.slider("Distance From Home (km)", 1, 30, 10)
MonthlyIncome = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000)
JobSatisfaction = st.sidebar.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)
YearsAtCompany = st.sidebar.slider("Years at Company", 0, 40, 5)

# Combine inputs
input_data = np.array([[Age, DistanceFromHome, MonthlyIncome, JobSatisfaction, YearsAtCompany]])

# Scale the input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Attrition"):
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error(" This employee is likely to leave the company.")
    else:
        st.success(" This employee is likely to stay at the company.")
