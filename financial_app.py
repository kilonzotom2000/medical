# Import libraries
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Function to compute compound interest
def future_value(current_savings, annual_rate, time_years, compounding_periods=12):
    return current_savings * (1 + annual_rate / compounding_periods) ** (compounding_periods * time_years)

# Function to adjust target for inflation
def adjust_for_inflation(target_amount, inflation_rate, time_years):
    return target_amount * (1 + inflation_rate) ** time_years

# Load the trained model
model = joblib.load("readmission_model.pkl")

# Streamlit app
st.title("Financial Target Contribution Calculator")
st.write("Calculate how much you need to contribute monthly to reach your financial goal.")

# User inputs
current_savings = st.number_input("Current Savings ($)", min_value=0.0, value=5000.0, step=100.0)
target_amount = st.number_input("Target Amount ($)", min_value=0.0, value=50000.0, step=1000.0)
time_to_target = st.number_input("Time to Target (months)", min_value=1, value=24, step=1)
interest_rate = st.number_input("Annual Interest Rate (%)", min_value=0.0, max_value=100.0, value=5.0) / 100
inflation_rate = st.number_input("Annual Inflation Rate (%)", min_value=0.0, max_value=100.0, value=2.0) / 100
income = st.number_input("Annual Income ($)", min_value=0.0, value=60000.0, step=1000.0)
monthly_expenses = st.number_input("Monthly Expenses ($)", min_value=0.0, value=2000.0, step=100.0)
monthly_debt = st.number_input("Monthly Debt Payments ($)", min_value=0.0, value=500.0, step=100.0)

# Feature engineering based on user inputs
future_savings = future_value(current_savings, interest_rate, time_to_target / 12)
adjusted_target = adjust_for_inflation(target_amount, inflation_rate, time_to_target / 12)
available_income = income - (monthly_expenses + monthly_debt) * 12

# Prepare input features for the model
input_data = pd.DataFrame({
    "future_savings": [future_savings],
    "adjusted_target": [adjusted_target],
    "time_to_target": [time_to_target],
    "interest_rate": [interest_rate],
    "available_income": [available_income]
})

# Predict the monthly contribution
if st.button("Calculate Contribution"):
    predicted_contribution = model.predict(input_data)[0]
    st.success(f"Estimated Monthly Contribution: ${predicted_contribution:.2f}")

# Footer
st.write("Developed using Machine Learning and Streamlit.")
