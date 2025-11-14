import streamlit as st
import joblib
import pandas as pd
import os
import sys

# Fix working directory for Streamlit Cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)


# -------------------------
# Load The Full Pipeline
# -------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("artifacts/emi_pipeline.joblib")

pipeline = load_pipeline()

st.title("💰 EMI Eligibility & Prediction App")
st.write("Fill the details below to check EMI eligibility and estimated EMI amount.")


# -------------------------
# Persistent Inputs Function
# -------------------------
def user_inputs():

    st.subheader("📌 Enter Your Details")

    age = st.number_input("Age", 18, 80, 30, key="age")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender")
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], key="marital")
    education = st.selectbox("Education", ["Graduate", "Post-Graduate", "10th", "12th", "Other"], key="edu")
    monthly_salary = st.number_input("Monthly Salary", 0, 2000000, 50000, key="salary")
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business", "Student"], key="emp_type")
    years_of_employment = st.number_input("Years of Employment", 0, 60, 2, key="exp_years")
    company_type = st.selectbox("Company Type", ["Private", "Government", "Startup", "Other"], key="company_type")
    house_type = st.selectbox("House Type", ["Owned", "Rented", "Hostel", "Company Provided"], key="house_type")
    monthly_rent = st.number_input("Monthly Rent", 0, 200000, 5000, key="rent")
    family_size = st.number_input("Family Size", 1, 20, 4, key="family")
    dependents = st.number_input("Dependents", 0, 10, 1, key="dependents")
    school_fees = st.number_input("School Fees", 0, 200000, 0, key="school")
    college_fees = st.number_input("College Fees", 0, 200000, 0, key="college")
    travel_expenses = st.number_input("Travel Expenses", 0, 100000, 2000, key="travel")
    groceries_utilities = st.number_input("Groceries & Utilities", 0, 100000, 5000, key="grocery")
    other_monthly_expenses = st.number_input("Other Monthly Expenses", 0, 100000, 2000, key="other_exp")
    existing_loans = st.number_input("Existing Loans Count", 0, 10, 0, key="loans")
    current_emi_amount = st.number_input("Current Total EMI", 0, 200000, 0, key="current_emi")
    credit_score = st.number_input("Credit Score", 300, 900, 700, key="credit")
    bank_balance = st.number_input("Bank Balance", 0, 5000000, 10000, key="balance")
    emergency_fund = st.number_input("Emergency Fund", 0, 5000000, 5000, key="fund")
    emi_scenario = st.selectbox("EMI Scenario", ["New EMI", "Refinance"], key="scenario")
    requested_amount = st.number_input("Requested Loan Amount", 1000, 5000000, 200000, key="loan_amount")
    requested_tenure = st.number_input("Requested Tenure (Months)", 1, 360, 60, key="tenure")

    data = {
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": monthly_salary,
        "employment_type": employment_type,
        "years_of_employment": years_of_employment,
        "company_type": company_type,
        "house_type": house_type,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure
    }

    return pd.DataFrame([data])


# -------------------------
# Show Inputs First
# -------------------------
user_df = user_inputs()


# -------------------------
# Prediction Button
# -------------------------
st.write("---")

if st.button("🔍 Predict EMI Eligibility & Amount"):
    try:
        y_pred = pipeline.predict(user_df)
        emi_amount = float(y_pred[0])

        eligibility = "Eligible" if emi_amount < user_df["monthly_salary"][0] * 0.45 else "Not Eligible"

        st.success(f"🏦 **Eligibility:** {eligibility}")
        st.info(f"💳 **Estimated EMI:** ₹{emi_amount:,.2f}")

    except Exception as e:
        st.error(f" Prediction failed: {str(e)}")
