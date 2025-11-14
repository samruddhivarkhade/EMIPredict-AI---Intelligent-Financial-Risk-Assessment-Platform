'''# src/feature_engineering.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths
# -----------------------------
data_path = "artifacts/processed_data.csv"
train_path = "artifacts/train_data.csv"
test_path = "artifacts/test_data.csv"

# -----------------------------
# Load processed data
# -----------------------------
print("📂 Loading processed data...")
df = pd.read_csv(data_path)
print(f"✅ Data loaded! Shape: {df.shape}")

# -----------------------------
# Ensure numeric columns are clean
# -----------------------------
numeric_cols = [
    "age", "monthly_salary", "years_of_employment", "monthly_rent",
    "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "existing_loans", "current_emi_amount", "credit_score",
    "bank_balance", "emergency_fund", "requested_amount",
    "requested_tenure", "max_monthly_emi"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing numeric values with median
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# -----------------------------
# Derived / Feature Engineering
# -----------------------------
print("🛠️ Creating derived features...")

# Debt-to-income ratio
df["debt_to_income"] = df["current_emi_amount"] / (df["monthly_salary"] + 1e-6)

# Expense-to-income ratio
df["expenses_to_income"] = (
    df["monthly_rent"] + df["school_fees"] + df["college_fees"] +
    df["travel_expenses"] + df["groceries_utilities"] + df["other_monthly_expenses"]
) / (df["monthly_salary"] + 1e-6)

# Affordability ratio
df["affordability_ratio"] = df["requested_amount"] / ((df["monthly_salary"] + 1e-6) * df["requested_tenure"])

# Risk score based on existing loans and dependents
df["risk_score"] = df["existing_loans"] * 0.4 + df["dependents"] * 0.2 + df["debt_to_income"] * 0.4

# -----------------------------
# Target columns
# -----------------------------
# Classification target: EMI Eligibility
df["emi_eligibility"] = df["current_emi_amount"] <= df["monthly_salary"] * 0.5
df["emi_eligibility"] = df["emi_eligibility"].map({True: "Eligible", False: "Not_Eligible"})

# Regression target: EMI Amount
df["emi_amount"] = df["current_emi_amount"]

# -----------------------------
# Encode categorical variables
# -----------------------------
categorical_cols = [
    "gender", "marital_status", "education", "employment_type",
    "company_type", "house_type", "emi_scenario"
]

print("🔢 Encoding categorical columns...")
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str)

# One-hot encoding for modeling (skip target columns)
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# -----------------------------
# Train-test split
# -----------------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# -----------------------------
# Save processed datasets
# -----------------------------
os.makedirs("artifacts", exist_ok=True)
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"💾 Train and test data saved to {train_path} and {test_path}")'''

# src/feature_engineering.py

import pandas as pd
import numpy as np

# -------------------------------------
#  Helper: Clean and preprocess new input
# -------------------------------------
def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess incoming user data for model inference.
    Performs feature engineering, encoding, and scaling alignment.
    """
    # ----------------------------
    # Ensure required columns exist
    # ----------------------------
    required_cols = [
        "age", "gender", "marital_status", "education", "monthly_salary",
        "employment_type", "years_of_employment", "company_type", "house_type",
        "monthly_rent", "family_size", "dependents", "school_fees", "college_fees",
        "travel_expenses", "groceries_utilities", "other_monthly_expenses",
        "existing_loans", "current_emi_amount", "credit_score", "bank_balance",
        "emergency_fund", "emi_scenario", "requested_amount", "requested_tenure"
    ]

    # Add missing columns with default values
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # ----------------------------
    # Numeric cleanup
    # ----------------------------
    numeric_cols = [
        "age", "monthly_salary", "years_of_employment", "monthly_rent",
        "family_size", "dependents", "school_fees", "college_fees",
        "travel_expenses", "groceries_utilities", "other_monthly_expenses",
        "existing_loans", "current_emi_amount", "credit_score",
        "bank_balance", "emergency_fund", "requested_amount", "requested_tenure"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ----------------------------
    # Derived Features
    # ----------------------------
    df["debt_to_income"] = df["current_emi_amount"] / (df["monthly_salary"] + 1e-6)

    df["expenses_to_income"] = (
        df["monthly_rent"] + df["school_fees"] + df["college_fees"] +
        df["travel_expenses"] + df["groceries_utilities"] + df["other_monthly_expenses"]
    ) / (df["monthly_salary"] + 1e-6)

    df["affordability_ratio"] = df["requested_amount"] / (
        (df["monthly_salary"] + 1e-6) * df["requested_tenure"]
    )

    df["risk_score"] = (
        df["existing_loans"] * 0.4 +
        df["dependents"] * 0.2 +
        df["debt_to_income"] * 0.4
    )

    # ----------------------------
    # Encode categoricals
    # ----------------------------
    categorical_cols = [
        "gender", "marital_status", "education", "employment_type",
        "company_type", "house_type", "emi_scenario"
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("Unknown")

    # One-hot encode
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)



    return df


# -------------------------------------
#  (Optional) Test this file manually
# -------------------------------------
if __name__ == "__main__":
    sample = pd.DataFrame([{
        "age": 30,
        "gender": "Male",
        "marital_status": "Single",
        "education": "Graduate",
        "monthly_salary": 60000,
        "employment_type": "Full-time",
        "years_of_employment": 5,
        "company_type": "Private",
        "house_type": "Rented",
        "monthly_rent": 8000,
        "family_size": 3,
        "dependents": 1,
        "school_fees": 2000,
        "college_fees": 0,
        "travel_expenses": 3000,
        "groceries_utilities": 10000,
        "other_monthly_expenses": 4000,
        "existing_loans": 1,
        "current_emi_amount": 10000,
        "credit_score": 750,
        "bank_balance": 50000,
        "emergency_fund": 20000,
        "emi_scenario": "Personal",
        "requested_amount": 500000,
        "requested_tenure": 24
    }])

    processed = preprocess_input(sample)
    print("✅ Processed sample shape:", processed.shape)
    print(processed.head())
