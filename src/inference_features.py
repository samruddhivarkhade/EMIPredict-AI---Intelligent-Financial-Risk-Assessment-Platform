# src/inference_features.py

import pandas as pd

def preprocess_user_input(data: pd.DataFrame) -> pd.DataFrame:
   
    df = data.copy()

    # Ensure numeric types
    numeric_cols = [
        "age", "monthly_salary", "years_of_employment", "monthly_rent",
        "family_size", "dependents", "school_fees", "college_fees",
        "travel_expenses", "groceries_utilities", "other_monthly_expenses",
        "existing_loans", "current_emi_amount", "credit_score",
        "bank_balance", "emergency_fund", "requested_amount",
        "requested_tenure"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Derived features
    df["debt_to_income"] = df["current_emi_amount"] / (df["monthly_salary"] + 1e-6)
    df["expenses_to_income"] = (
        df["monthly_rent"] + df["school_fees"] + df["college_fees"] +
        df["travel_expenses"] + df["groceries_utilities"] + df["other_monthly_expenses"]
    ) / (df["monthly_salary"] + 1e-6)
    df["affordability_ratio"] = df["requested_amount"] / ((df["monthly_salary"] + 1e-6) * df["requested_tenure"])
    df["risk_score"] = df["existing_loans"] * 0.4 + df["dependents"] * 0.2 + df["debt_to_income"] * 0.4

    # Same categorical columns as training
    categorical_cols = [
        "gender", "marital_status", "education", "employment_type",
        "company_type", "house_type", "emi_scenario"
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # One-hot encoding
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df
