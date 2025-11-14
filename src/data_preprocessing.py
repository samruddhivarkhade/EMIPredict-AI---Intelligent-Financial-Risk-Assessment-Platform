# src/data_preprocessing.py

import pandas as pd
import numpy as np
import os

# -----------------------------
# Paths
# -----------------------------
raw_data_path = "data/financial_data.csv"   # your raw CSV
processed_data_path = "artifacts/processed_data.csv"

# -----------------------------
# Load raw data
# -----------------------------
print(" Loading raw data...")
df = pd.read_csv(raw_data_path)
print(f" Raw data loaded! Shape: {df.shape}")

# -----------------------------
# Columns to treat as numeric
# -----------------------------
numeric_cols = [
    "age", "monthly_salary", "years_of_employment", "monthly_rent",
    "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "existing_loans", "current_emi_amount", "credit_score",
    "bank_balance", "emergency_fund", "requested_amount",
    "requested_tenure"
]

# -----------------------------
# Clean numeric columns
# -----------------------------
print(" Cleaning numeric columns...")
for col in numeric_cols:
    if col in df.columns:
        # Remove extra dots, commas, spaces
        df[col] = df[col].astype(str).str.replace(r"[^\d.]", "", regex=True)
        # Convert to numeric, coerce errors to NaN
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill NaN values with median
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# -----------------------------
# Handle categorical columns
# -----------------------------
categorical_cols = [
    "gender", "marital_status", "education", "employment_type",
    "company_type", "house_type", "emi_scenario", "emi_eligibility"
]

print(" Cleaning categorical columns...")
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().replace("nan", "Unknown")

# -----------------------------
# Remove duplicates
# -----------------------------
df = df.drop_duplicates()
print(f" Data shape after removing duplicates: {df.shape}")

# -----------------------------
# Save processed data
# -----------------------------
os.makedirs("artifacts", exist_ok=True)
df.to_csv(processed_data_path, index=False)
print(f" Processed data saved to {processed_data_path}")
