import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ✅ Ensure artifacts folder exists
os.makedirs("artifacts", exist_ok=True)

# 📂 Load processed data
data_path = "artifacts/processed_data.csv"
df = pd.read_csv(data_path)
print("✅ Processed data loaded:", df.shape)

# 🧾 Basic dataset overview
print("\n📊 Data Info:")
print(df.info())

print("\n📈 Statistical Summary:")
print(df.describe())

#  Convert columns that look numeric but may contain strings
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

#  Identify non-numeric columns
non_numeric_cols = df.select_dtypes(include=["object"]).columns
print("\n Non-numeric columns (ignored in correlation):", list(non_numeric_cols))

#  Filter only numeric columns for correlation
numeric_df = df.select_dtypes(include=["number"])

#  Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap (Numeric Features Only)")
plt.tight_layout()
plt.savefig("artifacts/correlation_heatmap.png")
plt.close()

#  EMI Eligibility Distribution
if "emi_eligibility" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x="emi_eligibility", data=df)
    plt.title("EMI Eligibility Distribution")
    plt.savefig("artifacts/emi_eligibility_distribution.png")
    plt.close()

#  Salary vs Maximum EMI Scatter Plot
if "monthly_salary" in df.columns and "max_monthly_emi" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x="monthly_salary", y="max_monthly_emi", data=df, alpha=0.5)
    plt.title("Monthly Salary vs Maximum EMI")
    plt.savefig("artifacts/salary_vs_emi.png")
    plt.close()

#  Credit Score vs EMI Eligibility
if "credit_score" in df.columns and "emi_eligibility" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="emi_eligibility", y="credit_score", data=df)
    plt.title("Credit Score vs EMI Eligibility")
    plt.savefig("artifacts/credit_vs_eligibility.png")
    plt.close()

print("\n EDA completed successfully!")
print(" Visualizations saved in 'artifacts/' folder:")
print(" - correlation_heatmap.png")
print(" - emi_eligibility_distribution.png")
print(" - salary_vs_emi.png")
print(" - credit_vs_eligibility.png")
