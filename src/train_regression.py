# src/train_regression.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import mlflow

# -----------------------------
# Load train/test datasets
# -----------------------------
train_path = "artifacts/train_data.csv"
test_path = "artifacts/test_data.csv"

print("📂 Loading train/test datasets...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print("✅ Data loaded!")

# -----------------------------
# Target and features
# -----------------------------
target_col = "emi_amount"

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# -----------------------------
# Identify numeric and categorical features
# -----------------------------
numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

# -----------------------------
# Preprocessing pipeline
# -----------------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# -----------------------------
# Regression pipeline
# -----------------------------
reg_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# -----------------------------
# Train model
# -----------------------------
mlflow.set_experiment("EMI_Regression")
with mlflow.start_run():
    print("🔹 Training Linear Regression...")
    reg_pipeline.fit(X_train, y_train)
    
    y_pred = reg_pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"✅ MSE: {mse:.4f} | R2: {r2:.4f}")
    
    # Save model
    os.makedirs("artifacts/models", exist_ok=True)
    joblib.dump(reg_pipeline, "artifacts/models/regression_model.pkl")
    print("💾 Regression model saved!")
