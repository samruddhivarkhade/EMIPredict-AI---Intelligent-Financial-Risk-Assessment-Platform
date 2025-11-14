# src/train_models.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = "artifacts"
MODEL_DIR = "artifacts/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Load train/test datasets
# -----------------------------
print("📂 Loading train/test datasets...")
train_data = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
test_data = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))
print("✅ Data loaded!")

# -----------------------------
# Split features & targets
# -----------------------------
# Classification target
y_train_class = train_data["emi_eligibility"].map({"Eligible": 1, "Not_Eligible": 0})
y_test_class = test_data["emi_eligibility"].map({"Eligible": 1, "Not_Eligible": 0})

# Regression target
y_train_reg = train_data["emi_amount"]
y_test_reg = test_data["emi_amount"]

# Features
X_train = train_data.drop(columns=["emi_eligibility", "emi_amount"])
X_test = test_data.drop(columns=["emi_eligibility", "emi_amount"])

# -----------------------------
# Handle missing values
# -----------------------------
# Numeric columns
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
imputer_num = SimpleImputer(strategy="median")
X_train[numeric_cols] = imputer_num.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = imputer_num.transform(X_test[numeric_cols])
joblib.dump(imputer_num, os.path.join(MODEL_DIR, "imputer_numeric.joblib"))

# Categorical columns (if any)
categorical_cols = X_train.select_dtypes(include="object").columns.tolist()
if categorical_cols:
    imputer_cat = SimpleImputer(strategy="most_frequent")
    X_train[categorical_cols] = imputer_cat.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = imputer_cat.transform(X_test[categorical_cols])
    joblib.dump(imputer_cat, os.path.join(MODEL_DIR, "imputer_categorical.joblib"))

# -----------------------------
# Feature scaling
# -----------------------------
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

# -----------------------------
# Train classification models
# -----------------------------
print("🔹 Training Classification Models...")

models_class = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models_class.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train_class)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test_class, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))

# -----------------------------
# Train regression models
# -----------------------------
print("🔹 Training Regression Models...")

models_reg = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models_reg.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train_reg)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test_reg, y_pred, squared=False)
    print(f"{name} RMSE: {rmse:.4f}")
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))

print("✅ All models trained and saved successfully!")
