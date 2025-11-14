# src/train_classification.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
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
target_col = "emi_eligibility"

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
# Classification pipeline
# -----------------------------
clf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# -----------------------------
# Train model
# -----------------------------
mlflow.set_experiment("EMI_Classification")
with mlflow.start_run():
    print("🔹 Training Logistic Regression...")
    clf_pipeline.fit(X_train, y_train)
    
    y_pred = clf_pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save model
    os.makedirs("artifacts/models", exist_ok=True)
    joblib.dump(clf_pipeline, "artifacts/models/classification_model.pkl")
    print("💾 Classification model saved!")
