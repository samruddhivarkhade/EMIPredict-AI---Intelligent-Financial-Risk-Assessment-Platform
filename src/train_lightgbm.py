# train_lightgbm.py

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------
# 1. Load processed dataset
# ---------------------------------------------------
print("📂 Loading processed data...")
df = pd.read_csv("artifacts/processed_data.csv")

print("Shape before column removal:", df.shape)

# ---------------------------------------------------
# 2. Remove unwanted columns
# ---------------------------------------------------
REMOVE_COLS = ["Unnamed: 27", "max_monthly_emi", "emi_eligibility"]  # target is requested_amount or EMI?
df.drop(columns=[c for c in REMOVE_COLS if c in df.columns], inplace=True)

print("Shape after column removal:", df.shape)

# ---------------------------------------------------
# 3. Define target + features
# ---------------------------------------------------
TARGET = "requested_amount"        # <<–– CHANGE IF NEEDED
if TARGET not in df.columns:
    raise ValueError(f"❌ Target column '{TARGET}' not found in dataset!")

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ---------------------------------------------------
# 4. Identify numeric + categorical columns
# ---------------------------------------------------
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("Numeric Columns:", numeric_cols)
print("Categorical Columns:", categorical_cols)

# ---------------------------------------------------
# 5. Build preprocessing pipeline
# ---------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# ---------------------------------------------------
# 6. Build LightGBM Model Pipeline
# ---------------------------------------------------
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    objective="regression",
    boosting_type="gbdt"
)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", model)
])

# ---------------------------------------------------
# 7. Train-test split
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🚀 Training model...")
pipeline.fit(X_train, y_train)

# ---------------------------------------------------
# 8. Evaluate
# ---------------------------------------------------
preds = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)

print(f"📊 MAE:  {mae:.4f}")
print(f"📊 RMSE: {rmse:.4f}")

# ---------------------------------------------------
# 9. Save pipeline + model
# ---------------------------------------------------
os.makedirs("artifacts", exist_ok=True)

joblib.dump(pipeline, "artifacts/emi_pipeline.joblib")
print("💾 Saved pipeline to artifacts/emi_pipeline.joblib")
