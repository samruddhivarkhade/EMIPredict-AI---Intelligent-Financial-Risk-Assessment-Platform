# EMIPredict-AI - Intelligent Financial Risk Assessment Platform

EMI Predict AI is a machine learning–based financial assessment tool that predicts:

🔹 Loan Eligibility (Eligible / Not Eligible)
🔹 Maximum Monthly EMI a user can safely afford

This project combines clean ML engineering, a modular pipeline, and an interactive Streamlit web interface, making it ideal for fintech solutions, academic projects, and real-world deployments.

 Key Features
 1. Loan Eligibility Prediction

Determines whether a person qualifies for a loan based on:

Income & employment type
Credit score
Monthly expenses
Existing EMIs
Dependents & financial obligations

 2. Maximum EMI Prediction
A regression model estimates the safest EMI amount, ensuring responsible lending.

 3. Streamlit Web App

Minimalistic UI
Real-time predictions
Validated inputs
Fully automated preprocessing

 4. Modular ML Pipeline

Clean preprocessing (handling categorical & numerical data)
Encoders + Scalers
LightGBM / RandomForest models
Saved pipelines for deployment (joblib)

 Tech Stack

Machine Learning:
Python, Pandas, NumPy, Scikit-Learn, LightGBM

App & Deployment:
Streamlit, Joblib

⚙️ Installation

Dataset: https://drive.google.com/file/d/1C7tcEdnRIlxwIsFnsN6F0jkpU1FRlieS/view?usp=sharing

1️⃣ Clone the repo
git clone https://github.com/samruddhivarkhade/emi-predict-ai.git
cd emi-predict-ai

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the app
streamlit run app.py
