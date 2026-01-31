import streamlit as st
import joblib
import numpy as np
import os

st.title("💳 Credit Wise Loan Prediction")

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

# Load model + scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.write("Fill the details below to check loan approval:")

# ================= INPUT FIELDS =================
Applicant_Income = st.number_input("Applicant Income", min_value=0.0)
Coapplicant_Income = st.number_input("Coapplicant Income", min_value=0.0)
Age = st.number_input("Age", min_value=18, max_value=100)
Dependents = st.number_input("Number of Dependents", min_value=0, max_value=10)
Credit_Score = st.number_input("Credit Score", min_value=300, max_value=900)
Existing_Loans = st.number_input("Number of Existing Loans", min_value=0, max_value=10)
DTI_Ratio = st.number_input("DTI Ratio (%)", min_value=0.0)
Savings = st.number_input("Savings (₹)", min_value=0.0)
Collateral_Value = st.number_input("Collateral Value", min_value=0.0)
Loan_Amount = st.number_input("Loan Amount", min_value=0.0)
Loan_Term = st.number_input("Loan Term (months)", min_value=1, max_value=360)
Education_Level = st.selectbox("Education Level", ["0", "1"])  # same as your encoding

Employment_Status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
Marital_Status_Single = st.selectbox("Marital Status", ["Married", "Single"])
Loan_Purpose = st.selectbox("Loan Purpose", ["Car", "Education", "Home", "Personal"])
Property_Area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
Employer_Category = st.selectbox("Employer Category", ["Government", "MNC", "Private", "Unemployed"])
Gender_Male = st.selectbox("Gender", ["Female", "Male"])


# ================= PREDICT BUTTON =================
if st.button("Predict Loan Approval"):
    
    features = [
        float(Applicant_Income),
        float(Coapplicant_Income),
        float(Age),
        float(Dependents),
        float(Credit_Score),
        float(Existing_Loans),
        float(DTI_Ratio),
        float(Savings),
        float(Collateral_Value),
        float(Loan_Amount),
        float(Loan_Term),
        int(Education_Level),

        # Employment Status
        1 if Employment_Status == "Salaried" else 0,
        1 if Employment_Status == "Self-employed" else 0,
        1 if Employment_Status == "Unemployed" else 0,

        1 if Marital_Status_Single == "Single" else 0,

        # Loan Purpose
        1 if Loan_Purpose == "Car" else 0,
        1 if Loan_Purpose == "Education" else 0,
        1 if Loan_Purpose == "Home" else 0,
        1 if Loan_Purpose == "Personal" else 0,

        # Property Area
        1 if Property_Area == "Semiurban" else 0,
        1 if Property_Area == "Urban" else 0,

        # Employer Category
        1 if Employer_Category == "Government" else 0,
        1 if Employer_Category == "MNC" else 0,
        1 if Employer_Category == "Private" else 0,
        1 if Employer_Category == "Unemployed" else 0,

        1 if Gender_Male == "Male" else 0
    ]

    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    result = model.predict(X_scaled)[0]

    if result == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
