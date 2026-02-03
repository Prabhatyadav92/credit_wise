import streamlit as st
import joblib
import numpy as np
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="CreditWise | Loan Prediction",
    page_icon="💳",
    layout="centered"
)

st.title("💳 CreditWise – Loan Approval Predictor")
st.caption("AI-powered loan approval prediction system")

# ================= LOAD MODEL =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ================= USER INPUT =================
st.subheader("📌 Applicant Details")

col1, col2 = st.columns(2)

with col1:
    Applicant_Income = st.number_input("Applicant Income (₹)", min_value=0.0)
    Coapplicant_Income = st.number_input("Co-Applicant Income (₹)", min_value=0.0)
    Age = st.number_input("Age", min_value=18, max_value=100)
    Dependents = st.number_input("Dependents", min_value=0, max_value=10)
    Credit_Score = st.number_input("Credit Score", min_value=300, max_value=900)

with col2:
    Existing_Loans = st.number_input("Existing Loans", min_value=0, max_value=10)
    DTI_Ratio = st.number_input("DTI Ratio (%)", min_value=0.0)
    Savings = st.number_input("Savings (₹)", min_value=0.0)
    Collateral_Value = st.number_input("Collateral Value (₹)", min_value=0.0)

st.subheader("📌 Loan Information")

col3, col4 = st.columns(2)

with col3:
    Loan_Amount = st.number_input("Loan Amount (₹)", min_value=0.0)
    Loan_Term = st.number_input("Loan Term (months)", min_value=1, max_value=360)
    Education_Level = st.selectbox("Education Level", ["0", "1"], help="0 = Not Graduate, 1 = Graduate")

with col4:
    Employment_Status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
    Marital_Status = st.selectbox("Marital Status", ["Married", "Single"])
    Gender = st.selectbox("Gender", ["Female", "Male"])

st.subheader("📌 Property & Purpose")

col5, col6 = st.columns(2)

with col5:
    Loan_Purpose = st.selectbox("Loan Purpose", ["Car", "Education", "Home", "Personal"])
    Property_Area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

with col6:
    Employer_Category = st.selectbox("Employer Category", ["Government", "MNC", "Private", "Unemployed"])

# ================= PREDICTION =================
st.divider()

if st.button("🔍 Predict Loan Approval", use_container_width=True):

    features = [
        Applicant_Income,
        Coapplicant_Income,
        Age,
        Dependents,
        Credit_Score,
        Existing_Loans,
        DTI_Ratio,
        Savings,
        Collateral_Value,
        Loan_Amount,
        Loan_Term,
        int(Education_Level),

        # Employment Status
        1 if Employment_Status == "Salaried" else 0,
        1 if Employment_Status == "Self-employed" else 0,
        1 if Employment_Status == "Unemployed" else 0,

        # Marital
        1 if Marital_Status == "Single" else 0,

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

        # Gender
        1 if Gender == "Male" else 0
    ]

    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]

    if prediction == 1:
        st.success("✅ **Loan Approved**")
    else:
        st.error("❌ **Loan Rejected**")

    # Optional probability (if model supports it)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_scaled)[0][prediction]
        st.info(f"📊 Confidence: **{prob * 100:.2f}%**")

# ================= FOOTER =================
st.divider()
st.caption("© 2026 CreditWise | Built with Streamlit & Machine Learning| by Prabhat")
