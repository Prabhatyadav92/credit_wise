
from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)
app.config["DEBUG"] = True


# ================= PATHS =================

MODEL_PATH = r"C:\Users\Administrator\OneDrive\Desktop\Credit_wise loan predictiion\model\model.pkl"
SCALER_PATH = r"C:\Users\Administrator\OneDrive\Desktop\Credit_wise loan predictiion\model\scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# ================= ROUTES =================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        f = request.form

        # Categorical helpers
        emp = f["Employment_Status"]
        loan = f["Loan_Purpose"]
        prop = f["Property_Area"]
        emp_cat = f["Employer_Category"]

        # ================= FEATURE ORDER (CRITICAL) =================
        features = [
            float(f["Applicant_Income"]),
            float(f["Coapplicant_Income"]),
            float(f["Age"]),
            float(f["Dependents"]),
            float(f["Credit_Score"]),
            float(f["Existing_Loans"]),
            float(f["DTI_Ratio"]),
            float(f["Savings"]),
            float(f["Collateral_Value"]),
            float(f["Loan_Amount"]),
            float(f["Loan_Term"]),
            int(f["Education_Level"]),

            1 if emp == "Salaried" else 0,
            1 if emp == "Self-employed" else 0,
            1 if emp == "Unemployed" else 0,

            int(f["Marital_Status_Single"]),

            1 if loan == "Car" else 0,
            1 if loan == "Education" else 0,
            1 if loan == "Home" else 0,
            1 if loan == "Personal" else 0,

            1 if prop == "Semiurban" else 0,
            1 if prop == "Urban" else 0,

            1 if emp_cat == "Government" else 0,
            1 if emp_cat == "MNC" else 0,
            1 if emp_cat == "Private" else 0,
            1 if emp_cat == "Unemployed" else 0,

            int(f["Gender_Male"])
        ]

        # Convert → scale → predict
        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        result = model.predict(X_scaled)[0]

        prediction = "✅ Loan Approved" if result == 1 else "❌ Loan Rejected"

    return render_template("index.html", prediction=prediction)

# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True)

