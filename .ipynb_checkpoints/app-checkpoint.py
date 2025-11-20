import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Default Predictor", page_icon="💰")
st.title("💰 Microfinance Loan Default Prediction")

# Load saved model
model = joblib.load("loan_default_model.joblib")

st.subheader("Enter Client Details")

# User inputs
credit_policy = st.selectbox("Credit Policy (1 = yes, 0 = no)", [0, 1])
purpose = st.selectbox("Purpose", [
    "credit_card", "debt_consolidation", "educational",
    "major_purchase", "small_business", "other"
])
int_rate = st.number_input("Interest Rate", min_value=0.0, max_value=1.0, value=0.12)
installment = st.number_input("Installment", min_value=0.0, value=250.0)
log_annual_inc = st.number_input("Log Annual Income", min_value=0.0, value=10.5)
dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=15.0)
fico = st.number_input("FICO Score", min_value=300, max_value=850, value=700)
days_with_cr_line = st.number_input("Days with Credit Line", min_value=0.0, value=3500.0)
revol_bal = st.number_input("Revolving Balance", min_value=0.0, value=5000.0)
revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=30.0)
inq_last_6mths = st.number_input("Inquiries (last 6 months)", min_value=0, value=1)
delinq_2yrs = st.number_input("Delinquencies (last 2 years)", min_value=0, value=0)
pub_rec = st.number_input("Public Records", min_value=0, value=0)

# Prediction button
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "credit.policy": credit_policy,
        "purpose": purpose,
        "int.rate": float(int_rate),
        "installment": float(installment),
        "log.annual.inc": float(log_annual_inc),
        "dti": float(dti),
        "fico": int(fico),
        "days.with.cr.line": float(days_with_cr_line),
        "revol.bal": float(revol_bal),
        "revol.util": float(revol_util),
        "inq.last.6mths": int(inq_last_6mths),
        "delinq.2yrs": int(delinq_2yrs),
        "pub.rec": int(pub_rec),
    }])

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.write("---")
    if pred == 1:
        st.error(f"⚠️ Client is likely to **DEFAULT**. (Risk: {prob:.2%})")
    else:
        st.success(f"✅ Client is likely to **FULLY PAY**. (Risk: {prob:.2%})")
