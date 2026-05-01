import streamlit as st
import joblib
import pandas as pd
import numpy as np

preprocessor = joblib.load('models/preprocessor.pkl')
model = joblib.load('models/rf_model.pkl')

st.set_page_config(page_title="Credit Risk Prediction", page_icon="💳", layout="wide")
st.title("Credit Risk Prediction System")
st.caption("CS F415 Data Mining — BITS Pilani Goa | Group C")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Applicant Profile")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["male", "female"])
    education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
    income = st.number_input("Annual Income ($)", min_value=0, max_value=1000000, value=50000, step=1000)
    emp_exp = st.number_input("Employment Experience (years)", min_value=0, max_value=50, value=3)
    home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    loan_intent = st.selectbox("Loan Intent", [
        "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
    ])

with col2:
    st.subheader("Loan & Credit Details")
    loan_amnt = st.number_input("Loan Amount ($)", min_value=0, max_value=100000, value=10000, step=500)
    int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.1)
    loan_pct = st.number_input("Loan % of Income", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    cred_hist = st.number_input("Credit History Length (years)", min_value=0, max_value=30, value=5)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    prev_default = st.selectbox("Previous Loan Default on File", ["No", "Yes"])

st.markdown("---")

if st.button("Predict Risk", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        'person_age': age,
        'person_gender': gender,
        'person_education': education,
        'person_income': income,
        'person_emp_exp': emp_exp,
        'person_home_ownership': home,
        'loan_amnt': loan_amnt,
        'loan_intent': loan_intent,
        'loan_int_rate': int_rate,
        'loan_percent_income': loan_pct,
        'cb_person_cred_hist_length': cred_hist,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': prev_default
    }])

    processed = preprocessor.transform(input_df)

    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0]

    st.markdown("### Prediction Result")
    if prediction == 1:
        st.error(f"**BAD RISK** — Default Probability: {probability[1]:.1%}")
        st.progress(probability[1])
    else:
        st.success(f"**GOOD RISK** — Repayment Probability: {probability[0]:.1%}")
        st.progress(probability[0])

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Default Probability", f"{probability[1]:.1%}")
    with col_b:
        st.metric("Repayment Probability", f"{probability[0]:.1%}")
