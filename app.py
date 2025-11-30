import streamlit as st
import pandas as pd
import joblib

# =========================
# 1. PAGE CONFIG & STYLING
# =========================
st.set_page_config(
    page_title="Microfinance Loan Default Predictor",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for fintech-style UI
st.markdown("""
<style>
    /* Global background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #020617 40%, #0b1220 100%);
        color: #e5e7eb;
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    /* Main title */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f9fafb;
        letter-spacing: 0.03em;
    }
    .sub-title {
        color: #9ca3af;
        font-size: 0.95rem;
    }
    /* Card container */
    .glass-card {
        background: rgba(15,23,42,0.85);
        border-radius: 18px;
        padding: 1.3rem 1.5rem;
        border: 1px solid rgba(148,163,184,0.35);
        box-shadow: 0 18px 45px rgba(15,23,42,0.7);
    }
    .glass-card h3 {
        color: #e5e7eb;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    label, .stSelectbox label, .stNumberInput label {
        font-weight: 500 !important;
        color: #e5e7eb !important;
    }
    .risk-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .risk-badge.low {
        background: rgba(22,163,74,0.15);
        color: #4ade80;
        border: 1px solid rgba(34,197,94,0.6);
    }
    .risk-badge.high {
        background: rgba(220,38,38,0.15);
        color: #fca5a5;
        border: 1px solid rgba(239,68,68,0.7);
    }
    .metric-label {
        font-size: 0.9rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #e5e7eb;
    }
    .footer-note {
        font-size: 0.8rem;
        color: #6b7280;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# 2. LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("loan_default_model.joblib")
        return model, None
    except Exception as e:
        return None, str(e)

model, model_error = load_model()

# =========================
# 3. HEADER / SIDEBAR
# =========================
with st.sidebar:
    st.markdown("### 💼 Project Info")
    st.markdown("""
    **Loan Default Classification – Microfinance**  
    Predict whether a borrower is likely to **fully pay** or **default** on a loan.

    **Tech stack:**
    - Python, scikit-learn  
    - Streamlit  
    - joblib (model loading)
    """)
    st.markdown("---")
    st.markdown("""
    **Created by:** Group 2  
    **Course:** Machine Learning  
    **Institution:** Pentecost University
    """)


# Main header
st.markdown('<div class="main-title">💰 Microfinance Loan Default Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Interactive dashboard to estimate the default risk of a borrower using machine learning.</div><br>', unsafe_allow_html=True)

if model_error:
    st.error(f"❌ Could not load model: `{model_error}`")
    st.stop()

# =========================
# 4. LAYOUT: TWO MAIN CARDS
# =========================
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🧍 Borrower Profile")

    col1, col2 = st.columns(2)

    with col1:
        credit_policy = st.selectbox("Credit Policy (1 = meets, 0 = not)", [1, 0])
        purpose = st.selectbox(
            "Loan Purpose",
            ["debt_consolidation", "credit_card", "educational",
             "major_purchase", "small_business", "home_improvement", "other"]
        )
        log_annual_inc = st.number_input("Log Annual Income", min_value=0.0, value=10.5, step=0.1)
        dti = st.number_input("Debt-to-Income Ratio (DTI)", min_value=0.0, value=15.0, step=0.5)

    with col2:
        fico = st.number_input("FICO Score", min_value=300, max_value=850, value=680, step=1)
        days_with_cr_line = st.number_input("Days with Credit Line", min_value=0.0, value=4000.0, step=30.0)
        inq_last_6mths = st.number_input("Inquiries (last 6 months)", min_value=0, value=1, step=1)
        delinq_2yrs = st.number_input("Delinquencies (last 2 years)", min_value=0, value=0, step=1)

    st.markdown("### 💳 Revolving Credit / Installment")
    col3, col4, col5 = st.columns(3)
    with col3:
        int_rate = st.number_input("Interest Rate (e.g. 0.12)", min_value=0.0, max_value=1.0, value=0.12, step=0.01)
    with col4:
        installment = st.number_input("Monthly Installment", min_value=0.0, value=250.0, step=10.0)
    with col5:
        pub_rec = st.number_input("Public Records", min_value=0, value=0, step=1)

    col6, col7 = st.columns(2)
    with col6:
        revol_bal = st.number_input("Revolving Balance", min_value=0.0, value=8000.0, step=500.0)
    with col7:
        revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=35.0, step=1.0)

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Risk Summary")

    st.markdown("Use the inputs on the left and click **Predict Risk** to see the default probability.")

    # Placeholder containers for metrics
    risk_placeholder = st.empty()
    prob_bar_placeholder = st.empty()
    explanation_placeholder = st.empty()

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 5. PREDICTION LOGIC
# =========================
# Prepare input DataFrame
input_data = pd.DataFrame([{
    "credit.policy": int(credit_policy),
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
    "pub.rec": int(pub_rec)
}])

st.markdown("<br>", unsafe_allow_html=True)
center = st.container()
with center:
    predict_btn = st.button("🔮 Predict Default Risk", use_container_width=True)

if predict_btn:
    try:
        pred = model.predict(input_data)[0]
        prob_default = model.predict_proba(input_data)[0][1]  # P(not fully paid)
        prob_pay = 1 - prob_default

        # HIGH RISK if prob_default >= 0.5
        risk_level = "HIGH RISK" if prob_default >= 0.5 else "LOW RISK"
        badge_class = "high" if prob_default >= 0.5 else "low"

        with right_col:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Risk Summary")

            # Risk badge + main probability
            st.markdown(
                f'<span class="risk-badge {badge_class}">{risk_level}</span>',
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)
            colA, colB = st.columns(2)
            with colA:
                st.markdown('<div class="metric-label">Default Probability</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{prob_default*100:.1f}%</div>', unsafe_allow_html=True)
            with colB:
                st.markdown('<div class="metric-label">Payback Probability</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{prob_pay*100:.1f}%</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.write("**Risk Gauge (Default Probability)**")
            st.progress(float(prob_default))

            # Short explanation
            if prob_default >= 0.7:
                msg = "This borrower is at **very high risk** of not fully paying the loan. Consider stricter conditions, smaller amounts, or rejection."
            elif prob_default >= 0.5:
                msg = "This borrower is at **moderate-to-high risk**. Review additional documents or collateral before approval."
            elif prob_default >= 0.3:
                msg = "This borrower has a **moderate risk** level. Standard checks are recommended."
            else:
                msg = "This borrower appears to be **low risk** based on the current data."

            st.markdown(f"> {msg}")
            st.markdown(
                '<div class="footer-note">Note: This model is a decision support tool and should complement, not replace, human judgment.</div>',
                unsafe_allow_html=True
            )

            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
