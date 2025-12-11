import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(page_title="Churn Predictor", page_icon="📡", layout="wide")

# ---------------------------------------------------------------
# MINIMAL ELEGANT THEME
# ---------------------------------------------------------------
st.markdown("""
    <style>
        .main-title {
            font-size: 32px;
            font-weight: 800;
            margin-bottom: -10px;
            background: linear-gradient(90deg, #0066ff, #00e5ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .section-header {
            font-size: 20px;
            font-weight: 600;
            margin-top: 10px;
            color: #333;
        }
        .risk-badge {
            padding: 6px 14px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 16px;
            display: inline-block;
        }
        .low { background:#d1ffd6; color:#0f7b28; }
        .med { background:#fff2cc; color:#a67c00; }
        .high { background:#ffd6d6; color:#a60000; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    for path in ["churn_model.pkl", "notebooks/churn_model.pkl"]:
        if os.path.exists(path):
            return joblib.load(path)
    return None

model = load_model()

# ---------------------------------------------------------------
# TITLE
# ---------------------------------------------------------------
st.markdown("<h1 class='main-title'>Telecom Churn Predictor</h1>", unsafe_allow_html=True)
st.write("Provide customer details to estimate churn probability.")

# ---------------------------------------------------------------
# 2-COLUMN UI LAYOUT
# ---------------------------------------------------------------
left, right = st.columns([1.2, 1])

# ========================= LEFT INPUT PANE =========================
with left:
    st.markdown("<div class='section-header'>Customer Profile</div>", unsafe_allow_html=True)

    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    senior = st.checkbox("Senior Citizen")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    st.markdown("<div class='section-header'>Services & Usage</div>", unsafe_allow_html=True)
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    colA, colB = st.columns(2)
    online_security = colA.checkbox("Online Security")
    tech_support = colB.checkbox("Tech Support")

    st.markdown("<div class='section-header'>Billing</div>", unsafe_allow_html=True)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.checkbox("Paperless Billing", value=True)

    col1, col2 = st.columns(2)
    monthly = col1.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
    total = col2.number_input("Total Charges ($)", 0.0, 9000.0, monthly * tenure)

    input_data = {
        'gender': gender,
        'SeniorCitizen': str(int(senior)),
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': "Yes",
        'MultipleLines': "No",
        'InternetService': internet,
        'OnlineSecurity': "Yes" if online_security else "No",
        'OnlineBackup': "No",
        'DeviceProtection': "No",
        'TechSupport': "Yes" if tech_support else "No",
        'StreamingTV': "No",
        'StreamingMovies': "No",
        'Contract': contract,
        'PaperlessBilling': "Yes" if paperless else "No",
        'PaymentMethod': payment,
        'MonthlyCharges': monthly,
        'TotalCharges': total
    }

    run = st.button("Predict Churn")

# ========================= RIGHT OUTPUT PANE =========================
with right:
    if run:
        df_input = pd.DataFrame([input_data])

        if model:
            try:
                prob = model.predict_proba(df_input)[0][1]
            except:
                prob = 0.50
        else:
            prob = 0.20
            if contract == "Month-to-month": prob += 0.30
            if internet == "Fiber optic": prob += 0.15
            if payment == "Electronic check": prob += 0.10
            if tenure < 6: prob += 0.10
            prob = min(prob, 0.99)

        st.markdown("<div class='section-header'>Churn Prediction</div>", unsafe_allow_html=True)
        st.metric("Churn Probability", f"{prob*100:.1f}%")

        # Risk Badge
        if prob > 0.6:
            badge = "<div class='risk-badge high'>High Risk</div>"
        elif prob > 0.4:
            badge = "<div class='risk-badge med'>Moderate Risk</div>"
        else:
            badge = "<div class='risk-badge low'>Low Risk</div>"

        st.markdown(badge, unsafe_allow_html=True)

        # Mini gauge style progress bar
        st.progress(float(prob))

        # Recommendations
        if prob > 0.6:
            st.write("• Offer contract renewal discounts.")
            st.write("• Improve service quality and support.")
        elif prob > 0.4:
            st.write("• Monitor customer usage patterns.")
        else:
            st.write("• Customer is stable, no immediate action.")

    else:
        st.info("Fill the form and click **Predict Churn**.")
