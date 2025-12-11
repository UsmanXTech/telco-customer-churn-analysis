import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(page_title="Telecom Churn Dashboard", page_icon="📊", layout="wide")

# ---------------------------------------------------------------
# MINIMAL CLEAN THEME
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
        font-size: 22px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 6px;
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
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Select a Section", ["Churn Predictor", "EDA Visualization"])

# PATH FOR PROCESSED DATA
processed_path = "data/processed/processed_data.csv"

# ===============================================================
# PAGE 1 — CHURN PREDICTOR
# ===============================================================
if page == "Churn Predictor":

    st.markdown("<h1 class='main-title'>Telecom Churn Predictor</h1>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.3, 1])

    # -------------------- LEFT INPUT FORM --------------------
    with col_left:

        st.subheader("Customer Input")

        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        senior = st.checkbox("Senior Citizen")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        colA, colB = st.columns(2)
        online_security = colA.checkbox("Online Security")
        tech_support = colB.checkbox("Tech Support")

        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

        paperless = st.checkbox("Paperless Billing", value=True)

        col1, col2 = st.columns(2)
        monthly = col1.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
        total = col2.number_input("Total Charges ($)", 0.0, 9000.0, monthly * tenure)

        run = st.button("Predict Churn")

    # -------------------- RIGHT OUTPUT PANEL --------------------
    with col_right:

        if run:
            st.subheader("Prediction Output")

            data = {
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

            df_input = pd.DataFrame([data])

            # Prediction or fallback
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

            st.metric("Churn Probability", f"{prob*100:.1f}%")

            if prob > 0.6:
                st.markdown("<div class='risk-badge high'>High Risk</div>", unsafe_allow_html=True)
            elif prob > 0.4:
                st.markdown("<div class='risk-badge med'>Moderate Risk</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='risk-badge low'>Low Risk</div>", unsafe_allow_html=True)

            st.progress(float(prob))

        else:
            st.info("Fill details & click Predict.")


# ===============================================================
# PAGE 2 — EDA VISUALIZATION (SELECT GRAPH)
# ===============================================================
elif page == "EDA Visualization":

    st.markdown("<h1 class='main-title'>EDA Visualization</h1>", unsafe_allow_html=True)

    if not os.path.exists(processed_path):
        st.error(f"Processed dataset NOT found at: {processed_path}")
        st.stop()

    df = pd.read_csv(processed_path)
    st.success(f"Processed dataset loaded → {processed_path}")

    st.dataframe(df.head())

    # Sidebar Graph Selection
    graph_option = st.sidebar.selectbox(
        "Select Visualization",
        [
            "Missing Values Heatmap",
            "Numerical Distribution",
            "Categorical Distribution",
            "Correlation Heatmap",
            "Churn Distribution"
        ]
    )

    # -------------------------------
    # 1. Missing Value Heatmap
    # -------------------------------
    if graph_option == "Missing Values Heatmap":
        st.markdown("### ❗ Missing Values Heatmap")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(df.isnull(), cbar=False, ax=ax)
        st.pyplot(fig)

    # -------------------------------
    # 2. Numerical Distribution
    # -------------------------------
    elif graph_option == "Numerical Distribution":
        st.markdown("### 📈 Numerical Column Distributions")
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns

        for col in num_cols:
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(col)
            st.pyplot(fig)

    # -------------------------------
    # 3. Categorical Distribution
    # -------------------------------
    elif graph_option == "Categorical Distribution":
        st.markdown("### 📊 Categorical Feature Counts")
        cat_cols = df.select_dtypes(include=['object']).columns

        for col in cat_cols:
            fig, ax = plt.subplots(figsize=(6, 3))
            df[col].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(col)
            st.pyplot(fig)

    # -------------------------------
    # 4. Correlation Heatmap
    # -------------------------------
    elif graph_option == "Correlation Heatmap":
        st.markdown("### 🔥 Correlation Heatmap")
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns

        plt.figure(figsize=(10, 5))
        heat = sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
        st.pyplot(heat.figure)

    # -------------------------------
    # 5. Churn Distribution
    # -------------------------------
    elif graph_option == "Churn Distribution":
        if "Churn" in df.columns:
            st.markdown("### 📉 Churn Distribution")
            fig, ax = plt.subplots(figsize=(4, 3))
            df["Churn"].value_counts().plot(kind='bar', color=['#00aaff', '#ff0066'])
            ax.set_title("Churn Count")
            st.pyplot(fig)
        else:
            st.warning("Churn column not found!")
