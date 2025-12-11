import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os

# =========================================================================
# 1. PAGE CONFIGURATION
# =========================================================================
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# =========================================================================
# 2. MODERN DARK THEME CSS
# =========================================================================
st.markdown("""
    <style>
    /* Global White Background */
    .stApp {
        background-color: #ffffff !important;
        background-image: none !important;
        color: #1a1a1a !important;
    }

    /* All Text Elements - Dark Text */
    p, div, span, label, li {
        color: #1a1a1a !important;
    }

    /* Markdown paragraphs */
    .stMarkdown p {
        color: #333333 !important;
    }

    /* Info/Warning/Success boxes */
    .stAlert {
        background-color: #e3f2fd !important;
        color: #1a1a1a !important;
        border: 1px solid #2196f3;
    }

    .stAlert p, .stAlert div {
        color: #1a1a1a !important;
    }

    /* Input Fields */
    .stSelectbox div[data-baseweb="select"] > div, 
    .stNumberInput div[data-baseweb="input"] > div {
        background-color: #f5f5f5 !important;
        color: #1a1a1a !important;
        border: 1px solid #d0d0d0;
        border-radius: 8px;
    }

    /* Dropdown options */
    [data-baseweb="popover"] {
        background-color: #ffffff !important;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a !important;
        font-family: 'Inter', sans-serif;
    }

    h1 {
        background: linear-gradient(90deg, #2196f3 0%, #00bcd4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Result Card */
    .result-card {
        background: #f8f9fa !important;
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        text-align: center;
        color: #1a1a1a !important;
    }

    .result-card h3 {
        color: #1a1a1a !important;
    }

    .result-card p, .result-card div {
        color: #333333 !important;
    }

    /* Markdown Content */
    .stMarkdown {
        color: #1a1a1a !important;
    }

    /* Button */
    .stButton>button {
        background: linear-gradient(90deg, #2196f3 0%, #00bcd4 100%);
        color: #ffffff !important;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        margin-top: 20px;
        transition: transform 0.2s;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }

    /* Radio and Checkbox */
    .stRadio label, .stCheckbox label {
        color: #1a1a1a !important;
    }

    .stRadio > label, .stCheckbox > label {
        color: #1a1a1a !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f5f5f5 !important;
        color: #1a1a1a !important;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }

    .streamlit-expanderHeader p, 
    .streamlit-expanderHeader div,
    .streamlit-expanderHeader svg {
        color: #1a1a1a !important;
        fill: #1a1a1a !important;
    }

    /* Slider labels */
    .stSlider label {
        color: #1a1a1a !important;
    }

    /* Number input labels */
    .stNumberInput label {
        color: #1a1a1a !important;
    }

    /* Select box labels */
    .stSelectbox label {
        color: #1a1a1a !important;
    }

    /* Slider track */
    .stSlider [data-baseweb="slider"] {
        background-color: #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)


# =========================================================================
# 3. LOAD MODEL
# =========================================================================
@st.cache_resource
def load_model():
    paths = ['notebooks/churn_model.pkl', 'churn_model.pkl']
    for path in paths:
        if os.path.exists(path):
            return joblib.load(path)
    return None

model = load_model()

# =========================================================================
# 4. MAIN LAYOUT
# =========================================================================
st.title("🤖 Telecom Churn Predictor")
st.markdown("Enter customer details below to estimate the risk of churn.")

# Create two columns: Inputs on Left, Results on Right
col_inputs, col_result = st.columns([1.2, 1], gap="large")

with col_inputs:
    st.markdown("### 📋 Customer Profile")
    
    with st.expander("👤 Demographics", expanded=True):
        c1, c2 = st.columns(2)
        gender = c1.radio("Gender", ["Male", "Female"], horizontal=True)
        senior = c2.checkbox("Senior Citizen")
        partner = c1.selectbox("Partner", ["Yes", "No"])
        dependents = c2.selectbox("Dependents", ["Yes", "No"])
        
    with st.expander("📞 Services & Tenure", expanded=True):
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        c1, c2 = st.columns(2)
        # Simplified service inputs for clean UI
        online_security = c1.checkbox("Online Security")
        tech_support = c2.checkbox("Tech Support")
        
    with st.expander("💳 Contract & Billing", expanded=True):
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = st.checkbox("Paperless Billing", value=True)
        
        c1, c2 = st.columns(2)
        monthly = c1.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0, step=0.5)
        # Auto-calculate default Total Charges based on tenure
        total = c2.number_input("Total Charges ($)", 0.0, 9000.0, monthly * tenure, step=10.0)

    # Prepare Input Dictionary (Mapping UI inputs to Model features)
    # Note: We hardcode 'No' for services not explicitly asked to keep UI simple, 
    # or you can add more checkboxes if your model requires them strictly.
    input_data = {
        'gender': gender,
        'SeniorCitizen': str(1 if senior else 0),  # Convert to string to match model training
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

    analyze_btn = st.button("🚀 Analyze Churn Risk")

# =========================================================================
# 5. PREDICTION LOGIC & VISUALIZATION
# =========================================================================
with col_result:
    if analyze_btn:
        st.markdown("### 📊 Analysis Result")
        
        # ---------------------------------------------------------
        # PREDICTION (Use Model or Fallback Mock)
        # ---------------------------------------------------------
        if model:
            try:
                # Convert dict to DataFrame
                df_input = pd.DataFrame([input_data])
                prob = model.predict_proba(df_input)[0][1]
            except Exception as e:
                st.error(f"Model Error: {e}")
                prob = 0.5 # Fallback
        else:
            # MOCK LOGIC (If no model file found)
            # Logic: Monthly contract + High Price + Fiber = High Risk
            prob = 0.20
            if contract == "Month-to-month": prob += 0.30
            if internet == "Fiber optic": prob += 0.15
            if payment == "Electronic check": prob += 0.10
            if tenure < 6: prob += 0.10
            prob = min(prob, 0.99)
        
        # ---------------------------------------------------------
        # GAUGE CHART (Plotly)
        # ---------------------------------------------------------
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability", 'font': {'size': 24, 'color': 'white'}},
            number = {'suffix': "%", 'font': {'color': 'white', 'size': 40}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "rgba(0,0,0,0)"}, # Transparent bar
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#333",
                'steps': [
                    {'range': [0, 40], 'color': "#00f2fe"},  # Safe (Cyan)
                    {'range': [40, 70], 'color': "#4facfe"},  # Warning (Blue)
                    {'range': [70, 100], 'color': "#ff0080"}  # Danger (Pink)
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 5},
                    'thickness': 0.75,
                    'value': prob * 100
                }
            }
        ))
        
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Inter"}, height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # ---------------------------------------------------------
        # TEXT INSIGHTS
        # ---------------------------------------------------------
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        
        if prob > 0.6:
            st.markdown(f"### ⚠️ High Risk Customer")
            st.markdown(f"This customer has a **{prob:.1%}** probability of leaving.")
            st.markdown("---")
            st.markdown("**Recommended Actions:**")
            st.markdown("👉 Offer a 12-month contract discount.")
            st.markdown("👉 Waive current month's support fees.")
        elif prob > 0.4:
            st.markdown(f"### ⚖️ Moderate Risk")
            st.markdown(f"Monitor usage. Probability: **{prob:.1%}**")
        else:
            st.markdown(f"### ✅ Loyal Customer")
            st.markdown(f"Low risk of churn (**{prob:.1%}**). No immediate action needed.")
            
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Placeholder State
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("👈 Fill in the customer details on the left and click **'Analyze Churn Risk'**.")
        # Fixed Image Line (No style argument)
        st.image("https://cdn-icons-png.flaticon.com/512/3261/3261308.png", width=150)