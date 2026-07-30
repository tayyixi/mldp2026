"""
Term Deposit Subscription Predictor — Modern Dashboard
------------------------------------------------------
Streamlit app for CAI2C08 Machine Learning for Developers — Project Codes submission.
"""

import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------------------------
# Page Config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="LeadLens | Term Deposit Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------------
# Custom CSS Design System
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* App Container */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.1);
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        color: #F8FAFC;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1rem;
        margin: 0;
        max-width: 650px;
    }
    
    /* Cards & Containers */
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        border-radius: 12px !important;
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    
    /* Metric Glass Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.25rem;
    }
    
    /* Custom Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    
    div.stButton > button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35);
        transform: translateY(-1px);
    }
    
    /* Badge styling */
    .badge-high {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Model Loading
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("final_model.pkl")

try:
    model = load_model()
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False

# ----------------------------------------------------------------------------
# Sidebar — App Info & Model Diagnostics
# ----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric-folders/100/phone-office.png", width=64)
    st.title("LeadLens AI")
    st.caption("Call Center Priority Scoring System")
    
    st.divider()
    
    st.markdown("### 📊 Model Performance")
    col_a, col_b = st.columns(2)
    col_a.metric("ROC-AUC", "0.79")
    col_b.metric("Recall", "59%")
    
    st.markdown("""
    **Model Type:** Logistic Regression  
    **Optimization:** `RandomizedSearchCV`  
    **Class Balancing:** Applied  
    **Dataset:** UCI Bank Marketing
    """)
    
    st.divider()
    
    if not MODEL_LOADED:
        st.error("⚠️ `final_model.pkl` missing! Place the exported file in the root folder.")
    else:
        st.success("✅ Prediction Engine Ready")

# ----------------------------------------------------------------------------
# Hero Header
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">⚡ Term Deposit Lead Predictor</div>
    <p class="hero-subtitle">Optimize call-center targeting by estimating subscription probabilities for prospective banking clients in real time.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Input Form Tabs
# ----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["👤 Demographic Profile", "📞 Campaign Contact Data", "📈 Macroeconomic Indicators"])

with tab1:
    with st.container(border=True):
        st.subheader("Client Demographic Information")
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Client Age", 18, 95, 38, help="Client's age in years")
            job = st.selectbox(
                "Occupation / Job Category",
                ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                 'retired', 'self-employed', 'services', 'student', 'technician',
                 'unemployed', 'unknown'],
                index=0
            )
            marital = st.selectbox("Marital Status", ['divorced', 'married', 'single', 'unknown'], index=1)
        
        with col2:
            education = st.selectbox(
                "Education Level",
                ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
                 'professional.course', 'university.degree', 'unknown'],
                index=6
            )
            default = st.selectbox("Credit in Default?", ['no', 'yes', 'unknown'], index=0)
            housing = st.selectbox("Housing Loan?", ['no', 'yes', 'unknown'], index=0)
            loan = st.selectbox("Personal Loan?", ['no', 'yes', 'unknown'], index=0)

with tab2:
    with st.container(border=True):
        st.subheader("Campaign Details & History")
        col3, col4 = st.columns(2)
        with col3:
            contact = st.selectbox("Communication Channel", ['cellular', 'telephone'], index=0)
            month = st.selectbox(
                "Last Contact Month",
                ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                index=4
            )
            day_of_week = st.selectbox("Last Contact Day", ['mon', 'tue', 'wed', 'thu', 'fri'], index=0)
            campaign = st.slider("Contacts Performed This Campaign", 1, 35, 2)
            
        with col4:
            previous = st.slider("Contacts Performed Prior to Campaign", 0, 6, 0)
            contacted_before_label = st.radio(
                "Was Client Contacted in Any Past Campaign?", 
                ["No", "Yes"], 
                index=0, 
                horizontal=True
            )
            poutcome = st.selectbox("Previous Campaign Outcome", ['nonexistent', 'failure', 'success'], index=0)

with tab3:
    with st.container(border=True):
        st.subheader("Economic Environment Benchmarks")
        col5, col6, col7 = st.columns(3)
        with col5:
            euribor3m = st.number_input("Euribor 3-Month Rate (%)", min_value=0.5, max_value=6.0, value=4.86, step=0.01)
        with col6:
            cons_price_idx = st.number_input("Consumer Price Index (CPI)", min_value=90.0, max_value=96.0, value=93.75, step=0.01)
        with col7:
            cons_conf_idx = st.number_input("Consumer Confidence Index", min_value=-60.0, max_value=-20.0, value=-41.8, step=0.1)

# Preprocessing Inputs
contacted_before = 1 if contacted_before_label == "Yes" else 0

input_row = pd.DataFrame([{
    "age": age, "job": job, "marital": marital, "education": education,
    "default": default, "housing": housing, "loan": loan, "contact": contact,
    "month": month, "day_of_week": day_of_week, "campaign": campaign,
    "previous": previous, "poutcome": poutcome,
    "cons.price.idx": cons_price_idx, "cons.conf.idx": cons_conf_idx,
    "euribor3m": euribor3m, "contacted_before": contacted_before,
}])

# ----------------------------------------------------------------------------
# Action & Results Dashboard
# ----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔮 Evaluate Lead Probability", use_container_width=True, disabled=not MODEL_LOADED)

if predict_clicked and MODEL_LOADED:
    proba = model.predict_proba(input_row)[0, 1]
    pred = model.predict(input_row)[0]
    
    st.markdown("### 🎯 Lead Evaluation Summary")
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        with st.container(border=True):
            st.caption("PREDICTED LIKELIHOOD")
            st.markdown(f"# {proba*100:.1f}%")
            
            if proba >= 0.5:
                st.success("High Conversion Potential")
            elif proba >= 0.3:
                st.warning("Moderate Conversion Potential")
            else:
                st.error("Low Conversion Potential")
                
    with res_col2:
        with st.container(border=True):
            st.caption("RECOMMENDED ACTION PLAN")
            if proba >= 0.5:
                st.markdown("### 🟢 Top Priority Lead")
                st.markdown("Assign to senior call specialists immediately. This prospect shows strong conversion traits based on economic indicators and past interactions.")
            elif proba >= 0.3:
                st.markdown("### 🟡 Medium Priority Lead")
                st.markdown("Queue for secondary follow-ups. Schedule calls during off-peak hours or run automated follow-up email/SMS touchpoints first.")
            else:
                st.markdown("### 🔴 Low Priority Lead")
                st.markdown("Deprioritize active phone calls to maximize team efficiency. Consider enrolling in standard drip marketing campaigns.")
            
            st.progress(min(max(proba, 0.0), 1.0))

elif not MODEL_LOADED:
    st.warning("Prediction disabled because `final_model.pkl` was not found.")