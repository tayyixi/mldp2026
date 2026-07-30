"""
LeadLens AI — Enterprise Term Deposit Intelligence
--------------------------------------------------
Streamlit Dashboard for CAI2C08 Machine Learning Project
"""

import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------------------------
# 1. Page Configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="LeadLens AI | Lead Scoring System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------------
# 2. Universal Modern CSS Design System (Theme-Agnostic & High Contrast)
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    /* CSS Variables & Global Resets */
    :root {
        --primary-color: #2563EB;
        --primary-dark: #1D4ED8;
        --border-radius: 12px;
    }
    
    /* Clean Top Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: var(--border-radius);
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    
    .hero-banner h1 {
        color: #FFFFFF !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin: 0 0 0.4rem 0 !important;
        letter-spacing: -0.02em;
    }
    
    .hero-banner p {
        color: #94A3B8 !important;
        font-size: 0.95rem !important;
        margin: 0 !important;
    }

    /* Section Header Badges */
    .section-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: rgba(37, 99, 235, 0.1);
        color: #2563EB;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }

    /* Card Box Container Styling */
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        border-radius: var(--border-radius) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
    }

    /* Big Callout Metric Card */
    .scoring-card-high {
        background: linear-gradient(135deg, #064E3B 0%, #022C22 100%);
        border: 1px solid #059669;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .scoring-card-medium {
        background: linear-gradient(135deg, #78350F 0%, #451A03 100%);
        border: 1px solid #D97706;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }

    .scoring-card-low {
        background: linear-gradient(135deg, #7F1D1D 0%, #450A0A 100%);
        border: 1px solid #DC2626;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }

    .score-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.85;
    }

    .score-number {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0.3rem 0;
    }

    /* Primary Action Button */
    div.stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.5rem !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 3. Model Loader
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
# 4. Sidebar Diagnostics
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ LeadLens AI")
    st.caption("Call Center Priority Scoring System")
    st.divider()
    
    st.markdown("### 📊 Model Metrics")
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("ROC-AUC", "0.79")
    col_s2.metric("Recall", "59%")
    
    st.markdown("""
    * **Algorithm**: Logistic Regression
    * **Tuning**: RandomizedSearchCV
    * **Class Weight**: Balanced
    * **Dataset**: UCI Bank Marketing
    """)
    st.divider()
    
    if MODEL_LOADED:
        st.success("● Pipeline Engine Loaded", icon="🟢")
    else:
        st.error("Model file `final_model.pkl` not found.", icon="🚨")

# ----------------------------------------------------------------------------
# 5. App Header Banner
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1>⚡ Term Deposit Subscription Intelligence</h1>
    <p>Predict client conversion probability and prioritize call-center workflows using AI.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 6. Main Dashboard Layout (2 Columns: Inputs vs Live Prediction)
# ----------------------------------------------------------------------------
col_input, col_output = st.columns([1.6, 1], gap="large")

with col_input:
    st.markdown('<span class="section-badge">Client & Campaign Parameters</span>', unsafe_allow_html=True)
    
    tab_demo, tab_camp, tab_econ = st.tabs(["👤 Client Profile", "📞 Contact History", "📈 Market Factors"])
    
    with tab_demo:
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Client Age", 18, 95, 38)
                job = st.selectbox("Job Category", [
                    'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                    'retired', 'self-employed', 'services', 'student', 'technician',
                    'unemployed', 'unknown'
                ], index=0)
                marital = st.selectbox("Marital Status", ['divorced', 'married', 'single', 'unknown'], index=1)
            
            with c2:
                education = st.selectbox("Education Level", [
                    'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
                    'professional.course', 'university.degree', 'unknown'
                ], index=6)
                default = st.selectbox("Credit in Default?", ['no', 'yes', 'unknown'], index=0)
                housing = st.selectbox("Housing Loan?", ['no', 'yes', 'unknown'], index=0)
                loan = st.selectbox("Personal Loan?", ['no', 'yes', 'unknown'], index=0)

    with tab_camp:
        with st.container(border=True):
            c3, c4 = st.columns(2)
            with c3:
                contact = st.selectbox("Contact Channel", ['cellular', 'telephone'], index=0)
                month = st.selectbox("Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], index=4)
                day_of_week = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'], index=0)
                campaign = st.number_input("Current Campaign Contacts", min_value=1, max_value=50, value=2)
                
            with c4:
                previous = st.number_input("Previous Campaign Contacts", min_value=0, max_value=20, value=0)
                contacted_before_label = st.radio("Contacted in Past Campaign?", ["No", "Yes"], index=0, horizontal=True)
                poutcome = st.selectbox("Previous Campaign Outcome", ['nonexistent', 'failure', 'success'], index=0)

    with tab_econ:
        with st.container(border=True):
            c5, c6, c7 = st.columns(3)
            with c5:
                euribor3m = st.number_input("Euribor 3M Rate (%)", 0.5, 6.0, 4.86, step=0.01)
            with c6:
                cons_price_idx = st.number_input("Consumer Price Index", 90.0, 96.0, 93.75, step=0.01)
            with c7:
                cons_conf_idx = st.number_input("Consumer Confidence", -60.0, -20.0, -41.8, step=0.1)

    predict_btn = st.button("🔮 Calculate Lead Subscription Score", use_container_width=True, disabled=not MODEL_LOADED)

# Preprocessing Input Data
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
# 7. Output & AI Scoring Panel
# ----------------------------------------------------------------------------
with col_output:
    st.markdown('<span class="section-badge">AI Prediction Output</span>', unsafe_allow_html=True)
    
    with st.container(border=True):
        if predict_btn and MODEL_LOADED:
            proba = model.predict_proba(input_row)[0, 1]
            prob_percent = proba * 100
            
            # Dynamic Card Style based on Score
            if proba >= 0.5:
                card_class = "scoring-card-high"
                priority_label = "HIGH PRIORITY LEAD"
                action_text = "🎯 **Action**: Connect immediately. High propensity to subscribe to term deposit."
            elif proba >= 0.3:
                card_class = "scoring-card-medium"
                priority_label = "MEDIUM PRIORITY LEAD"
                action_text = "📞 **Action**: Queue for follow-up call during regular agent cycles."
            else:
                card_class = "scoring-card-low"
                priority_label = "LOW PRIORITY LEAD"
                action_text = "📧 **Action**: Deprioritize direct phone calls. Add to automated email nurture campaign."

            st.markdown(f"""
            <div class="{card_class}">
                <div class="score-title">{priority_label}</div>
                <div class="score-number">{prob_percent:.1f}%</div>
                <div>Predicted Subscription Probability</div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(min(max(proba, 0.0), 1.0))
            st.info(action_text)
            
            # Summary Metrics Grid
            m1, m2 = st.columns(2)
            m1.metric("Lead Quality", "Grade A" if proba >= 0.5 else ("Grade B" if proba >= 0.3 else "Grade C"))
            m2.metric("Contact Risk", "Low" if campaign <= 3 else "High (Over-contacted)")
            
        elif not MODEL_LOADED:
            st.warning("Please ensure `final_model.pkl` exists to see live predictions.")
        else:
            st.info("👈 Adjust client parameters on the left and click **Calculate Lead Subscription Score**.")