import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------------------------------
# 1. Page Configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="LeadLens AI | Call Center Lead Scoring Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------------
# 2. Universal Modern CSS Design System
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2rem !important;
        max-width: 95% !important;
    }

    /* ==========================================================================
       SIDEBAR CUSTOMIZATION
       ========================================================================== */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 0.25rem;
    }

    .brand-icon {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
    }

    .brand-title {
        color: #F8FAFC !important;
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        margin: 0 !important;
        line-height: 1.2;
    }

    .brand-subtitle {
        color: #64748B !important;
        font-size: 0.78rem !important;
        font-weight: 500;
        margin-top: 2px;
    }

    .sb-divider {
        height: 1px;
        background: linear-gradient(90deg, #1E293B 0%, #334155 50%, #1E293B 100%);
        margin: 1.2rem 0;
    }

    .sb-section-label {
        color: #94A3B8;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
    }

    .metric-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-bottom: 1.2rem;
    }

    .metric-tile {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 0.75rem;
        text-align: center;
    }

    .metric-tile-val {
        color: #38BDF8 !important;
        font-size: 1.4rem !important;
        font-weight: 800 !important;
        line-height: 1.1;
    }

    .metric-tile-lbl {
        color: #94A3B8;
        font-size: 0.68rem;
        font-weight: 600;
        margin-top: 3px;
        text-transform: uppercase;
    }

    .spec-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 10px;
        padding: 0.85rem;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    .spec-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8rem;
    }

    .spec-key {
        color: #94A3B8;
        font-weight: 500;
    }

    .spec-value {
        color: #F1F5F9;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.06);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
    }

    .status-pill {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 20px;
        padding: 0.5rem 1rem;
        color: #34D399;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 1.2rem;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #34D399;
        border-radius: 50%;
        box-shadow: 0 0 8px #34D399;
    }

    /* ==========================================================================
       MAIN DASHBOARD CARDS & COMPONENTS
       ========================================================================== */
    .hero-banner {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }

    .hero-title {
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        font-weight: 400;
    }

    .academic-badge {
        background: rgba(37, 99, 235, 0.15);
        border: 1px solid rgba(37, 99, 235, 0.3);
        color: #60A5FA;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.8rem;
    }

    .glass-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    /* Score Output Badges */
    .score-high {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        color: #34D399;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
    }

    .score-low {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #EF4444;
        color: #FCA5A5;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
    }

    /* Streamlit Input Cleanups */
    .stButton>button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }

    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.4) !important;
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
# 4. Sidebar Component
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">⚡</div>
        <div>
            <div class="brand-title">LeadLens AI</div>
            <div class="brand-subtitle">Call Center Scoring System</div>
        </div>
    </div>
    <div class="sb-divider"></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section-label">Model Metrics</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-container">
        <div class="metric-tile">
            <div class="metric-tile-val">0.79</div>
            <div class="metric-tile-lbl">ROC-AUC</div>
        </div>
        <div class="metric-tile">
            <div class="metric-tile-val">59%</div>
            <div class="metric-tile-lbl">Recall (Yes)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section-label">Pipeline Specs</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="spec-card">
        <div class="spec-row">
            <span class="spec-key">Algorithm</span>
            <span class="spec-value">Logistic Reg.</span>
        </div>
        <div class="spec-row">
            <span class="spec-key">Tuning</span>
            <span class="spec-value">RandomizedSearch</span>
        </div>
        <div class="spec-row">
            <span class="spec-key">Class Weight</span>
            <span class="spec-value">Balanced</span>
        </div>
        <div class="spec-row">
            <span class="spec-key">Dataset</span>
            <span class="spec-value">UCI Bank Mktg</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if MODEL_LOADED:
        st.markdown("""
        <div class="status-pill">
            <div class="pulse-dot"></div>
            <span>Pipeline Engine Active</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("`final_model.pkl` not found. Place model in directory.", icon="🚨")

# ----------------------------------------------------------------------------
# 5. Main Content Area
# ----------------------------------------------------------------------------

# Academic Metadata Header
with st.expander("🎓 Module Information & Declaration of Originality", expanded=False):
    st.markdown("""
    **School of Informatics & IT** | Diploma in Applied Artificial Intelligence  
    **Course:** Machine Learning for Developers (`CAI2C08`) — AY2026/2027  
    **Developer:** Tay Yi Xi  

    ---
    **Declaration:**  
    *I am the originator of this work, and I have appropriately acknowledged all other original sources used as my references. I understand that Plagiarism is an academic offence and disciplinary action will be enforced if violated.*
    """)

# Hero Section
st.markdown("""
<div class="hero-banner">
    <div class="academic-badge">CAI2C08 MLDP Production App</div>
    <div class="hero-title">Predictive Term Deposit Lead Scoring</div>
    <div class="hero-subtitle">Optimize call center resource allocation by ranking client conversion likelihood using Logistic Regression.</div>
</div>
""", unsafe_allow_html=True)

# Main Application Tabs
tab_single, tab_batch, tab_eda = st.tabs(["🎯 Single Lead Scoring", "📊 Batch Processing", "📈 Model Insights"])

# ----------------------------------------------------------------------------
# TAB 1: Single Lead Scoring Form
# ----------------------------------------------------------------------------
with tab_single:
    st.markdown("### Client Profile & Interaction Features")
    
    with st.form("single_lead_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Demographic Profile")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            job = st.selectbox("Job Type", ["admin.", "blue-collar", "technician", "services", "management", "retired", "entrepreneur", "self-employed", "housemaid", "unemployed", "student", "unknown"])
            marital = st.selectbox("Marital Status", ["married", "single", "divorced", "unknown"])
            education = st.selectbox("Education Level", ["university.degree", "high.school", "basic.9y", "professional.course", "basic.4y", "basic.6y", "illiterate", "unknown"])

        with col2:
            st.markdown("##### 📞 Campaign Contacts")
            default = st.selectbox("Credit in Default?", ["no", "yes", "unknown"])
            housing = st.selectbox("Housing Loan?", ["yes", "no", "unknown"])
            loan = st.selectbox("Personal Loan?", ["no", "yes", "unknown"])
            contact = st.selectbox("Contact Communication", ["cellular", "telephone"])
            duration = st.number_input("Last Call Duration (seconds)", min_value=0, max_value=5000, value=240, help="Important: Strongly impacts outcome, but only known after call completes.")

        with col3:
            st.markdown("##### 📉 Economic Indicators")
            emp_var_rate = st.slider("Employment Variation Rate", -3.4, 1.4, 1.1)
            cons_price_idx = st.slider("Consumer Price Index", 92.2, 94.8, 93.9)
            cons_conf_idx = st.slider("Consumer Confidence Index", -50.8, -26.9, -36.4)
            euribor3m = st.slider("Euribor 3 Month Rate", 0.6, 5.1, 4.8)
            nr_employed = st.number_input("Number of Employees", min_value=4900.0, max_value=5300.0, value=5191.0)

        # Context features hidden/defaulted for simplified UI
        month = "may"
        day_of_week = "thu"
        campaign = 1
        pdays = 999
        previous = 0
        poutcome = "nonexistent"

        submit_btn = st.form_submit_button("⚡ Compute Subscription Probability")

    if submit_btn:
        if MODEL_LOADED:
            # Construct DataFrame matching training schema
            input_data = pd.DataFrame([{
                'age': age, 'job': job, 'marital': marital, 'education': education,
                'default': default, 'housing': housing, 'loan': loan, 'contact': contact,
                'month': month, 'day_of_week': day_of_week, 'duration': duration,
                'campaign': campaign, 'pdays': pdays, 'previous': previous,
                'poutcome': poutcome, 'emp.var.rate': emp_var_rate,
                'cons.price.idx': cons_price_idx, 'cons.conf.idx': cons_conf_idx,
                'euribor3m': euribor3m, 'nr.employed': nr_employed
            }])

            try:
                # Predict
                prob = model.predict_proba(input_data)[0][1]
                pred = model.predict(input_data)[0]

                st.markdown("---")
                res_col1, res_col2 = st.columns([1, 2])

                with res_col1:
                    if prob >= 0.5:
                        st.markdown(f"""
                        <div class="score-high">
                            <h4 style="margin:0;">HIGH LIKELIHOOD</h4>
                            <h1 style="margin:0; font-size:2.8rem;">{prob:.1%}</h1>
                            <p style="margin:0;">Recommendation: Priority Call List</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="score-low">
                            <h4 style="margin:0;">LOW LIKELIHOOD</h4>
                            <h1 style="margin:0; font-size:2.8rem;">{prob:.1%}</h1>
                            <p style="margin:0;">Recommendation: Low Priority / Nurture</p>
                        </div>
                        """, unsafe_allow_html=True)

                with res_col2:
                    st.markdown("##### Model Decision Summary")
                    st.progress(float(prob))
                    st.write(f"- **Target Prediction**: {'`Subscribed (Yes)`' if pred == 'yes' or pred == 1 else '`Not Subscribed (No)`'}")
                    st.write(f"- **Confidence Score**: `{prob:.4f}`")
                    st.caption("Note: Probability calculated using standard decision threshold (0.50). Adjust operational threshold depending on call center capacity.")

            except Exception as e:
                st.error(f"Prediction Pipeline Error: {e}")
        else:
            st.warning("Cannot generate prediction because `final_model.pkl` is missing.")

# ----------------------------------------------------------------------------
# TAB 2: Batch Processing
# ----------------------------------------------------------------------------
with tab_batch:
    st.markdown("### Batch Scoring via CSV Upload")
    st.write("Upload a CSV file containing client data matching the UCI Bank Marketing format (`bank-additional.csv`).")

    uploaded_file = st.file_uploader("Upload Lead Dataset", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file, sep=None, engine='python')
            st.dataframe(batch_df.head(5), use_container_width=True)

            if st.button("Score Uploaded Dataset"):
                if MODEL_LOADED:
                    with st.spinner("Processing records..."):
                        preds = model.predict(batch_df)
                        probs = model.predict_proba(batch_df)[:, 1]

                        batch_df['Predicted_Subscription'] = preds
                        batch_df['Conversion_Probability'] = probs

                        st.success("Batch scoring complete!")

                        # KPI Summary
                        kpi1, kpi2, kpi3 = st.columns(3)
                        kpi1.metric("Total Leads Processed", len(batch_df))
                        kpi2.metric("Target High-Probability Leads", sum(probs >= 0.5))
                        kpi3.metric("Avg Conversion Likelihood", f"{np.mean(probs):.1%}")

                        st.dataframe(batch_df[['age', 'job', 'contact', 'duration', 'Conversion_Probability', 'Predicted_Subscription']], use_container_width=True)

                        # CSV Download
                        csv = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Scored Leads CSV", data=csv, file_name="scored_leads.csv", mime="text/csv")
                else:
                    st.error("Model unavailable for batch scoring.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# ----------------------------------------------------------------------------
# TAB 3: Model & Business Insights
# ----------------------------------------------------------------------------
with tab_eda:
    st.markdown("### Business Context & Model Architecture")
    st.markdown("""
    * **Objective**: Rank prospective clients based on their likelihood to subscribe to a term deposit (`y = yes/no`).
    * **Metric Choice**: Focus on **Recall (Class 'Yes')** to minimize missed potential subscribers while maintaining balanced **Precision** to avoid wasted call center hours.
    * **Key Drivers**:
        * **Call Duration**: Single strongest predictor of interest.
        * **Economic Climate**: `euribor3m` and `nr.employed` heavily influence financial decision-making.
        * **Previous Success**: Prior campaign outcome (`poutcome = success`) is a high-yield lead signal.
    """)