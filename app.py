"""
Term Deposit Subscription Predictor
------------------------------------
Streamlit app for CAI2C08 Machine Learning for Developers — Project Codes submission.

Loads the tuned Logistic Regression pipeline exported from the project notebook
(`final_model.pkl`) and lets a call-centre user enter a client profile to get a
predicted probability of term-deposit subscription, so calls can be prioritised.

To run locally:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Subscription Predictor | LeadLens",
    page_icon="📞",
    layout="centered",
)

PRIMARY = "#0B3D2E"      # deep bank-teal green
ACCENT = "#C89B3C"       # muted gold, "value" accent
BG_CARD = "#F6F5F1"      # warm off-white card background
TEXT_MUTED = "#5B6660"

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: #FFFFFF;
        }}
        .lead-header {{
            padding: 1.6rem 1.8rem;
            border-radius: 14px;
            background: linear-gradient(135deg, {PRIMARY} 0%, #12594A 100%);
            color: white;
            margin-bottom: 1.6rem;
        }}
        .lead-header h1 {{
            font-size: 1.7rem;
            margin-bottom: 0.2rem;
            font-weight: 700;
        }}
        .lead-header p {{
            color: #DCE9E3;
            font-size: 0.95rem;
            margin: 0;
        }}
        .result-card {{
            padding: 1.4rem 1.6rem;
            border-radius: 14px;
            background-color: {BG_CARD};
            border-left: 6px solid {ACCENT};
            margin-top: 1rem;
        }}
        .metric-label {{
            color: {TEXT_MUTED};
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .section-label {{
            color: {PRIMARY};
            font-weight: 700;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 1.4rem;
            margin-bottom: 0.4rem;
        }}
        div.stButton > button {{
            background-color: {PRIMARY};
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.6rem 1.4rem;
            border: none;
        }}
        div.stButton > button:hover {{
            background-color: #12594A;
            color: white;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="lead-header">
        <h1>📞 LeadLens — Subscription Predictor</h1>
        <p>Rank clients by likelihood to subscribe to a term deposit, so your call-centre team spends time on the leads most likely to convert.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("final_model.pkl")

try:
    model = load_model()
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False
    st.error(
        "Could not find `final_model.pkl`. Run the project notebook through Section 6.3 "
        "(Export Final Model) first, then place the generated file next to this app."
    )

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
st.markdown('<div class="section-label">Client Profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 88, 38)
    job = st.selectbox(
        "Job",
        ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
         'retired', 'self-employed', 'services', 'student', 'technician',
         'unemployed', 'unknown'],
        index=0,
    )
    marital = st.selectbox("Marital status", ['divorced', 'married', 'single', 'unknown'], index=1)
    education = st.selectbox(
        "Education",
        ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
         'professional.course', 'university.degree', 'unknown'],
        index=6,
    )
    default = st.selectbox("Has credit in default?", ['no', 'yes', 'unknown'], index=0)
    housing = st.selectbox("Has housing loan?", ['no', 'yes', 'unknown'], index=0)
    loan = st.selectbox("Has personal loan?", ['no', 'yes', 'unknown'], index=0)

with col2:
    contact = st.selectbox("Contact method", ['cellular', 'telephone'], index=0)
    month = st.selectbox(
        "Contact month",
        ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        index=4,
    )
    day_of_week = st.selectbox("Contact day of week", ['mon', 'tue', 'wed', 'thu', 'fri'], index=0)
    campaign = st.slider("Contacts made this campaign", 1, 35, 2)
    previous = st.slider("Contacts made before this campaign", 0, 6, 0)
    contacted_before_label = st.radio("Previously contacted (any earlier campaign)?", ["No", "Yes"], index=0, horizontal=True)
    poutcome = st.selectbox("Outcome of previous campaign", ['nonexistent', 'failure', 'success'], index=0)

st.markdown('<div class="section-label">Current Economic Conditions</div>', unsafe_allow_html=True)
col3, col4, col5 = st.columns(3)
with col3:
    euribor3m = st.slider("Euribor 3-month rate", 0.6, 5.1, 4.86, step=0.01)
with col4:
    cons_price_idx = st.slider("Consumer price index", 92.2, 94.8, 93.75, step=0.01)
with col5:
    cons_conf_idx = st.slider("Consumer confidence index", -50.8, -26.9, -41.8, step=0.1)

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
# Prediction
# ----------------------------------------------------------------------------
st.markdown("---")
predict_clicked = st.button("🔮 Predict Subscription Likelihood", use_container_width=True, disabled=not MODEL_LOADED)

if predict_clicked and MODEL_LOADED:
    proba = model.predict_proba(input_row)[0, 1]
    pred = model.predict(input_row)[0]

    label = "Likely to Subscribe ✅" if pred == 1 else "Unlikely to Subscribe"
    st.markdown(
        f"""
        <div class="result-card">
            <div class="metric-label">Predicted outcome</div>
            <h2 style="margin-top:0.2rem; color:{PRIMARY};">{label}</h2>
            <div class="metric-label" style="margin-top:0.8rem;">Predicted probability of subscribing</div>
            <h1 style="color:{ACCENT}; margin-top:0.1rem;">{proba*100:.1f}%</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(proba, 0.0), 1.0))

    if proba >= 0.5:
        st.success("Recommendation: prioritise this client in today's call list.")
    elif proba >= 0.3:
        st.info("Recommendation: moderate priority — call if capacity allows.")
    else:
        st.warning("Recommendation: low priority for this campaign.")

elif not MODEL_LOADED:
    pass
else:
    st.caption("Set the client profile above and click **Predict Subscription Likelihood** to see a result.")

st.markdown("---")
st.caption(
    "Model: Logistic Regression (class-weight balanced, tuned via RandomizedSearchCV) · "
    "Test-set F1 ≈ 0.42, Recall ≈ 0.59, ROC-AUC ≈ 0.79 · "
    "Trained on the UCI Bank Marketing (bank-additional) dataset."
)
