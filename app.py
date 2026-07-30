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
# 2. Universal Modern CSS Design System (Sidebar CSS Overhaul)
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    /* ==========================================================================
       SIDEBAR & SCROLLBAR ELIMINATION
       ========================================================================== */
    /* Remove scrollbars completely from sidebar */
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {
        overflow: hidden !important;
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Ensure Sidebar Container fits viewport tightly */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important; /* Rich Dark Slate Background */
        border-right: 1px solid #1E293B !important;
    }

    /* Sidebar Header Component */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 0.25rem;
    }

    .brand-icon {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
    }

    .brand-title {
        color: #F8FAFC !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
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

    /* Compact Divider */
    .sb-divider {
        height: 1px;
        background: linear-gradient(90deg, #1E293B 0%, #334155 50%, #1E293B 100%);
        margin: 1.2rem 0;
    }

    /* Section Label */
    .sb-section-label {
        color: #94A3B8;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
    }

    /* Modern KPI Cards Grid */
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
        color: #38BDF8 !important; /* Bright cyan accent for high contrast */
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        line-height: 1.1;
    }

    .metric-tile-lbl {
        color: #94A3B8;
        font-size: 0.7rem;
        font-weight: 600;
        margin-top: 2px;
        text-transform: uppercase;
    }

    /* Model Specs Key-Value Cards */
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
        font-size: 0.82rem;
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
        font-size: 0.78rem;
    }

    /* Sleek Status Pill */
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
        font-size: 0.82rem;
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
# 4. Redesigned Sidebar (No Scrollbar & High Contrast)
# ----------------------------------------------------------------------------
with st.sidebar:
    # Brand Header
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
    
    # Model Metrics Section
    st.markdown('<div class="sb-section-label">Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-container">
        <div class="metric-tile">
            <div class="metric-tile-val">0.79</div>
            <div class="metric-tile-lbl">ROC-AUC</div>
        </div>
        <div class="metric-tile">
            <div class="metric-tile-val">59%</div>
            <div class="metric-tile-lbl">Recall</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Architecture Details Card
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
    
    # Status Pill Indicator
    if MODEL_LOADED:
        st.markdown("""
        <div class="status-pill">
            <div class="pulse-dot"></div>
            <span>Pipeline Engine Active</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Model File Missing", icon="🚨")