"""
src/dashboard.py — CyberShield Bank Cyber Fraud Intelligence Dashboard
Enterprise-grade Streamlit Frontend: Honeypot Screening, Priority Queue, SHAP XAI & Incident Intake.
"""

from __future__ import annotations

import os
import sys
import time
import json
import io
from datetime import datetime

# Setup project root import paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
src_path = os.path.join(ROOT, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CyberShield | Bank Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Luminous Executive Slate Theme (High Contrast & Consistent Typography) ────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* ── Root Color Variables ── */
    :root {
        --bg-main: #1b263b;
        --card-bg: #22304a;
        --card-border: rgba(0, 245, 160, 0.25);
        --accent-emerald: #00f5a0;
        --accent-cyan: #38bdf8;
        --accent-red: #ff3366;
        --accent-amber: #f59e0b;
        --text-pure: #ffffff;
        --text-muted: #cbd5e1;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #f8fafc !important;
    }

    /* ── Lighter Slate-Blue Gradient Background ── */
    .stApp {
        background: linear-gradient(135deg, #1b283f 0%, #243452 40%, #1a253a 100%) !important;
        color: #f8fafc !important;
    }

    /* ── Hide Default Streamlit Chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Custom Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #162032; }
    ::-webkit-scrollbar-thumb { background: #00f5a0; border-radius: 4px; }

    /* ── Sidebar Styling ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f2d47 0%, #152033 100%) !important;
        border-right: 1px solid rgba(0, 245, 160, 0.25) !important;
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] p {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }

    /* ── Left Radio Navigation Buttons ── */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 8px !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: rgba(36, 52, 82, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        margin-bottom: 6px !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(0, 245, 160, 0.15) !important;
        border-color: #00f5a0 !important;
        transform: translateX(4px) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(135deg, rgba(0, 245, 160, 0.2) 0%, rgba(56, 189, 248, 0.15) 100%) !important;
        border: 1px solid #00f5a0 !important;
        box-shadow: 0 0 16px rgba(0, 245, 160, 0.25) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label div {
        color: #ffffff !important;
        font-size: 0.86rem !important;
        font-weight: 700 !important;
    }

    /* ── Glass Containers & Headers ── */
    .glass-panel {
        background: #202e48 !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 16px !important;
        padding: 22px !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2) !important;
        margin-bottom: 20px !important;
    }

    .header-banner {
        background: linear-gradient(135deg, #243554 0%, #202f4a 100%) !important;
        border: 1px solid rgba(0, 245, 160, 0.3) !important;
        border-radius: 16px !important;
        padding: 22px 26px !important;
        margin-bottom: 22px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15) !important;
    }

    /* ── KPI Metric Cards ── */
    .kpi-card {
        background: #223250 !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 14px !important;
        padding: 18px 20px !important;
        transition: all 0.25s ease !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15) !important;
    }
    .kpi-card:hover {
        border-color: #00f5a0 !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 25px rgba(0, 245, 160, 0.2) !important;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 4px; height: 100%;
        background: #00f5a0;
    }
    .kpi-card.red::before { background: #ff3366; }
    .kpi-card.amber::before { background: #f59e0b; }
    .kpi-card.blue::before { background: #38bdf8; }

    .kpi-label {
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #94a3b8 !important;
        margin-bottom: 4px !important;
    }
    .kpi-value {
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .kpi-sub {
        font-size: 0.75rem !important;
        color: #cbd5e1 !important;
        margin-top: 4px !important;
    }

    /* ── Pulsing Threat Alerts ── */
    .threat-banner-critical {
        background: radial-gradient(circle at center, rgba(255, 51, 102, 0.3) 0%, rgba(180, 10, 40, 0.2) 100%) !important;
        border: 2px solid #ff3366 !important;
        border-radius: 16px !important;
        padding: 24px 26px !important;
        text-align: center !important;
        box-shadow: 0 0 35px rgba(255, 51, 102, 0.4) !important;
    }
    .threat-banner-safe {
        background: radial-gradient(circle at center, rgba(0, 245, 160, 0.2) 0%, rgba(0, 150, 100, 0.1) 100%) !important;
        border: 2px solid #00f5a0 !important;
        border-radius: 16px !important;
        padding: 24px 26px !important;
        text-align: center !important;
        box-shadow: 0 0 30px rgba(0, 245, 160, 0.3) !important;
    }

    .reason-chip {
        background: rgba(245, 158, 11, 0.15) !important;
        border: 1px solid rgba(245, 158, 11, 0.4) !important;
        border-left: 4px solid #f59e0b !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        margin-bottom: 10px !important;
        color: #fef08a !important;
        font-size: 0.9rem !important;
        line-height: 1.5 !important;
    }
    .reason-chip-safe {
        background: rgba(0, 245, 160, 0.15) !important;
        border: 1px solid rgba(0, 245, 160, 0.4) !important;
        border-left: 4px solid #00f5a0 !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        margin-bottom: 10px !important;
        color: #a7f3d0 !important;
        font-size: 0.9rem !important;
        line-height: 1.5 !important;
    }

    /* ── UNIFORM HIGH-CONTRAST FORM INPUTS ── */
    div[data-baseweb="input"], div[data-baseweb="base-input"] {
        background-color: #243554 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="input"] input, div[data-baseweb="base-input"] input {
        background-color: #243554 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #243554 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    textarea {
        background-color: #243554 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }

    /* ── ALL BUTTONS & DOWNLOAD BUTTONS (High Contrast & Visible Text) ── */
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button, div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #00f5a0 0%, #0284c7 100%) !important;
        color: #061524 !important;
        font-weight: 800 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 18px rgba(0, 245, 160, 0.35) !important;
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover, div[data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(0, 245, 160, 0.55) !important;
        color: #000000 !important;
    }

    /* ── Checkboxes & Sliders ── */
    div[data-testid="stCheckbox"] label span {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSlider"] label {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }

    /* ── Clean Dataframe Container ── */
    div[data-testid="stDataFrame"] {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── API Constants ─────────────────────────────────────────────────────────────
API_BASE = "http://127.0.0.1:8000"
DATA_PATH = os.path.join(ROOT, "data", "raw", "synthetic_bank_fraud.csv")
MODEL_PATH = os.path.join(ROOT, "models", "new_fraud_model.pkl")

# ── Sidebar Brand & Navigation ────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 8px 4px 16px 4px; text-align: center;">
            <div style="font-size: 2.5rem; filter: drop-shadow(0 0 16px rgba(0,245,160,0.6));">🛡️</div>
            <div style="font-size: 1.35rem; font-weight: 800; color: #ffffff; letter-spacing: 0.04em; margin-top: 2px;">
                Cyber<span style="color: #00f5a0;">Shield</span>
            </div>
            <div style="font-size: 0.72rem; color: #cbd5e1; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2px;">
                Autonomous Fraud Defense
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # API Telemetry Box
    try:
        r = requests.get(f"{API_BASE}/health", timeout=1.5)
        if r.status_code == 200:
            health_data = r.json()
            st.markdown(
                f"""
                <div style="background: rgba(0, 245, 160, 0.12); border: 1px solid rgba(0, 245, 160, 0.35); border-radius: 12px; padding: 10px 14px; margin-bottom: 18px;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <span style="font-size: 0.72rem; font-weight: 700; color: #cbd5e1; letter-spacing: 0.05em;">API ENGINE</span>
                        <span style="font-size: 0.75rem; font-weight: 800; color: #FF0000;">● LIVE</span>
                    </div>
                    <div style="font-size: 0.76rem; color: #ffffff; margin-top: 4px; font-family: monospace;">
                        Model: <span style="color: #38bdf8; font-weight: 700;">{health_data.get('model', 'XGBoost')}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            raise Exception("Non-200 response")
    except Exception:
        st.markdown(
            """
            <div style="background: rgba(255, 51, 102, 0.15); border: 1px solid rgba(255, 51, 102, 0.4); border-radius: 12px; padding: 10px 14px; margin-bottom: 18px;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 0.72rem; font-weight: 700; color: #cbd5e1;">API ENGINE</span>
                    <span style="font-size: 0.75rem; font-weight: 800; color: #ff3366;">● OFFLINE</span>
                </div>
                <div style="font-size: 0.72rem; color: #f8fafc; margin-top: 4px;">Run: <code>uvicorn backend.app:app</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div style="font-size: 0.72rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">CONTROL WINDOWS</div>', unsafe_allow_html=True)

    # 4 Left Navigation Windows
    nav_options = [
        "🎯 Real-Time Honeypot Monitor",
        "📋 Live Threat Priority Board AI",
        "🔬 Audit Forensic Report(SHAP)",
        "📝 Fraud Incident & Feedback ",
    ]

    selected_window = st.radio(
        "Navigation",
        options=nav_options,
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background: #223250; border: 1px solid rgba(255,255,255,0.12); border-radius: 12px; padding: 14px; font-size: 0.75rem; color: #cbd5e1; line-height: 1.6;">
            <div style="font-weight: 800; color: #ffffff; margin-bottom: 6px;">🛡️ DEFENSE PROTOCOLS</div>
            <div>• Real-time Social Engg Interception</div>
            <div>• Dynamic Expected Loss Prioritization</div>
            <div>• Mathematical XAI Audit Trail</div>
            <div style="margin-top: 10px; font-size: 0.68rem; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 6px;">
                Bank Cyber Fraud Detection v2.0
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# WINDOW 1: REAL-TIME HONEYPOT MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
if selected_window == nav_options[0]:
    st.markdown(
        """
        <div class="header-banner">
            <div>
                <h2 style="margin: 0; font-size: 1.75rem; font-weight: 800; color: #ffffff;">
                    🎯 Real-Time Honeypot Interceptor
                </h2>
                <div style="color: #cbd5e1; font-size: 0.88rem; margin-top: 4px;">
                    Autonomous transaction screening with behavioral honeypots & social engineering heuristics.
                </div>
            </div>
            <div style="background: rgba(0, 245, 160, 0.15); border: 1px solid rgba(0, 245, 160, 0.4); border-radius: 10px; padding: 8px 16px; font-size: 0.8rem; font-weight: 800; color: #00f5a0;">
                ⚡ ACTIVE MONITORING
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1-Click Scenario Preset Injectors
    st.markdown('<div style="font-size: 0.82rem; font-weight: 800; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;">⚡ QUICK PRESET SIMULATORS:</div>', unsafe_allow_html=True)
    sc_col1, sc_col2, sc_col3, sc_col4, sc_col5 = st.columns(5)

    if "preset_data" not in st.session_state:
        st.session_state["preset_data"] = {
            "txn_id": "TXN-DEMO-001",
            "amount": 450000.0,
            "type": "UPI",
            "call_dur": 120,
            "otp_fails": 3,
            "payee_mins": 5.0,
            "new_dev": True,
            "high_ip": True,
            "acc_age": 15,
        }

    with sc_col1:
        if st.button("🚨 Digital Arrest (₹4.5L)", use_container_width=True):
            st.session_state["preset_data"] = {
                "txn_id": f"TXN-DA-{int(time.time())%10000}",
                "amount": 450000.0,
                "type": "UPI",
                "call_dur": 120,
                "otp_fails": 3,
                "payee_mins": 5.0,
                "new_dev": True,
                "high_ip": True,
                "acc_age": 15,
            }
            st.rerun()

    with sc_col2:
        if st.button("📱 SIM-Swap Steal (₹85k)", use_container_width=True):
            st.session_state["preset_data"] = {
                "txn_id": f"TXN-SS-{int(time.time())%10000}",
                "amount": 85000.0,
                "type": "IMPS",
                "call_dur": 5,
                "otp_fails": 4,
                "payee_mins": 25.0,
                "new_dev": True,
                "high_ip": False,
                "acc_age": 220,
            }
            st.rerun()

    with sc_col3:
        if st.button("👤 Shadow Transfer (₹2.2L)", use_container_width=True):
            st.session_state["preset_data"] = {
                "txn_id": f"TXN-ST-{int(time.time())%10000}",
                "amount": 220000.0,
                "type": "RTGS",
                "call_dur": 75,
                "otp_fails": 1,
                "payee_mins": 3.0,
                "new_dev": True,
                "high_ip": True,
                "acc_age": 8,
            }
            st.rerun()

    with sc_col4:
        if st.button("🌐 Mule IP Node (₹1.8L)", use_container_width=True):
            st.session_state["preset_data"] = {
                "txn_id": f"TXN-ML-{int(time.time())%10000}",
                "amount": 180000.0,
                "type": "NEFT",
                "call_dur": 0,
                "otp_fails": 2,
                "payee_mins": 12.0,
                "new_dev": True,
                "high_ip": True,
                "acc_age": 6,
            }
            st.rerun()

    with sc_col5:
        if st.button("🟢 Legitimate Salary (₹65k)", use_container_width=True):
            st.session_state["preset_data"] = {
                "txn_id": f"TXN-OK-{int(time.time())%10000}",
                "amount": 65000.0,
                "type": "NEFT",
                "call_dur": 0,
                "otp_fails": 0,
                "payee_mins": 1800.0,
                "new_dev": False,
                "high_ip": False,
                "acc_age": 750,
            }
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    col_input, col_verdict = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown(
            """
            <div style="font-size: 1rem; font-weight: 800; color: #00f5a0; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 12px;">
                📝 TRANSACTION TELEMETRY
            </div>
            """,
            unsafe_allow_html=True,
        )

        p = st.session_state["preset_data"]

        with st.form("honeypot_eval_form"):
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                in_txn_id = st.text_input("Transaction Reference ID", value=p["txn_id"])
            with col_t2:
                in_amount = st.number_input("Transaction Amount (₹)", min_value=1.0, value=float(p["amount"]), step=5000.0, format="%.2f")

            col_t3, col_t4 = st.columns(2)
            with col_t3:
                type_opts = ["UPI", "IMPS", "NEFT", "RTGS"]
                in_type = st.selectbox("Payment Rail", type_opts, index=type_opts.index(p["type"]) if p["type"] in type_opts else 0)
            with col_t4:
                in_call = st.slider("📞 Active Call Duration (min)", min_value=0, max_value=240, value=int(p["call_dur"]), help="Active phone call duration during transaction")

            col_t5, col_t6 = st.columns(2)
            with col_t5:
                in_otp = st.slider("🔐 Failed OTP Attempts", min_value=0, max_value=8, value=int(p["otp_fails"]))
            with col_t6:
                in_payee_mins = st.number_input("👤 Payee Added (Mins Ago)", min_value=0.0, value=float(p["payee_mins"]), step=1.0)

            col_t7, col_t8, col_t9 = st.columns(3)
            with col_t7:
                in_new_dev = st.checkbox("📱 New Device", value=bool(p["new_dev"]))
            with col_t8:
                in_high_ip = st.checkbox("🌐 Blacklisted IP", value=bool(p["high_ip"]))
            with col_t9:
                in_acc_age = st.number_input("🆕 Account Age (Days)", min_value=0, value=int(p["acc_age"]), step=1)

            submit_btn = st.form_submit_button("⚡ Run Threat Screening", use_container_width=True)

    with col_verdict:
        st.markdown(
            """
            <div style="font-size: 1rem; font-weight: 800; color: #00f5a0; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 12px;">
                🤖 AUTONOMOUS VERDICT & EXPLANATIONS
            </div>
            """,
            unsafe_allow_html=True,
        )

        if submit_btn:
            payload = {
                "transaction_id": in_txn_id,
                "transaction_amount": in_amount,
                "transaction_type": in_type,
                "active_call_duration": float(in_call),
                "otp_failed_attempts": int(in_otp),
                "new_payee_added_mins": float(in_payee_mins),
                "is_new_device": int(in_new_dev),
                "is_high_risk_ip": int(in_high_ip),
                "account_age_days": int(in_acc_age),
            }

            with st.spinner("Screening against AI neural weights and honeypot triggers..."):
                time.sleep(0.3)
                try:
                    resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=8)
                    resp.raise_for_status()
                    result = resp.json()
                except Exception as exc:
                    st.error(f"API Error: {exc}. Please verify backend server is running on port 8000.")
                    st.stop()

            is_fraud = result["is_fraud"]
            prob = result["fraud_probability"]
            alert_lvl = result["alert_level"]
            honeypot_title = result["honeypot_trigger"]
            reasons = result["reasons"]
            exp_loss = result["expected_loss"]

            if is_fraud:
                st.markdown(
                    f"""
                    <div class="threat-banner-critical">
                        <div style="font-size: 2.8rem; margin-bottom: 4px;">🚨</div>
                        <div style="font-size: 1.6rem; font-weight: 900; color: #ff3366; letter-spacing: 0.05em;">
                            {honeypot_title}
                        </div>
                        <div style="font-size: 0.85rem; font-weight: 800; color: #ff8fa3; margin-top: 6px; letter-spacing: 0.06em;">
                            CRITICAL THREAT LEVEL: <span style="background: rgba(255,51,102,0.35); padding: 3px 10px; border-radius: 6px;">{alert_lvl}</span>
                        </div>
                        <div style="font-size: 0.82rem; color: #ffffff; margin-top: 10px; font-family: monospace;">
                            TXN: {result['transaction_id']} &nbsp;|&nbsp; Fraud Confidence: {prob:.1%} &nbsp;|&nbsp; Financial Risk: ₹{exp_loss:,.2f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div style="font-size: 0.85rem; font-weight: 800; color: #f59e0b; text-transform: uppercase; margin-bottom: 8px;">⚠️ Why was this flagged? (Audit Trail)</div>', unsafe_allow_html=True)

                for r_text in reasons:
                    st.markdown(f'<div class="reason-chip">• {r_text}</div>', unsafe_allow_html=True)

            else:
                st.markdown(
                    f"""
                    <div class="threat-banner-safe">
                        <div style="font-size: 2.8rem; margin-bottom: 4px;">✅</div>
                        <div style="font-size: 1.6rem; font-weight: 900; color: #00f5a0; letter-spacing: 0.05em;">
                            TRANSACTION CLEARED
                        </div>
                        <div style="font-size: 0.85rem; font-weight: 800; color: #6ee7b7; margin-top: 6px;">
                            STATUS: <span style="background: rgba(0,245,160,0.25); padding: 3px 10px; border-radius: 6px;">SAFE TO SETTLE</span>
                        </div>
                        <div style="font-size: 0.82rem; color: #ffffff; margin-top: 10px; font-family: monospace;">
                            TXN: {result['transaction_id']} &nbsp;|&nbsp; Fraud Probability: {prob:.2%}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)
                for r_text in reasons:
                    st.markdown(f'<div class="reason-chip-safe">✔ {r_text}</div>', unsafe_allow_html=True)

            # Interactive Risk Breakdown Radar / Polar Chart
            st.markdown("<br>", unsafe_allow_html=True)
            radar_categories = ['Call Duration', 'OTP Failures', 'Payee Recency', 'Device Risk', 'IP Blacklist']
            call_score = min(100.0, (in_call / 120.0) * 100.0)
            otp_score = min(100.0, (in_otp / 4.0) * 100.0)
            payee_score = max(0.0, 100.0 - min(100.0, (in_payee_mins / 30.0) * 100.0))
            dev_score = 100.0 if in_new_dev else 0.0
            ip_score = 100.0 if in_high_ip else 0.0
            radar_values = [call_score, otp_score, payee_score, dev_score, ip_score]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_values + [radar_values[0]],
                theta=radar_categories + [radar_categories[0]],
                fill='toself',
                fillcolor='rgba(255, 51, 102, 0.3)' if is_fraud else 'rgba(0, 245, 160, 0.3)',
                line=dict(color='#ff3366' if is_fraud else '#00f5a0', width=2.5),
                name='Threat Vectors'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], color='#cbd5e1', gridcolor='rgba(255,255,255,0.15)'),
                    angularaxis=dict(color='#ffffff', gridcolor='rgba(255,255,255,0.15)')
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=30, b=30),
                height=260,
                showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        else:
            st.markdown(
                """
                <div class="glass-panel" style="text-align: center; padding: 70px 30px;">
                    <div style="font-size: 3rem; margin-bottom: 14px;">🎯</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #ffffff;">Ready for Screening</div>
                    <div style="font-size: 0.86rem; color: #cbd5e1; margin-top: 6px; max-width: 380px; margin-left: auto; margin-right: auto;">
                        Click a quick scenario preset above or enter transaction parameters to perform autonomous cyber fraud screening.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# WINDOW 2: LIVE THREAT PRIORITY BOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif selected_window == nav_options[1]:
    st.markdown(
        """
        <div class="header-banner">
            <div>
                <h2 style="margin: 0; font-size: 1.75rem; font-weight: 800; color: #ffffff;">
                    📋 Live Threat Priority Queue
                </h2>
                <div style="color: #cbd5e1; font-size: 0.88rem; margin-top: 4px;">
                    Dynamic Financial Loss Prioritization Engine: maximizes fund recovery by ranking active fraud threats.
                </div>
            </div>
            <div style="background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.4); border-radius: 10px; padding: 8px 16px; font-size: 0.8rem; font-weight: 800; color: #38bdf8;">
                📊 EXPECTED LOSS SORTED
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_c1, col_c2, col_c3, col_c4 = st.columns([1.2, 1.2, 1, 1])
    with col_c1:
        pq_threshold = st.slider("Min Fraud Probability Threshold", 0.05, 0.99, 0.50, 0.05)
    with col_c2:
        pq_top_n = st.number_input("Max Threats to Display", min_value=10, max_value=1000, value=50, step=10)
    with col_c3:
        rail_filter = st.multiselect("Filter Payment Rails", ["UPI", "IMPS", "NEFT", "RTGS"], default=["UPI", "IMPS", "NEFT", "RTGS"])
    with col_c4:
        st.write("")
        st.write("")
        refresh_queue = st.button("🔄 Refresh Pipeline", use_container_width=True)

    @st.cache_data(ttl=60, show_spinner=False)
    def fetch_priority_queue_data(data_path: str, model_path: str, threshold: float):
        try:
            from priority_queue import generate_priority_queue
            return generate_priority_queue(data_path=data_path, model_path=model_path, threshold=threshold)
        except Exception as e:
            return str(e)

    if refresh_queue:
        st.cache_data.clear()

    with st.spinner("Computing Expected Financial Loss across active threat queue..."):
        q_result = fetch_priority_queue_data(DATA_PATH, MODEL_PATH, pq_threshold)

    if isinstance(q_result, str):
        st.error(f"Priority Queue Error: {q_result}")
    elif q_result is None or (isinstance(q_result, pd.DataFrame) and q_result.empty):
        st.warning("No threats found above the selected probability threshold. Try lowering the threshold.")
    else:
        raw_df: pd.DataFrame = q_result

        total_threats = len(raw_df)
        total_exposure = raw_df["Expected_Loss"].sum()
        critical_count = (raw_df["Fraud_Probability"] >= 0.85).sum()
        top_loss = raw_df["Expected_Loss"].max() if not raw_df.empty else 0.0

        if total_exposure >= 10_000_000:
            formatted_exposure = f"₹{total_exposure/10_000_000:.2f} Cr"
        elif total_exposure >= 100_000:
            formatted_exposure = f"₹{total_exposure/100_000:.2f} L"
        else:
            formatted_exposure = f"₹{total_exposure:,.0f}"

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(
                f"""
                <div class="kpi-card red">
                    <div class="kpi-label">🚨 Intercepted Threats</div>
                    <div class="kpi-value">{total_threats:,}</div>
                    <div class="kpi-sub">Flagged by AI engine</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi2:
            st.markdown(
                f"""
                <div class="kpi-card amber">
                    <div class="kpi-label">💸 Total Exposure</div>
                    <div class="kpi-value">{formatted_exposure}</div>
                    <div class="kpi-sub">Total ₹{total_exposure:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi3:
            st.markdown(
                f"""
                <div class="kpi-card blue">
                    <div class="kpi-label">⚡ Critical Alerts (≥85%)</div>
                    <div class="kpi-value">{critical_count:,}</div>
                    <div class="kpi-sub">Immediate freeze required</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi4:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">🎯 Max Single Risk</div>
                    <div class="kpi-value">₹{top_loss/100_000:.2f} L</div>
                    <div class="kpi-sub">Top priority transaction</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        col_tbl, col_chart = st.columns([1.6, 1], gap="medium")

        with col_tbl:
            st.markdown('<div style="font-size: 0.95rem; font-weight: 800; color: #00f5a0; margin-bottom: 8px;">HIGH-VALUE THREAT QUEUE (RANKED BY FINANCIAL RISK)</div>', unsafe_allow_html=True)

            table_df = raw_df.head(int(pq_top_n)).copy()
            table_df.insert(0, "Rank", range(1, len(table_df) + 1))

            table_df["Amount_Display"] = table_df["Amount"].apply(lambda x: f"₹{x:,.2f}")
            table_df["Loss_Display"] = table_df["Expected_Loss"].apply(lambda x: f"₹{x:,.2f}")
            table_df["Prob_Display"] = table_df["Fraud_Probability"].apply(lambda x: f"{x:.2%}")

            clean_display_df = table_df[[
                "Rank", "Tx_ID", "Amount_Display", "Call_Mins", "OTP_Fails", "Prob_Display", "Loss_Display"
            ]].rename(columns={
                "Tx_ID": "Transaction ID",
                "Amount_Display": "Amount (₹)",
                "Call_Mins": "Call Duration (Min)",
                "OTP_Fails": "OTP Failures",
                "Prob_Display": "AI Confidence",
                "Loss_Display": "Expected Loss (₹)",
            })

            st.dataframe(
                clean_display_df,
                use_container_width=True,
                hide_index=True,
                height=420,
            )

            csv_data = raw_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Export Complete Threat Queue (CSV)",
                data=csv_data,
                file_name=f"cyber_fraud_priority_queue_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )

        with col_chart:
            st.markdown('<div style="font-size: 0.95rem; font-weight: 800; color: #38bdf8; margin-bottom: 8px;">EXPECTED LOSS DISTRIBUTION</div>', unsafe_allow_html=True)

            fig_dist = px.histogram(
                raw_df.head(100),
                x="Expected_Loss",
                nbins=25,
                color_discrete_sequence=['#00f5a0'],
                labels={"Expected_Loss": "Expected Financial Loss (₹)"},
            )
            fig_dist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(36, 53, 84, 0.7)',
                font=dict(color='#cbd5e1'),
                margin=dict(l=20, r=20, t=20, b=20),
                height=220,
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            st.markdown('<div style="font-size: 0.85rem; font-weight: 800; color: #ffffff; margin-top: 10px; margin-bottom: 4px;">Top Attack Modalities in Queue</div>', unsafe_allow_html=True)
            digital_arrest_count = int((raw_df["Call_Mins"] >= 60).sum())
            otp_theft_count = int((raw_df["OTP_Fails"] >= 3).sum())
            other_mod = max(0, total_threats - (digital_arrest_count + otp_theft_count))

            fig_donut = go.Figure(data=[go.Pie(
                labels=['Digital Arrest Scam', 'OTP / SIM Swap Theft', 'Account Takeover / Mule'],
                values=[digital_arrest_count, otp_theft_count, other_mod],
                hole=.6,
                marker=dict(colors=['#ff3366', '#f59e0b', '#0284c7'])
            )])
            fig_donut.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff', size=11),
                margin=dict(l=10, r=10, t=10, b=10),
                height=180,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, font=dict(color="#ffffff", size=11))
            )
            st.plotly_chart(fig_donut, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# WINDOW 3: AUDITOR FORENSIC REPORT (SHAP XAI)
# ═══════════════════════════════════════════════════════════════════════════════
elif selected_window == nav_options[2]:
    st.markdown(
        """
        <div class="header-banner">
            <div>
                <h2 style="margin: 0; font-size: 1.75rem; font-weight: 800; color: #ffffff;">
                    🔬 Auditor Forensic Report (SHAP XAI)
                </h2>
                <div style="color: #cbd5e1; font-size: 0.88rem; margin-top: 4px;">
                    Explainable AI (XAI) Matrix: Shapley additive explanations backing each neural decision with mathematical proof.
                </div>
            </div>
            <div style="background: rgba(168, 85, 247, 0.15); border: 1px solid rgba(168, 85, 247, 0.4); border-radius: 10px; padding: 8px 16px; font-size: 0.8rem; font-weight: 800; color: #c084fc;">
                📐 REGULATORY COMPLIANT
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    shap_output_file = os.path.join(ROOT, "reports", "figures", "advanced_shap_audit.png")
    rf_model_file = os.path.join(ROOT, "models", "random_forest.pkl")
    scaler_model_file = os.path.join(ROOT, "models", "scaler.pkl")

    col_x1, col_x2, col_x3 = st.columns([1.5, 1.2, 1])
    with col_x1:
        audit_txn_id = st.text_input("Audit Case / Transaction ID", value="TXN-AUDIT-2026-X1")
    with col_x2:
        audit_bg_samples = st.slider("SHAP Baseline Population Size", 50, 400, 200, 50)
    with col_x3:
        st.write("")
        st.write("")
        gen_shap_btn = st.button("🔬 Compute SHAP Explanations", use_container_width=True)

    @st.cache_data(ttl=180, show_spinner=False)
    def execute_shap_pipeline(txn_id: str, n_samples: int, model_path: str, scaler_path: str, out_path: str):
        try:
            from explainer import FraudExplainer, generate_digital_arrest_sample, generate_background_data
            explainer = FraudExplainer(model_path=model_path, scaler_path=scaler_path)
            sample_data = generate_digital_arrest_sample()
            bg_data = generate_background_data(n_samples=n_samples)
            explainer.generate_audit_dashboard(txn_id, sample_data, bg_data, output_path=out_path)
            return True, out_path
        except Exception as exc:
            return False, str(exc)

    if gen_shap_btn:
        with st.spinner("Computing Shapley values across multidimensional feature space (approx 10-20s)..."):
            success, msg = execute_shap_pipeline(
                audit_txn_id, audit_bg_samples, rf_model_file, scaler_model_file, shap_output_file
            )
            if success:
                st.success("✅ SHAP Forensic Audit Matrix successfully computed!")
            else:
                st.error(f"SHAP generation error: {msg}")

    if os.path.exists(shap_output_file):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="glass-panel">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;">
                    <div style="font-size: 1rem; font-weight: 800; color: #00f5a0;">
                        📊 XAI MULTI-QUADRANT AUDIT CANVAS
                    </div>
                    <div style="font-size: 0.78rem; color: #cbd5e1; font-weight: 600;">
                        Features: Call Duration, Amount, OTP Failures, Payee Recency, Device, IP
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        with open(shap_output_file, "rb") as f:
            img_data = f.read()

        st.image(img_data, use_container_width=True)

        st.markdown(
            """
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 18px; font-size: 0.83rem; color: #f1f5f9; line-height: 1.6;">
                    <div style="background: #243554; padding: 14px; border-radius: 10px; border-left: 3px solid #00f5a0;">
                        <strong style="color: #00f5a0;">Quadrant 1: Global Feature Impact</strong><br>
                        Shows which behavioral markers exert the largest global SHAP pull across all historical bank transactions.
                    </div>
                    <div style="background: #243554; padding: 14px; border-radius: 10px; border-left: 3px solid #38bdf8;">
                        <strong style="color: #38bdf8;">Quadrant 2: Decision Convergence Plot</strong><br>
                        Traces the step-by-step risk accumulation from baseline prior probability to final frozen verdict.
                    </div>
                    <div style="background: #243554; padding: 14px; border-radius: 10px; border-left: 3px solid #c084fc;">
                        <strong style="color: #c084fc;">Quadrant 3: Relative Feature Importance</strong><br>
                        Random Forest tree ensemble feature split importance affirming SHAP attribution weights.
                    </div>
                    <div style="background: #243554; padding: 14px; border-radius: 10px; border-left: 3px solid #ff3366;">
                        <strong style="color: #ff3366;">Quadrant 4: Explanation Heatmap</strong><br>
                        Clustered local attributions exposing syndicate-level social engineering attack patterns.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.download_button(
            label="⬇️ Download Official Auditor Proof Report (PNG)",
            data=img_data,
            file_name=f"audit_shap_proof_{audit_txn_id}.png",
            mime="image/png",
        )
    else:
        st.info("Click 'Compute SHAP Explanations' above to generate the forensic audit dashboard.")


# ═══════════════════════════════════════════════════════════════════════════════
# WINDOW 4: FRAUD INCIDENT & FEEDBACK PORTAL
# ═══════════════════════════════════════════════════════════════════════════════
elif selected_window == nav_options[3]:
    st.markdown(
        """
        <div class="header-banner">
            <div>
                <h2 style="margin: 0; font-size: 1.75rem; font-weight: 800; color: #ffffff;">
                    📝 Fraud Incident & Feedback Portal
                </h2>
                <div style="color: #cbd5e1; font-size: 0.88rem; margin-top: 4px;">
                    Incident intake for missed fraud escalations, false positive disputes & regulatory reporting.
                </div>
            </div>
            <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 10px; padding: 8px 16px; font-size: 0.8rem; font-weight: 800; color: #f59e0b;">
                🔒 24/7 INCIDENT INTAKE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_fb_form, col_fb_info = st.columns([1.2, 1], gap="large")

    with col_fb_form:
        st.markdown('<div style="font-size: 1rem; font-weight: 800; color: #00f5a0; margin-bottom: 12px;">INCIDENT DISPUTE & ESCALATION FORM</div>', unsafe_allow_html=True)

        with st.form("incident_report_form", clear_on_submit=True):
            f_txn_id = st.text_input("Flagged Transaction Reference ID *", placeholder="e.g. TXN-9988234")
            f_reporter = st.text_input("Officer / Customer Identifier *", placeholder="e.g. Off. Jane Doe (ID: SEC-889)")
            f_email = st.text_input("Notification Contact Email (optional)", placeholder="security.desk@bank.in")
            
            f_type = st.selectbox(
                "Incident Classification *",
                ["MISSED_FRAUD", "FALSE_POSITIVE", "CYBER_ATTACK_EMERGENCY", "OTHER"],
                help="Select the exact nature of the disputed transaction."
            )

            f_desc = st.text_area(
                "Detailed Incident Narrative & Indicators Observed *",
                placeholder="Detail the timeline, customer communications, suspicious payee details, or justification for dispute...",
                height=150,
            )

            submit_fb = st.form_submit_button("📨 Transmit Incident Report", use_container_width=True)

        if submit_fb:
            if not f_txn_id.strip() or not f_reporter.strip() or not f_desc.strip():
                st.error("Please fill in all required fields marked with *.")
            else:
                fb_payload = {
                    "transaction_id": f_txn_id.strip(),
                    "reported_by": f_reporter.strip(),
                    "complaint_type": f_type,
                    "description": f_desc.strip(),
                    "contact_email": f_email.strip() if f_email.strip() else None,
                }
                try:
                    res = requests.post(f"{API_BASE}/feedback", json=fb_payload, timeout=8)
                    res.raise_for_status()
                    fb_res = res.json()
                    st.success(f"✅ Incident report logged with Ticket ID: **{fb_res['feedback_id']}**")
                    st.info(fb_res["message"])
                except Exception as exc:
                    st.error(f"Failed to transmit report: {exc}")

    with col_fb_info:
        st.markdown('<div style="font-size: 1rem; font-weight: 800; color: #38bdf8; margin-bottom: 12px;">SECURITY PROTOCOLS & SLA</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="glass-panel" style="font-size: 0.85rem; color: #f1f5f9; line-height: 1.8;">
                <div style="margin-bottom: 12px;">
                    <strong style="color: #ff3366;">🚨 Priority 1 — Active Digital Arrest / Impersonation</strong><br>
                    Mandatory account freeze initiated within <strong>15 minutes</strong>. Automatic cyber crime node alert sent to CERT-In.
                </div>
                <div style="margin-bottom: 12px;">
                    <strong style="color: #f59e0b;">⚠️ Priority 2 — Missed Fraud Escalation</strong><br>
                    Fraud desk forensic triage within <strong>4 hours</strong>. Fund trace initiated via inter-bank fast settlement network.
                </div>
                <div style="margin-bottom: 12px;">
                    <strong style="color: #00f5a0;">🟢 Priority 3 — False Positive Dispute</strong><br>
                    Customer identity re-verification completed within <strong>24 hours</strong> with temporary risk ceiling elevation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        log_file_path = os.path.join(ROOT, "reports", "feedback_log.jsonl")
        if os.path.exists(log_file_path):
            st.markdown('<div style="font-size: 0.9rem; font-weight: 800; color: #ffffff; margin-top: 14px; margin-bottom: 8px;">RECENT AUDIT INTAKE LOG</div>', unsafe_allow_html=True)
            try:
                entries = []
                with open(log_file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            entries.append(json.loads(line.strip()))
                if entries:
                    df_log = pd.DataFrame(entries[-8:][::-1])[["feedback_id", "transaction_id", "complaint_type", "submitted_at"]]
                    df_log["submitted_at"] = pd.to_datetime(df_log["submitted_at"]).dt.strftime("%d %b, %H:%M")
                    st.dataframe(df_log.rename(columns={
                        "feedback_id": "Ticket ID",
                        "transaction_id": "Tx ID",
                        "complaint_type": "Category",
                        "submitted_at": "Timestamp"
                    }), use_container_width=True, hide_index=True)
            except Exception:
                pass
