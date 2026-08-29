"""
src/app.py — Streamlit UI for Bank Cyber Fraud Detection
Author  : Ekansh Gupta (Frontend Architect)
Purpose : Four-tab Streamlit dashboard:
          Tab 1 — Real-Time Monitor  : Honeypot transaction form → POST /predict
          Tab 2 — Live Threat Board  : Priority Queue from Aryan's generate_priority_queue()
          Tab 3 — Auditor Report     : SHAP XAI dashboard from Varshith's FraudExplainer
          Tab 4 — Complaint Portal   : Missed-fraud report form → POST /feedback
"""

from __future__ import annotations

import os
import sys
import time
import io

# Make project root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import pandas as pd
import requests
import streamlit as st

# ── Streamlit Page Config (MUST be first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="CyberShield — Bank Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Dark gradient background ── */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #0f2a1e 100%);
        color: #e2e8f0;
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e1a 0%, #0d1b2a 100%);
        border-right: 1px solid rgba(0,255,136,0.12);
    }

    /* ── Tab bar ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03);
        border-radius: 14px;
        padding: 6px;
        gap: 4px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 22px;
        font-weight: 600;
        font-size: 0.88rem;
        color: #94a3b8;
        transition: all 0.25s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00ff88 0%, #00d4aa 100%) !important;
        color: #0a0e1a !important;
        box-shadow: 0 4px 20px rgba(0,255,136,0.35);
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 18px;
        backdrop-filter: blur(10px);
    }

    /* ── Form inputs ── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    .stTextArea textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #00ff88 0%, #00c47a 100%);
        color: #0a0e1a;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 10px 28px;
        font-size: 0.92rem;
        transition: all 0.25s ease;
        box-shadow: 0 4px 15px rgba(0,255,136,0.3);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,255,136,0.45);
    }

    /* ── Dataframe ── */
    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Custom cards ── */
    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        margin-bottom: 16px;
    }
    .alert-critical {
        background: linear-gradient(135deg, rgba(255,20,20,0.18) 0%, rgba(180,0,0,0.12) 100%);
        border: 2px solid rgba(255,50,50,0.6);
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        animation: pulse-red 1.5s ease-in-out infinite;
    }
    @keyframes pulse-red {
        0%,100% { box-shadow: 0 0 20px rgba(255,0,0,0.3); }
        50%      { box-shadow: 0 0 60px rgba(255,0,0,0.7); }
    }
    .alert-safe {
        background: linear-gradient(135deg, rgba(0,255,136,0.12) 0%, rgba(0,180,100,0.08) 100%);
        border: 2px solid rgba(0,255,136,0.4);
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
    }
    .reason-pill {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,200,0,0.3);
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.9rem;
        line-height: 1.5;
        color: #f1c40f;
    }
    .reason-safe-pill {
        background: rgba(0,255,136,0.06);
        border: 1px solid rgba(0,255,136,0.25);
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.9rem;
        color: #00ff88;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #00ff88;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(0,255,136,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────
API_BASE   = "http://127.0.0.1:8000"
DATA_PATH  = os.path.join(ROOT, "data", "raw", "synthetic_bank_fraud.csv")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 20px 0 10px 0;">
            <div style="font-size:3rem;">🛡️</div>
            <div style="font-size:1.4rem; font-weight:800; color:#00ff88; letter-spacing:0.05em;">CyberShield</div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">Bank Fraud Intelligence Platform</div>
        </div>
        <hr style="border-color: rgba(0,255,136,0.15); margin:12px 0;">
        """,
        unsafe_allow_html=True,
    )

    # API health check
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        if r.status_code == 200:
            data = r.json()
            st.markdown(
                f"""
                <div class="glass-card" style="padding:14px;">
                    <div style="font-size:0.75rem; color:#64748b; margin-bottom:6px;">API STATUS</div>
                    <div style="color:#00ff88; font-weight:700; font-size:0.9rem;">🟢 Online</div>
                    <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">Model: {data.get('model','—')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except Exception:
        st.markdown(
            """
            <div class="glass-card" style="padding:14px; border-color:rgba(255,50,50,0.3);">
                <div style="font-size:0.75rem; color:#64748b; margin-bottom:6px;">API STATUS</div>
                <div style="color:#ff5050; font-weight:700; font-size:0.9rem;">🔴 Offline</div>
                <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">Start backend: uvicorn backend.app:app</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <hr style="border-color: rgba(255,255,255,0.06); margin:14px 0;">
        <div style="font-size:0.75rem; color:#475569; padding:0 4px;">
            <div style="margin-bottom:6px; color:#64748b; font-weight:600;">TEAM</div>
            <div>🤖 Aryan — Priority Queue</div>
            <div>🔬 Varshith — SHAP XAI</div>
            <div>⚙️ Sunil — FastAPI Backend</div>
            <div>🎨 Ekansh — Frontend</div>
        </div>
        <hr style="border-color: rgba(255,255,255,0.06); margin:14px 0;">
        <div style="font-size:0.7rem; color:#334155; text-align:center;">
            Bank Cyber Fraud Detection v1.0<br>© 2026 CyberShield Team
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="padding: 10px 0 30px 0;">
        <h1 style="font-size:2.4rem; font-weight:800; margin:0;
                   background: linear-gradient(135deg,#00ff88,#00d4ff,#a855f7);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                   background-clip:text;">
            🛡️ CyberShield — Bank Fraud Intelligence
        </h1>
        <p style="color:#64748b; margin-top:6px; font-size:1rem;">
            Real-time AI fraud detection · Priority risk queue · SHAP explainability · Customer portal
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯  Real-Time Monitor",
    "📋  Live Threat Board",
    "🔬  Auditor Report",
    "📝  Complaint Portal",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — REAL-TIME MONITOR (Honeypot)
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        """
        <div class="glass-card">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
                <span style="font-size:2rem;">🎯</span>
                <div>
                    <div style="font-size:1.25rem; font-weight:700; color:#e2e8f0;">Real-Time Honeypot Monitor</div>
                    <div style="font-size:0.85rem; color:#64748b;">Submit a transaction to the AI fraud detection engine</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-header">Transaction Details</div>', unsafe_allow_html=True)

        with st.form("honeypot_form", clear_on_submit=False):
            txn_id = st.text_input(
                "Transaction ID",
                value="TXN-DEMO-001",
                help="Unique identifier for this transaction",
            )
            txn_amount = st.number_input(
                "Transaction Amount (₹)",
                min_value=1.0,
                value=450000.0,
                step=1000.0,
                format="%.2f",
            )
            txn_type = st.selectbox(
                "Transaction Type",
                options=["UPI", "IMPS", "NEFT", "RTGS"],
                index=0,
            )

            st.markdown('<div style="margin-top:6px; font-size:0.8rem; color:#64748b; font-weight:600;">CYBER FRAUD INDICATORS</div>', unsafe_allow_html=True)

            call_duration = st.slider(
                "📞 Active Call Duration (minutes)",
                min_value=0,
                max_value=180,
                value=120,
                help="Duration of any active phone call during this transaction",
            )
            otp_fails = st.slider(
                "🔐 OTP Failed Attempts",
                min_value=0,
                max_value=10,
                value=3,
            )
            payee_mins = st.number_input(
                "👤 New Payee Added (minutes ago)",
                min_value=0.0,
                value=5.0,
                step=1.0,
                help="How many minutes ago was this payee added? (Low = suspicious)",
            )

            col_dev, col_ip = st.columns(2)
            with col_dev:
                is_new_device = st.checkbox("📱 New / Unknown Device", value=True)
            with col_ip:
                is_high_risk_ip = st.checkbox("🌐 High-Risk IP", value=True)

            account_age = st.number_input(
                "🆕 Account Age (days)",
                min_value=0,
                value=15,
                step=1,
            )

            submitted = st.form_submit_button("🚨 Analyse Transaction", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-header">AI Verdict</div>', unsafe_allow_html=True)

        if submitted:
            payload = {
                "transaction_id":       txn_id,
                "transaction_amount":   txn_amount,
                "transaction_type":     txn_type,
                "active_call_duration": float(call_duration),
                "otp_failed_attempts":  int(otp_fails),
                "new_payee_added_mins": float(payee_mins),
                "is_new_device":        int(is_new_device),
                "is_high_risk_ip":      int(is_high_risk_ip),
                "account_age_days":     int(account_age),
            }

            with st.spinner("Running AI inference…"):
                time.sleep(0.4)  # UX breathing room
                try:
                    resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
                    resp.raise_for_status()
                    result = resp.json()
                except requests.exceptions.ConnectionError:
                    st.error(
                        "**Cannot reach the backend.** "
                        "Please start it with:\n```bash\nuvicorn backend.app:app --reload\n```"
                    )
                    st.stop()
                except Exception as exc:
                    st.error(f"API error: {exc}")
                    st.stop()

            is_fraud   = result["is_fraud"]
            prob       = result["fraud_probability"]
            alert      = result["alert_level"]
            honeypot   = result["honeypot_trigger"]
            reasons    = result["reasons"]
            exp_loss   = result["expected_loss"]

            # ── MASSIVE RED WARNING ──────────────────────────────────────────
            if is_fraud:
                alert_color = {"CRITICAL": "#ff1414", "HIGH": "#ff6b14", "MEDIUM": "#ffb414"}.get(alert, "#ff4444")
                st.markdown(
                    f"""
                    <div class="alert-critical">
                        <div style="font-size:3.5rem; margin-bottom:8px;">🚨</div>
                        <div style="font-size:2rem; font-weight:900; color:{alert_color};
                                    text-shadow: 0 0 30px {alert_color}; letter-spacing:0.06em;">
                            {honeypot}
                        </div>
                        <div style="font-size:1rem; color:#ff8080; margin-top:8px; font-weight:600;">
                            ALERT LEVEL: {alert}
                        </div>
                        <div style="font-size:0.85rem; color:#94a3b8; margin-top:6px;">
                            TXN: {result['transaction_id']} &nbsp;|&nbsp;
                            Prob: {prob:.1%} &nbsp;|&nbsp;
                            Expected Loss: ₹{exp_loss:,.2f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-header" style="color:#f1c40f;">⚠️ Why was this flagged?</div>',
                    unsafe_allow_html=True,
                )
                for reason in reasons:
                    st.markdown(
                        f'<div class="reason-pill">• {reason}</div>',
                        unsafe_allow_html=True,
                    )

            else:
                st.markdown(
                    f"""
                    <div class="alert-safe">
                        <div style="font-size:3rem; margin-bottom:8px;">✅</div>
                        <div style="font-size:1.5rem; font-weight:800; color:#00ff88;">
                            TRANSACTION CLEARED
                        </div>
                        <div style="font-size:0.85rem; color:#94a3b8; margin-top:8px;">
                            TXN: {result['transaction_id']} &nbsp;|&nbsp; Prob: {prob:.1%}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)
                for reason in reasons:
                    st.markdown(
                        f'<div class="reason-safe-pill">✔ {reason}</div>',
                        unsafe_allow_html=True,
                    )

            # ── Probability gauge ────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            gauge_color = "#ff1414" if prob > 0.65 else ("#ffb414" if prob > 0.50 else "#00ff88")
            st.markdown(
                f"""
                <div class="glass-card" style="padding:16px;">
                    <div style="font-size:0.75rem; color:#64748b; margin-bottom:8px;">FRAUD PROBABILITY</div>
                    <div style="background:rgba(255,255,255,0.07); border-radius:999px; height:12px; overflow:hidden;">
                        <div style="background: linear-gradient(90deg, {gauge_color}, {gauge_color}aa);
                                    width:{prob*100:.1f}%; height:100%; border-radius:999px;
                                    transition: width 1s ease;"></div>
                    </div>
                    <div style="text-align:right; font-size:1.1rem; font-weight:700;
                                color:{gauge_color}; margin-top:6px;">{prob:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            st.markdown(
                """
                <div class="glass-card" style="text-align:center; padding:60px 30px;">
                    <div style="font-size:3rem; margin-bottom:12px;">🎯</div>
                    <div style="color:#475569; font-size:1rem;">
                        Fill in the transaction form and click<br>
                        <strong style="color:#00ff88;">Analyse Transaction</strong> to get the AI verdict.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE THREAT BOARD (Priority Queue)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        """
        <div class="glass-card">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
                <span style="font-size:2rem;">📋</span>
                <div>
                    <div style="font-size:1.25rem; font-weight:700; color:#e2e8f0;">Live Threat Board — Priority Queue</div>
                    <div style="font-size:0.85rem; color:#64748b;">Aryan's Expected Financial Loss engine, ranked by investigator priority</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])
    with col_ctrl1:
        pq_threshold = st.slider("Fraud Probability Threshold", 0.1, 0.95, 0.5, 0.05, key="pq_thresh")
    with col_ctrl2:
        pq_top_n = st.number_input("Show Top N Threats", min_value=5, max_value=500, value=25, key="pq_n")
    with col_ctrl3:
        refresh_pq = st.button("🔄 Refresh Queue", key="refresh_pq")

    @st.cache_data(ttl=30, show_spinner=False)
    def load_priority_queue(data_path: str, model_path: str, threshold: float):
        try:
            from priority_queue import generate_priority_queue
            return generate_priority_queue(
                data_path=data_path,
                model_path=model_path,
                threshold=threshold,
            )
        except Exception as exc:
            return str(exc)

    if refresh_pq:
        st.cache_data.clear()

    with st.spinner("Loading priority queue…"):
        pq_data_path  = os.path.join(ROOT, "data", "raw", "synthetic_bank_fraud.csv")
        pq_model_path = os.path.join(ROOT, "models", "new_fraud_model.pkl")
        queue_result  = load_priority_queue(pq_data_path, pq_model_path, pq_threshold)

    if isinstance(queue_result, str):
        st.error(f"Priority Queue Error: {queue_result}")
        st.info(
            "Make sure:\n"
            "- `data/raw/synthetic_bank_fraud.csv` exists (run `python src/generate_data.py`)\n"
            "- `models/new_fraud_model.pkl` exists (run `python src/train.py`)"
        )
    elif queue_result is None or (isinstance(queue_result, pd.DataFrame) and queue_result.empty):
        st.warning("No threats found above the selected probability threshold. Try lowering it.")
    else:
        queue_df: pd.DataFrame = queue_result

        # ── KPI cards ───────────────────────────────────────────────────────
        total_threats = len(queue_df)
        total_exposure = queue_df["Expected_Loss"].sum()
        critical_count = (queue_df["Fraud_Probability"] >= 0.85).sum()
        top_loss = queue_df["Expected_Loss"].iloc[0] if not queue_df.empty else 0

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("🚨 Active Threats", f"{total_threats:,}")
        kpi2.metric("💸 Total Exposure", f"₹{total_exposure:,.0f}")
        kpi3.metric("🔴 Critical (≥85%)", f"{critical_count:,}")
        kpi4.metric("⚡ Highest Loss", f"₹{top_loss:,.0f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Formatted display dataframe ──────────────────────────────────────
        display_df = queue_df.head(int(pq_top_n)).copy()
        display_df.insert(0, "Priority", range(1, len(display_df) + 1))

        # Format rupee columns
        display_df["Amount"]        = display_df["Amount"].apply(lambda x: f"₹{x:,.2f}")
        display_df["Expected_Loss"] = display_df["Expected_Loss"].apply(lambda x: f"₹{x:,.2f}")
        display_df["Fraud_Probability"] = display_df["Fraud_Probability"].apply(lambda x: f"{x:.2%}")

        # Rename for readability
        display_df = display_df.rename(columns={
            "Tx_ID":             "Transaction ID",
            "Amount":            "Amount (₹)",
            "Call_Mins":         "Call Duration (min)",
            "OTP_Fails":         "OTP Failures",
            "Fraud_Probability": "Fraud Probability",
            "Expected_Loss":     "Expected Loss (₹)",
        })

        st.markdown('<div class="section-header">High-Value Threat Prioritisation</div>', unsafe_allow_html=True)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=480,
        )

        # ── Download ─────────────────────────────────────────────────────────
        csv_bytes = queue_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Full Queue as CSV",
            data=csv_bytes,
            file_name="priority_queue_export.csv",
            mime="text/csv",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — AUDITOR REPORT (SHAP XAI)
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        """
        <div class="glass-card">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
                <span style="font-size:2rem;">🔬</span>
                <div>
                    <div style="font-size:1.25rem; font-weight:700; color:#e2e8f0;">Auditor Report — SHAP XAI Dashboard</div>
                    <div style="font-size:0.85rem; color:#64748b;">Varshith's SHAP engine: hard mathematics backing every plain-English reason</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    xai_col1, xai_col2 = st.columns([1, 1])
    with xai_col1:
        xai_txn_id = st.text_input("Transaction ID for Audit", value="TXN-DEMO-001", key="xai_txn")
    with xai_col2:
        xai_n_bg = st.slider("Background samples (SHAP)", 50, 500, 200, 25, key="xai_bg")

    xai_run = st.button("🔬 Generate SHAP Audit Dashboard", key="xai_run")

    @st.cache_data(ttl=120, show_spinner=False)
    def _run_shap_audit(txn_id: str, n_bg: int, model_path: str, scaler_path: str, output_path: str):
        try:
            import numpy as np
            from explainer import FraudExplainer, generate_digital_arrest_sample, generate_background_data

            exp = FraudExplainer(model_path=model_path, scaler_path=scaler_path)
            sample_df = generate_digital_arrest_sample()
            bg_df     = generate_background_data(n_samples=n_bg)

            exp.generate_audit_dashboard(txn_id, sample_df, bg_df, output_path=output_path)
            return True, output_path
        except Exception as exc:
            return False, str(exc)

    shap_output_path = os.path.join(ROOT, "reports", "figures", "advanced_shap_audit.png")
    model_path_shap  = os.path.join(ROOT, "models", "random_forest.pkl")
    scaler_path_shap = os.path.join(ROOT, "models", "scaler.pkl")

    if xai_run:
        with st.spinner("Running SHAP analysis — this may take 15–30 seconds…"):
            success, msg = _run_shap_audit(
                xai_txn_id, xai_n_bg,
                model_path_shap, scaler_path_shap, shap_output_path
            )

        if success:
            st.success("SHAP Audit Dashboard generated successfully!")
        else:
            st.error(f"SHAP generation failed: {msg}")
            st.info(
                "Make sure `models/random_forest.pkl` exists "
                "(run `python src/train.py` with the random_forest flag)."
            )

    # Display existing or newly generated plot
    if os.path.exists(shap_output_path):
        st.markdown('<div class="section-header">XAI Deep-Dive: Shadow Transfer Risk Profiling</div>', unsafe_allow_html=True)

        with open(shap_output_path, "rb") as f:
            img_bytes = f.read()

        st.image(img_bytes, caption="SHAP Audit Dashboard — Global + Local Explanations", use_container_width=True)

        st.markdown(
            """
            <div class="glass-card" style="margin-top:16px;">
                <div style="font-size:0.85rem; color:#94a3b8; line-height:1.7;">
                    <strong style="color:#00ff88;">Top-Left: Global Feature Impact (SHAP Summary)</strong> —
                    Shows which features most strongly drive fraud predictions across the entire dataset.<br>
                    <strong style="color:#00ff88;">Top-Right: Decision Convergence Plot</strong> —
                    Traces how each feature pushes the model's prediction from the base rate to the final score.<br>
                    <strong style="color:#00ff88;">Bottom-Left: Model Feature Importance</strong> —
                    The Random Forest's internal feature importance scores, confirming the SHAP rankings.<br>
                    <strong style="color:#00ff88;">Bottom-Right: Local Explanation Heatmap</strong> —
                    Clusters similar fraud patterns to reveal systemic attack vectors.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.download_button(
            label="⬇️ Download SHAP Audit Report (PNG)",
            data=img_bytes,
            file_name=f"shap_audit_{xai_txn_id}.png",
            mime="image/png",
        )
    else:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center; padding:60px 30px;">
                <div style="font-size:3rem; margin-bottom:12px;">🔬</div>
                <div style="color:#475569; font-size:1rem;">
                    Enter a Transaction ID and click<br>
                    <strong style="color:#00ff88;">Generate SHAP Audit Dashboard</strong>
                    to run Varshith's explainability engine.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CUSTOMER COMPLAINT PORTAL
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        """
        <div class="glass-card">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
                <span style="font-size:2rem;">📝</span>
                <div>
                    <div style="font-size:1.25rem; font-weight:700; color:#e2e8f0;">Customer Complaint Portal</div>
                    <div style="font-size:0.85rem; color:#64748b;">Report missed fraud or false positives to the Fraud Investigation Team</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fb_col_left, fb_col_right = st.columns([1, 1], gap="large")

    with fb_col_left:
        st.markdown('<div class="section-header">Submit a Report</div>', unsafe_allow_html=True)

        with st.form("feedback_form", clear_on_submit=True):
            fb_txn_id = st.text_input(
                "Transaction ID *",
                placeholder="e.g. TXN-A1B2C3D4",
                help="The Transaction ID you wish to report",
            )
            fb_reporter = st.text_input(
                "Your Name / Employee ID *",
                placeholder="e.g. Ekansh G / EMP-4421",
            )
            fb_email = st.text_input(
                "Contact Email (optional)",
                placeholder="name@bank.in",
            )
            fb_type = st.selectbox(
                "Complaint Type *",
                options=["MISSED_FRAUD", "FALSE_POSITIVE", "OTHER"],
                help=(
                    "MISSED_FRAUD: The AI missed a real fraud.\n"
                    "FALSE_POSITIVE: The AI flagged a legitimate transaction.\n"
                    "OTHER: General feedback."
                ),
            )
            fb_description = st.text_area(
                "Description *",
                placeholder=(
                    "Describe what happened. E.g.:\n"
                    "'Customer received a call from someone impersonating CBI. "
                    "₹2,50,000 was transferred but the AI did not flag it.'"
                ),
                height=160,
            )

            fb_submitted = st.form_submit_button("📨 Submit Report", use_container_width=True)

        if fb_submitted:
            missing = []
            if not fb_txn_id.strip():     missing.append("Transaction ID")
            if not fb_reporter.strip():   missing.append("Your Name / Employee ID")
            if not fb_description.strip(): missing.append("Description")

            if missing:
                st.error(f"Please fill in required fields: {', '.join(missing)}")
            else:
                payload = {
                    "transaction_id":  fb_txn_id.strip(),
                    "reported_by":     fb_reporter.strip(),
                    "complaint_type":  fb_type,
                    "description":     fb_description.strip(),
                    "contact_email":   fb_email.strip() if fb_email.strip() else None,
                }
                with st.spinner("Submitting report…"):
                    try:
                        resp = requests.post(f"{API_BASE}/feedback", json=payload, timeout=10)
                        resp.raise_for_status()
                        result = resp.json()
                        st.success(f"✅ Report submitted! **{result['feedback_id']}**")
                        st.info(result["message"])
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot reach the backend API. Please start it first.")
                    except Exception as exc:
                        st.error(f"Submission failed: {exc}")

    with fb_col_right:
        st.markdown('<div class="section-header">How to Report</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="glass-card">
                <div style="font-size:0.9rem; color:#94a3b8; line-height:1.9;">
                    <div style="margin-bottom:12px;">
                        <span style="color:#00ff88; font-weight:700;">1. Missed Fraud</span><br>
                        Use this when the AI system did <em>not</em> flag a transaction
                        that turned out to be fraudulent. Include the victim's account
                        details and the attack pattern observed.
                    </div>
                    <div style="margin-bottom:12px;">
                        <span style="color:#00ff88; font-weight:700;">2. False Positive</span><br>
                        Use this when a legitimate transaction was incorrectly flagged.
                        Include context such as prior customer relationship and
                        any verification steps already completed.
                    </div>
                    <div style="margin-bottom:12px;">
                        <span style="color:#f59e0b; font-weight:700;">⏱ SLA</span><br>
                        All reports are reviewed by the Fraud Investigation Team
                        within <strong style="color:#e2e8f0;">24 hours</strong>.
                        Critical cases are escalated to RBI within 4 hours.
                    </div>
                    <div>
                        <span style="color:#f59e0b; font-weight:700;">📞 Emergency Hotline</span><br>
                        For live fraud in progress, call the bank's Fraud Desk:<br>
                        <strong style="color:#e2e8f0;">1800-XXX-XXXX</strong> (24×7, Toll-free)
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Recent feedback log (if available)
        feedback_log = os.path.join(ROOT, "reports", "feedback_log.jsonl")
        if os.path.exists(feedback_log):
            st.markdown('<div class="section-header" style="margin-top:16px;">Recent Submissions</div>', unsafe_allow_html=True)
            try:
                import json as _json
                with open(feedback_log, "r", encoding="utf-8") as f:
                    entries = [_json.loads(line) for line in f if line.strip()]
                if entries:
                    recent = pd.DataFrame(entries[-10:][::-1])[
                        ["feedback_id", "transaction_id", "reported_by", "complaint_type", "submitted_at"]
                    ]
                    recent["submitted_at"] = pd.to_datetime(recent["submitted_at"]).dt.strftime("%d %b %H:%M")
                    st.dataframe(recent, use_container_width=True, hide_index=True)
            except Exception:
                pass
