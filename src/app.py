import streamlit as st
import requests
import pandas as pd
import os
from PIL import Image
from priority_queue import generate_priority_queue

# --- Configuration ---
st.set_page_config(page_title="Bank Fraud Threat Center", page_icon="🛡️", layout="wide")
API_URL = "http://localhost:8000/api/v1"

# --- Header ---
st.title("🛡️ Enterprise Cyber Fraud & Honeypot Command Center")
st.markdown("Monitoring real-time banking telemetry for **Digital Arrests, OTP Thefts, and UPI Scams**.")
st.divider()

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🚨 Live Transaction Monitor", 
    "📊 Live Threat Board (Priority Queue)", 
    "🧠 Auditor Report (SHAP XAI)", 
    "👥 Client Feedback Portal"
])

# ==========================================
# TAB 1: REAL-TIME MONITOR & HONEYPOT
# ==========================================
with tab1:
    st.subheader("Real-Time Transaction API Gateway")
    
    # Preset Demo Buttons
    st.markdown("**Quick Demo Scenarios:**")
    colA, colB, colC = st.columns(3)
    
    if "preset" not in st.session_state:
        st.session_state.preset = "normal"

    if colA.button("✅ Normal UPI Transfer"):
        st.session_state.preset = "normal"
    if colB.button("🚔 Simulate Digital Arrest"):
        st.session_state.preset = "digital_arrest"
    if colC.button("📱 Simulate OTP UPI Scam"):
        st.session_state.preset = "otp_scam"

    # Set default values based on preset
    defaults = {
        "normal": {"amt": 1500.0, "call": 0.0, "otp": 0, "device": False, "age": 365, "time": 14, "type": "UPI"},
        "digital_arrest": {"amt": 490000.0, "call": 145.0, "otp": 0, "device": False, "age": 2, "time": 11, "type": "IMPS"},
        "otp_scam": {"amt": 25000.0, "call": 0.0, "otp": 4, "device": True, "age": 5, "time": 3, "type": "UPI"}
    }
    d = defaults[st.session_state.preset]

    with st.form("transaction_form"):
        st.write("Transaction Telemetry:")
        c1, c2, c3 = st.columns(3)
        tx_id = c1.text_input("Transaction ID", "TXN-998877")
        amount = c2.number_input("Amount (₹)", value=d["amt"])
        tx_type = c3.selectbox("Transfer Type", ["UPI", "IMPS", "NEFT", "RTGS"], index=["UPI", "IMPS", "NEFT", "RTGS"].index(d["type"]))
        
        c4, c5, c6 = st.columns(3)
        call_min = c4.number_input("Active Call Duration (Mins)", value=d["call"])
        otp_fails = c5.number_input("Failed OTP Attempts", value=d["otp"])
        time_day = c6.number_input("Time of Day (0-23)", value=d["time"])
        
        c7, c8 = st.columns(2)
        payee_age = c7.number_input("Payee Account Age (Days)", value=d["age"])
        new_device = c8.checkbox("Unrecognized Device Used", value=d["device"])
        
        submitted = st.form_submit_button("Verify Transaction via AI")

    if submitted:
        payload = {
            "transaction_id": tx_id,
            "amount": amount,
            "call_duration_min": call_min,
            "otp_fails": otp_fails,
            "new_device": new_device,
            "beneficiary_age_days": payee_age,
            "time_of_day": time_day,
            "transaction_type": tx_type
        }
        
        try:
            response = requests.post(f"{API_URL}/transaction/verify", json=payload)
            if response.status_code == 200:
                data = response.json()
                
                if data["honeypot_triggered"]:
                    st.error(f"### 🚨 SHADOW TRANSFER TRIGGERED! (Risk: {data['fraud_probability']:.1%})")
                    st.warning("Hacker receives fake success receipt. Real funds frozen. Cyber SOC alerted.")
                    
                    st.markdown("#### 🧠 AI Reasoning (Plain English):")
                    for reason in data["reasons"]:
                        st.markdown(f"- 🚩 {reason}")
                elif data["is_flagged"]:
                    st.warning(f"⚠️ Transaction Flagged for Manual Review (Risk: {data['fraud_probability']:.1%})")
                else:
                    st.success(f"✅ Transaction Approved (Risk: {data['fraud_probability']:.1%})")
            else:
                st.error("API Error. Ensure FastAPI backend is running.")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")

# ==========================================
# TAB 2: LIVE THREAT BOARD
# ==========================================
with tab2:
    st.subheader("Expected Financial Loss - Triage Queue")
    st.markdown("Prioritizing human investigation by maximum potential Rupee loss.")
    
    if st.button("Refresh Threat Queue"):
        with st.spinner("Calculating expected losses..."):
            queue = generate_priority_queue()
            if queue is not None:
                display_queue = queue.head(15).copy()
                display_queue.insert(0, 'Priority', range(1, len(display_queue) + 1))
                display_queue['Amount'] = display_queue['Amount'].apply(lambda x: f"₹{x:,.2f}")
                display_queue['Expected_Loss'] = display_queue['Expected_Loss'].apply(lambda x: f"₹{x:,.2f}")
                display_queue['Fraud_Probability'] = display_queue['Fraud_Probability'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(display_queue, use_container_width=True, hide_index=True)
            else:
                st.error("Could not load queue. Run data pipeline first.")

# ==========================================
# TAB 3: AUDITOR REPORT (SHAP)
# ==========================================
with tab3:
    st.subheader("Mathematical Evidence (XAI)")
    st.markdown("This dashboard proves mathematically *why* the Shadow Transfer was triggered, ensuring regulatory compliance.")
    
    shap_img_path = "reports/figures/advanced_shap_audit.png"
    if os.path.exists(shap_img_path):
        img = Image.open(shap_img_path)
        st.image(img, caption="SHAP Deep-Dive Analysis", use_container_width=True)
    else:
        st.info("No audit report found. Run `src/explainer.py` to generate the latest mathematical proof.")

# ==========================================
# TAB 4: CLIENT FEEDBACK PORTAL
# ==========================================
with tab4:
    st.subheader("Victim Reporting & Active Learning")
    st.markdown("Human-in-the-loop feedback. Reported missed frauds are logged for future AI retraining.")
    
    with st.form("feedback_form"):
        f_tx_id = st.text_input("Transaction ID to Report")
        f_comment = st.text_area("Details of Scam (e.g., 'They pretended to be CBI...')")
        f_submitted = st.form_submit_button("Submit Fraud Report")
        
        if f_submitted:
            f_payload = {
                "transaction_id": f_tx_id,
                "is_confirmed_fraud": True,
                "user_comment": f_comment
            }
            try:
                res = requests.post(f"{API_URL}/feedback/report", json=f_payload)
                if res.status_code == 200:
                    st.success("✅ Report logged into `user_complaints_log.csv`. The AI will learn from this in the next cycle.")
                else:
                    st.error("API Error.")
            except:
                st.error("Failed to connect to backend.")