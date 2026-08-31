"""
backend/app.py — FastAPI Backend for Bank Cyber Fraud Detection
Purpose : Exposes fraud detection REST API endpoints consumed by Ekansh's Streamlit UI.
          - POST /predict  → runs the AI model and returns fraud verdict + plain-English reasons
          - POST /feedback → accepts missed-fraud reports from the Customer Complaint Portal
"""

from __future__ import annotations

import os
import sys
import uuid
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Bootstrap: make src/ importable when running from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App Initialisation
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Bank Cyber Fraud Detection API",
    description=(
        "AI-powered fraud detection with Honeypot Triggers, "
        "Priority Queuing, and SHAP Explainability."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
MODEL_PATH   = os.path.join("models", "new_fraud_model.pkl")
COLUMNS_PATH = os.path.join("models", "model_columns.pkl")
SCALER_PATH  = os.path.join("models", "scaler.pkl")

_model         = None
_model_columns = None
_scaler        = None


def _load_model() -> None:
    global _model, _model_columns, _scaler
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model not found at {MODEL_PATH}. Predictions will use fallback heuristics.")
        return
    _model         = joblib.load(MODEL_PATH)
    _model_columns = joblib.load(COLUMNS_PATH)
    _scaler        = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    logger.info("AI Model and Scaler loaded successfully.")


@app.on_event("startup")
async def startup_event() -> None:
    _load_model()


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    transaction_id:        str   = Field(default_factory=lambda: f"TXN-{uuid.uuid4().hex[:8].upper()}")
    transaction_amount:    float = Field(..., ge=1,       description="Transaction amount in INR")
    transaction_type:      str   = Field("UPI",           description="UPI | IMPS | NEFT | RTGS")
    active_call_duration:  float = Field(0.0,  ge=0,     description="Active phone call duration in minutes")
    otp_failed_attempts:   int   = Field(0,    ge=0,     description="Number of OTP failures in this session")
    new_payee_added_mins:  float = Field(9999, ge=0,     description="Minutes since new payee was added")
    is_new_device:         int   = Field(0,               description="1 if transaction from an unrecognised device")
    is_high_risk_ip:       int   = Field(0,               description="1 if source IP is blacklisted")
    account_age_days:      int   = Field(365, ge=0,      description="Age of the account in days")


class PredictionResponse(BaseModel):
    transaction_id:    str
    is_fraud:          bool
    fraud_probability: float
    expected_loss:     float
    alert_level:       str
    honeypot_trigger:  str
    reasons:           List[str]
    timestamp:         str


class FeedbackRequest(BaseModel):
    transaction_id:  str
    reported_by:     str
    complaint_type:  str
    description:     str
    contact_email:   Optional[str] = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    status:      str
    message:     str


# ---------------------------------------------------------------------------
# Plain-English Reason Engine
# ---------------------------------------------------------------------------

def _build_reasons(req: TransactionRequest, prob: float):
    reasons: List[str] = []
    threat_signals: List[str] = []

    if req.active_call_duration >= 60:
        reasons.append(
            f"Victim was on an active call for {req.active_call_duration:.0f} minutes — "
            "a hallmark 'Digital Arrest' social-engineering pattern."
        )
        threat_signals.append("digital_arrest")
    elif req.active_call_duration >= 20:
        reasons.append(
            f"Prolonged call detected ({req.active_call_duration:.0f} min) — "
            "possible social engineering in progress."
        )

    if req.otp_failed_attempts >= 3:
        reasons.append(
            f"{req.otp_failed_attempts} OTP failures recorded — "
            "indicates a SIM-swap attack or credential brute-force attempt."
        )
        threat_signals.append("otp_theft")
    elif req.otp_failed_attempts >= 1:
        reasons.append(
            f"{req.otp_failed_attempts} OTP failure(s) — unusual authentication activity detected."
        )

    if req.new_payee_added_mins <= 10:
        reasons.append(
            f"New payee was added just {req.new_payee_added_mins:.0f} minute(s) before this transfer — "
            "classic Shadow Transfer setup."
        )
        threat_signals.append("shadow_transfer")
    elif req.new_payee_added_mins <= 30:
        reasons.append(
            f"Payee added {req.new_payee_added_mins:.0f} mins ago — unusually fast payment to new beneficiary."
        )

    if req.is_new_device:
        reasons.append(
            "Transaction initiated from an unrecognised device — "
            "potential account takeover via device cloning."
        )
        threat_signals.append("account_takeover")

    if req.is_high_risk_ip:
        reasons.append(
            "Source IP address is on the RBI/CERT-In blacklist — "
            "transaction originates from a known fraud node."
        )
        threat_signals.append("blacklisted_ip")

    if req.account_age_days <= 30:
        reasons.append(
            f"Account is only {req.account_age_days} days old — "
            "mule account pattern consistent with money laundering."
        )
        threat_signals.append("mule_account")

    if req.transaction_amount >= 200_000:
        reasons.append(
            f"Transaction value Rs.{req.transaction_amount:,.0f} exceeds high-risk threshold — "
            "triggers mandatory RBI Fraud Reporting."
        )
    elif req.transaction_amount >= 50_000:
        reasons.append(
            f"Transaction value Rs.{req.transaction_amount:,.0f} is elevated — warrants manual review."
        )

    reasons.append(
        f"AI Fraud Probability: {prob:.1%} — model confidence exceeds 50% decision threshold."
    )

    trigger_map = {
        "shadow_transfer":  "SHADOW TRANSFER HONEYPOT",
        "digital_arrest":   "DIGITAL ARREST HONEYPOT",
        "otp_theft":        "OTP THEFT HONEYPOT",
        "account_takeover": "ACCOUNT TAKEOVER HONEYPOT",
        "blacklisted_ip":   "BLACKLISTED IP HONEYPOT",
        "mule_account":     "MULE ACCOUNT HONEYPOT",
    }

    honeypot = "GENERIC FRAUD ALERT"
    for key in ["shadow_transfer", "digital_arrest", "otp_theft", "account_takeover", "blacklisted_ip", "mule_account"]:
        if key in threat_signals:
            honeypot = trigger_map[key]
            break

    if prob >= 0.85:
        alert = "CRITICAL"
    elif prob >= 0.65:
        alert = "HIGH"
    elif prob >= 0.50:
        alert = "MEDIUM"
    else:
        alert = "SAFE"

    return honeypot, reasons, alert


# ---------------------------------------------------------------------------
# Inference Helper
# ---------------------------------------------------------------------------

def _run_inference(req: TransactionRequest) -> float:
    if _model is None or _model_columns is None:
        score = 0.0
        if req.active_call_duration >= 60:  score += 0.40
        if req.otp_failed_attempts >= 3:    score += 0.25
        if req.new_payee_added_mins <= 10:  score += 0.20
        if req.is_new_device:               score += 0.10
        if req.is_high_risk_ip:             score += 0.10
        if req.account_age_days <= 30:      score += 0.10
        return min(score, 0.99)

    row: Dict[str, Any] = {
        "Transaction_Amount":       req.transaction_amount,
        "Active_Call_Duration_Min": req.active_call_duration,
        "OTP_Failed_Attempts":      req.otp_failed_attempts,
        "New_Payee_Added_Mins_Ago": req.new_payee_added_mins,
        "Is_New_Device":            req.is_new_device,
        "Is_High_Risk_IP":          req.is_high_risk_ip,
        "Account_Age_Days":         req.account_age_days,
    }

    for tt in ["UPI", "IMPS", "NEFT", "RTGS"]:
        row[f"Transaction_Type_{tt}"] = 1 if req.transaction_type == tt else 0

    df = pd.DataFrame([row])
    if _scaler is not None:
        num_cols = [
            "Transaction_Amount",
            "Active_Call_Duration_Min",
            "New_Payee_Added_Mins_Ago",
            "Account_Age_Days",
        ]
        available_num_cols = [c for c in num_cols if c in df.columns]
        df[available_num_cols] = _scaler.transform(df[available_num_cols])

    for col in _model_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[_model_columns]

    prob: float = float(_model.predict_proba(df)[0][1])
    return prob


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root() -> Dict[str, str]:
    return {"service": "Bank Cyber Fraud Detection API", "status": "Online", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health() -> Dict[str, str]:
    model_status = "loaded" if _model is not None else "fallback_heuristics"
    return {"status": "ok", "model": model_status}


@app.post("/predict", response_model=PredictionResponse, tags=["Fraud Detection"])
async def predict(req: TransactionRequest) -> PredictionResponse:
    try:
        prob = _run_inference(req)
    except Exception as exc:
        logger.error(f"Inference failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Model inference error: {exc}")

    is_fraud = prob >= 0.50
    honeypot, reasons, alert = _build_reasons(req, prob)

    if not is_fraud:
        honeypot = "TRANSACTION CLEARED"
        reasons  = [
            f"AI Fraud Probability: {prob:.1%} — below decision threshold.",
            "No active honeypot triggers detected.",
            "Transaction appears consistent with normal banking behaviour.",
        ]
        alert = "SAFE"

    return PredictionResponse(
        transaction_id    = req.transaction_id,
        is_fraud          = is_fraud,
        fraud_probability = round(prob, 4),
        expected_loss     = round(prob * req.transaction_amount, 2),
        alert_level       = alert,
        honeypot_trigger  = honeypot,
        reasons           = reasons,
        timestamp         = datetime.utcnow().isoformat() + "Z",
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Customer Portal"])
async def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    feedback_id = f"FB-{uuid.uuid4().hex[:10].upper()}"

    log_entry = {
        "feedback_id":    feedback_id,
        "transaction_id": req.transaction_id,
        "reported_by":    req.reported_by,
        "complaint_type": req.complaint_type,
        "description":    req.description,
        "contact_email":  req.contact_email,
        "submitted_at":   datetime.utcnow().isoformat() + "Z",
    }

    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", "feedback_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    logger.info(f"Feedback received: {feedback_id} for TXN {req.transaction_id}")

    return FeedbackResponse(
        feedback_id = feedback_id,
        status      = "received",
        message     = (
            f"Thank you, {req.reported_by}. Your report ({feedback_id}) has been logged "
            "and will be reviewed by the Fraud Investigation Team within 24 hours."
        ),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
