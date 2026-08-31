"""
FastAPI service for Bank Cyber Fraud Detection
Endpoints:
  POST /api/v1/transaction/verify  - Run fraud inference + honeypot logic
  POST /api/v1/feedback/report     - Append user feedback to CSV log

Model contract (confirmed from src/data_prep.py + models/model_columns.pkl):
  Features (in order):
    Transaction_Amount        [StandardScaler]
    Time_of_Day               [raw int 0-23]
    Active_Call_Duration_Min  [StandardScaler]
    OTP_Failed_Attempts       [raw int]
    New_Device_Used           [raw int 0/1]
    Payee_Account_Age_Days    [StandardScaler]
    Transaction_Type_IMPS     [one-hot]
    Transaction_Type_NEFT     [one-hot]
    Transaction_Type_RTGS     [one-hot]
    Transaction_Type_UPI      [one-hot]
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger("fraud_api")

# ---------------------------------------------------------------------------
# Paths (relative to project root - start uvicorn from project root)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "new_fraud_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "model_columns.pkl"
FEEDBACK_CSV = BASE_DIR / "data" / "raw" / "user_complaints_log.csv"

# ---------------------------------------------------------------------------
# Startup: load artefacts once
# ---------------------------------------------------------------------------
try:
    _model = joblib.load(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)
    _model_columns: list = joblib.load(COLUMNS_PATH)
    logger.info("Model, scaler and column list loaded successfully.")
    logger.info("Expected columns: %s", _model_columns)
except Exception as exc:
    logger.exception("Failed to load model artefacts - server cannot start.")
    raise RuntimeError(f"Model loading failed: {exc}") from exc

# Columns that were StandardScaler-transformed during training
_SCALED_COLS = [
    "Transaction_Amount",
    "Time_of_Day",
    "Active_Call_Duration_Min",
    "Payee_Account_Age_Days",
]

# Fraud probability threshold for honeypot
HONEYPOT_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Bank Cyber Fraud Detection API",
    description=(
        "Real-time fraud scoring API with Shadow Transfer Honeypot integration "
        "and plain-English XAI reasons for flagged transactions."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Honeypot stub
# ---------------------------------------------------------------------------
def trigger_honeypot(transaction_id: str) -> None:
    """
    Stub for the Shadow Transfer Honeypot.
    In production this would freeze funds, alert the SOC, and route
    the transfer to a monitored mule-account mirror.
    """
    logger.warning(
        "[HONEYPOT TRIGGERED] transaction_id=%s - funds frozen, SOC alerted.",
        transaction_id,
    )
    # TODO: integrate with real honeypot orchestration layer


# ---------------------------------------------------------------------------
# Reason generator (NL-XAI)
# ---------------------------------------------------------------------------
def _build_reasons(payload) -> List[str]:
    """
    Generate plain-English explanations directly from telemetry values.
    No ML-interpretability library required - rules are anchored to the
    known fraud patterns injected during data generation.
    """
    reasons: List[str] = []

    # Digital Arrest pattern - long active call
    if payload.call_duration_min > 30:
        reasons.append(
            f"Victim was on an active phone call for {payload.call_duration_min:.0f} "
            "minutes, indicating a potential Digital Arrest / impersonation scam."
        )

    # OTP stealing / SIM-swap
    if payload.otp_fails > 0:
        reasons.append(
            f"Multiple failed OTP attempts detected ({payload.otp_fails}). "
            "This may indicate an OTP-theft or SIM-swap attack."
        )

    # Unrecognised device
    if payload.new_device:
        reasons.append(
            "Transaction initiated from an unrecognised device not previously "
            "associated with this account - possible device takeover."
        )

    # Freshly created mule account
    if payload.beneficiary_age_days < 7:
        reasons.append(
            f"Beneficiary account was created only {payload.beneficiary_age_days} "
            "day(s) ago. Newly created accounts are a strong mule-account pattern."
        )
    elif payload.beneficiary_age_days < 30:
        reasons.append(
            f"Beneficiary account is only {payload.beneficiary_age_days} day(s) old "
            "- potentially a recently opened mule account."
        )

    # High-value transaction
    if payload.amount > 50000:
        reasons.append(
            f"High-value transaction of Rs.{payload.amount:,.2f} - unusually large "
            "transfers are a common Digital Arrest indicator."
        )

    # Unusual transaction time (2 AM - 5 AM)
    if 2 <= payload.time_of_day <= 5:
        reasons.append(
            f"Transaction initiated at {payload.time_of_day:02d}:xx - odd-hour "
            "transactions increase fraud risk significantly."
        )

    # Rapid login-to-transfer (session hijack indicator) - optional field
    if payload.time_since_login_sec is not None and payload.time_since_login_sec < 30:
        reasons.append(
            f"Transaction was submitted only {payload.time_since_login_sec:.0f} "
            "second(s) after login - possible scripted/automated transfer."
        )

    return reasons


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------
class TransactionType(str, Enum):
    UPI = "UPI"
    IMPS = "IMPS"
    NEFT = "NEFT"
    RTGS = "RTGS"


class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    amount: float = Field(..., gt=0, description="Transaction amount in INR")
    call_duration_min: float = Field(
        0.0, ge=0, description="Duration of active call during transaction (minutes)"
    )
    otp_fails: int = Field(0, ge=0, description="Number of failed OTP attempts")
    new_device: bool = Field(False, description="True if an unrecognised device was used")
    beneficiary_age_days: int = Field(
        ..., ge=0, description="Age of beneficiary/payee account in days"
    )
    time_of_day: int = Field(
        ..., ge=0, le=23, description="Hour of transaction (0-23)"
    )
    transaction_type: TransactionType = Field(
        TransactionType.UPI, description="Payment rail (UPI, IMPS, NEFT, RTGS)"
    )
    time_since_login_sec: Optional[float] = Field(
        None, ge=0, description="Seconds elapsed since the session login (optional)"
    )


class TransactionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_flagged: bool
    honeypot_triggered: bool
    reasons: List[str]


class FeedbackRequest(BaseModel):
    transaction_id: str
    is_confirmed_fraud: bool
    user_comment: Optional[str] = ""


class FeedbackResponse(BaseModel):
    status: str
    transaction_id: str


# ---------------------------------------------------------------------------
# Helper: build feature vector for inference
# ---------------------------------------------------------------------------
def _build_feature_df(req: TransactionRequest) -> pd.DataFrame:
    """
    Construct a single-row DataFrame that exactly matches _model_columns,
    applying the same StandardScaler used during training.
    """
    # Build raw (unscaled) row
    raw = {
        "Transaction_Amount": req.amount,
        "Time_of_Day": float(req.time_of_day),
        "Active_Call_Duration_Min": req.call_duration_min,
        "OTP_Failed_Attempts": float(req.otp_fails),
        "New_Device_Used": float(int(req.new_device)),
        "Payee_Account_Age_Days": float(req.beneficiary_age_days),
        "Transaction_Type_IMPS": 0.0,
        "Transaction_Type_NEFT": 0.0,
        "Transaction_Type_RTGS": 0.0,
        "Transaction_Type_UPI": 0.0,
    }

    # One-hot encode transaction type
    one_hot_key = f"Transaction_Type_{req.transaction_type.value}"
    if one_hot_key in raw:
        raw[one_hot_key] = 1.0

    df = pd.DataFrame([raw])

    # Apply scaler only to the three columns it was fitted on
    df[_SCALED_COLS] = _scaler.transform(df[_SCALED_COLS])

    # Guarantee column order
    df = df[_model_columns]

    return df
# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post(
    "/api/v1/transaction/verify",
    response_model=TransactionResponse,
    summary="Verify a transaction for fraud",
    tags=["Fraud Detection"],
)
async def verify_transaction(req: TransactionRequest) -> TransactionResponse:
    """
    Run fraud inference on transaction telemetry.

    - Applies the training-time StandardScaler to numeric features.
    - If fraud_probability > 0.85 the Shadow Transfer Honeypot stub is triggered.
    - Returns plain-English reasons derived from the telemetry values.
    """
    try:
        feature_df = _build_feature_df(req)
        proba: float = float(_model.predict_proba(feature_df)[0][1])
    except Exception as exc:
        logger.exception("Model inference failed for transaction_id=%s", req.transaction_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference error: {exc}",
        ) from exc

    is_flagged = proba > 0.5
    honeypot_triggered = proba > HONEYPOT_THRESHOLD

    reasons = _build_reasons(req)

    if honeypot_triggered:
        trigger_honeypot(req.transaction_id)

    logger.info(
        "transaction_id=%s  fraud_prob=%.4f  flagged=%s  honeypot=%s",
        req.transaction_id,
        proba,
        is_flagged,
        honeypot_triggered,
    )

    return TransactionResponse(
        transaction_id=req.transaction_id,
        fraud_probability=round(proba, 6),
        is_flagged=is_flagged,
        honeypot_triggered=honeypot_triggered,
        reasons=reasons,
    )


@app.post(
    "/api/v1/feedback/report",
    response_model=FeedbackResponse,
    summary="Report confirmed fraud or false-positive feedback",
    tags=["Feedback Loop"],
)
async def report_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """
    Append user feedback to data/raw/user_complaints_log.csv.

    Creates the file with a header row if it does not yet exist.
    This feeds the Client Feedback Loop / active-learning pipeline.
    """
    try:
        FEEDBACK_CSV.parent.mkdir(parents=True, exist_ok=True)
        file_exists = FEEDBACK_CSV.exists()

        with FEEDBACK_CSV.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["timestamp_utc", "transaction_id", "is_confirmed_fraud", "user_comment"],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "transaction_id": req.transaction_id,
                    "is_confirmed_fraud": req.is_confirmed_fraud,
                    "user_comment": req.user_comment or "",
                }
            )
    except OSError as exc:
        logger.exception("Failed to write feedback for transaction_id=%s", req.transaction_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist feedback: {exc}",
        ) from exc

    logger.info(
        "Feedback recorded: transaction_id=%s  confirmed_fraud=%s",
        req.transaction_id,
        req.is_confirmed_fraud,
    )
    return FeedbackResponse(status="recorded", transaction_id=req.transaction_id)


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
