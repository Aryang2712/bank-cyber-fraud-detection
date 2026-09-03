import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

app = FastAPI(title="Bank Fraud & Honeypot API")

# Absolute paths to guarantee it finds the right files
BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "models" / "new_fraud_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")
model_columns = joblib.load(BASE_DIR / "models" / "model_columns.pkl")

class Transaction(BaseModel):
    transaction_id: str
    amount: float
    call_duration_min: float
    otp_fails: int
    new_device: bool
    beneficiary_age_days: int
    time_of_day: int
    transaction_type: str

@app.post("/api/v1/transaction/verify")
def verify_transaction(tx: Transaction):
    # 1. Convert incoming JSON to a DataFrame
    input_dict = {
        "Transaction_Amount": [tx.amount],
        "Active_Call_Duration_Min": [tx.call_duration_min],
        "OTP_Failed_Attempts": [tx.otp_fails],
        "Is_New_Device": [int(tx.new_device)],
        "Payee_Account_Age_Days": [tx.beneficiary_age_days],
        "Time_of_Day": [tx.time_of_day],
    }
    
    # Handle Transfer Type
    for t in ["UPI", "IMPS", "NEFT", "RTGS"]:
        input_dict[f"Transaction_Type_{t}"] = [1 if tx.transaction_type == t else 0]
        
    df = pd.DataFrame(input_dict)
    
    # 2. Align columns exactly as the model expects
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[model_columns].copy()
    
    # 3. THE CRITICAL FIX: Scale the numerical columns before predicting!
    num_cols = ['Transaction_Amount', 'Time_of_Day', 'Active_Call_Duration_Min', 'Payee_Account_Age_Days']
    df[num_cols] = scaler.transform(df[num_cols])
    
    # 4. Extract Probability
    prob = float(model.predict_proba(df)[0, 1])
    
    # 5. Honeypot Triage Logic
    is_honeypot = prob >= 0.85
    is_flagged = prob >= 0.50
    
    reasons = []
    if is_flagged:
        if tx.call_duration_min > 30: reasons.append("Excessive active call duration (High Digital Arrest risk)")
        if tx.otp_fails > 0: reasons.append("Multiple failed OTPs detected")
        if tx.amount > 50000: reasons.append("Unusually high transaction amount")
        if tx.new_device: reasons.append("Unrecognized device utilized")
        
    return {
        "fraud_probability": prob,
        "honeypot_triggered": is_honeypot,
        "is_flagged": is_flagged,
        "reasons": reasons
    }