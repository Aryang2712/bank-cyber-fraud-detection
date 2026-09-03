import os
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pathlib import Path

app = FastAPI(title="Bank Fraud & Honeypot API")

# Absolute paths
BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "models" / "new_fraud_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")
model_columns = joblib.load(BASE_DIR / "models" / "model_columns.pkl")

@app.post("/api/v1/transaction/verify")
async def verify_transaction(request: Request):
    # Bypass strict Pydantic validation to safely catch whatever app.py sends
    payload = await request.json()
    
    # Smart extraction (handles multiple possible naming conventions)
    amount = float(payload.get("amount", payload.get("transaction_amount", payload.get("Transaction_Amount", 0.0))))
    call_dur = float(payload.get("call_duration_min", payload.get("active_call_duration_min", payload.get("Active_Call_Duration_Min", 0.0))))
    otp_fails = int(payload.get("otp_fails", payload.get("otp_failed_attempts", payload.get("OTP_Failed_Attempts", 0))))
    new_device = bool(payload.get("new_device", payload.get("is_new_device", payload.get("Is_New_Device", False))))
    age_days = int(payload.get("beneficiary_age_days", payload.get("payee_account_age_days", payload.get("Payee_Account_Age_Days", 0))))
    time_of_day = int(payload.get("time_of_day", payload.get("Time_of_Day", 0)))
    tx_type = str(payload.get("transaction_type", payload.get("Transaction_Type", "UPI")))
    
    input_dict = {
        "Transaction_Amount": [amount],
        "Active_Call_Duration_Min": [call_dur],
        "OTP_Failed_Attempts": [otp_fails],
        "Is_New_Device": [int(new_device)],
        "Payee_Account_Age_Days": [age_days],
        "Time_of_Day": [time_of_day],
    }
    
    for t in ["UPI", "IMPS", "NEFT", "RTGS"]:
        input_dict[f"Transaction_Type_{t}"] = [1 if tx_type == t else 0]
        
    df = pd.DataFrame(input_dict)
    
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[model_columns].copy()
    
    # Apply Scaler to numerical columns
    num_cols = ['Transaction_Amount', 'Time_of_Day', 'Active_Call_Duration_Min', 'Payee_Account_Age_Days']
    df[num_cols] = scaler.transform(df[num_cols])
    
    # Predict
    prob = float(model.predict_proba(df)[0, 1])
    is_honeypot = prob >= 0.85
    is_flagged = prob >= 0.50
    
    reasons = []
    if is_flagged:
        if call_dur > 30: reasons.append(f"Excessive active call duration ({call_dur} mins) - High Digital Arrest risk")
        if otp_fails > 0: reasons.append("Multiple failed OTPs detected")
        if amount > 50000: reasons.append("Unusually high transaction amount")
        if new_device: reasons.append("Unrecognized device utilized")
        
    return {
        "fraud_probability": prob,
        "honeypot_triggered": is_honeypot,
        "is_flagged": is_flagged,
        "reasons": reasons
    }