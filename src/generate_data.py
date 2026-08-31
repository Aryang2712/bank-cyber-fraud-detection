# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=50000, random_seed=42):
    np.random.seed(random_seed)

    # 1. Base transaction attributes
    amounts         = np.random.exponential(scale=3000, size=num_samples) + 100
    tx_types        = np.random.choice(['UPI', 'IMPS', 'NEFT', 'RTGS'], size=num_samples, p=[0.6, 0.2, 0.1, 0.1])
    call_durations  = np.zeros(num_samples)
    otp_failures    = np.zeros(num_samples, dtype=int)
    is_new_device   = np.zeros(num_samples, dtype=int)
    is_high_risk_ip = np.zeros(num_samples, dtype=int)
    # New payee added X minutes ago (large = old payee = safe)
    new_payee_mins  = np.random.randint(30, 3000, size=num_samples).astype(float)
    # Account age in days
    account_age     = np.random.randint(90, 3000, size=num_samples)

    labels = np.zeros(num_samples, dtype=int)

    # 2. Inject clear cyber attack patterns
    for i in range(num_samples):
        rand_val = np.random.rand()

        # Pattern A: Digital Arrest Scam (long call + high amount + brand-new payee)
        if rand_val < 0.015:
            labels[i]          = 1
            call_durations[i]  = np.random.randint(45, 240)
            amounts[i]         = np.random.uniform(50000, 500000)
            is_new_device[i]   = 1
            new_payee_mins[i]  = np.random.randint(1, 10)

        # Pattern B: OTP Stealing / SIM Swap (multiple OTP fails + new device)
        elif rand_val < 0.025:
            labels[i]          = 1
            otp_failures[i]    = np.random.randint(2, 5)
            is_new_device[i]   = 1
            amounts[i]         = np.random.uniform(10000, 100000)

        # Pattern C: Blacklisted-IP / mule account
        elif rand_val < 0.030:
            labels[i]          = 1
            is_high_risk_ip[i] = 1
            account_age[i]     = np.random.randint(1, 30)
            amounts[i]         = np.random.uniform(20000, 300000)

    df = pd.DataFrame({
        'Transaction_ID':           np.arange(10000, 10000 + num_samples),
        'Transaction_Amount':       np.round(amounts, 2),
        'Transaction_Type':         tx_types,
        'Active_Call_Duration_Min': call_durations,
        'OTP_Failed_Attempts':      otp_failures,
        'New_Payee_Added_Mins_Ago': new_payee_mins,
        'Is_New_Device':            is_new_device,
        'Is_High_Risk_IP':          is_high_risk_ip,
        'Account_Age_Days':         account_age,
        'Is_Fraud':                 labels,
    })

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/synthetic_bank_fraud.csv", index=False)
    fraud_count = labels.sum()
    print(f"Generated {num_samples} records. Total Fraud Cases: {fraud_count}")

if __name__ == "__main__":
    generate_synthetic_data()