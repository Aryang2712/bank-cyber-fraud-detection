import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=50000, random_seed=42):
    np.random.seed(random_seed)
    
    # 1. Base transaction attributes
    amounts = np.random.exponential(scale=3000, size=num_samples) + 100
    times = np.random.randint(0, 24, size=num_samples)
    tx_types = np.random.choice(['UPI', 'IMPS', 'NEFT', 'RTGS'], size=num_samples, p=[0.6, 0.2, 0.1, 0.1])
    call_durations = np.zeros(num_samples)
    otp_failures = np.zeros(num_samples, dtype=int)
    new_devices = np.zeros(num_samples, dtype=int)
    payee_ages = np.random.randint(30, 3000, size=num_samples)
    
    labels = np.zeros(num_samples, dtype=int)
    
    # 2. Inject clear cyber attack patterns
    for i in range(num_samples):
        rand_val = np.random.rand()
        
        # Pattern A: Digital Arrest Scam (Long call + high amount + new payee)
        if rand_val < 0.015:
            labels[i] = 1
            call_durations[i] = np.random.randint(45, 240)
            amounts[i] = np.random.uniform(50000, 500000)
            new_devices[i] = 1
            payee_ages[i] = np.random.randint(1, 10)
            
        # Pattern B: OTP Stealing / SIM Swap (Multiple OTP fails + new device)
        elif rand_val < 0.025:
            labels[i] = 1
            otp_failures[i] = np.random.randint(2, 5)
            new_devices[i] = 1
            amounts[i] = np.random.uniform(10000, 100000)

    df = pd.DataFrame({
        'Transaction_ID': np.arange(10000, 10000 + num_samples),
        'Transaction_Amount': np.round(amounts, 2),
        'Time_of_Day': times,
        'Transaction_Type': tx_types,
        'Active_Call_Duration_Min': call_durations,
        'OTP_Failed_Attempts': otp_failures,
        'New_Device_Used': new_devices,
        'Payee_Account_Age_Days': payee_ages,
        'Is_Fraud': labels
    })
    
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/synthetic_bank_fraud.csv", index=False)
    print(f"Generated {num_samples} records. Total Fraud Cases: {labels.sum()}")

if __name__ == "__main__":
    generate_synthetic_data()