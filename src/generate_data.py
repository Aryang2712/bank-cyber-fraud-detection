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
        # Victim is coerced on a call and transfers from their OWN, already-recognized
        # device/phone -- so New_Device_Used should NOT be forced to 1 here. That was
        # the bug: it made New_Device_Used a perfect proxy for fraud and the model
        # never learned to weight call duration / amount / payee age at all.
        if rand_val < 0.015:
            labels[i] = 1
            call_durations[i] = np.random.randint(45, 240)
            amounts[i] = np.random.uniform(50000, 500000)
            new_devices[i] = 1 if np.random.rand() < 0.2 else 0  # occasionally true, not always
            payee_ages[i] = np.random.randint(1, 10)
            
        # Pattern B: OTP Stealing / SIM Swap (Multiple OTP fails + new device)
        # This pattern genuinely does usually involve a new device (attacker's SIM-swapped
        # phone), so New_Device_Used stays a strong signal here -- but not the ONLY one.
        elif rand_val < 0.025:
            labels[i] = 1
            otp_failures[i] = np.random.randint(2, 5)
            new_devices[i] = 1 if np.random.rand() < 0.85 else 0
            amounts[i] = np.random.uniform(10000, 100000)
    
    # 3. Add noise: a small fraction of NORMAL transactions also use a new device
    # (e.g. someone genuinely logging in from a new phone) so the model can't just
    # memorize "New_Device_Used == fraud" and is forced to weigh it alongside the
    # other features instead.
    normal_idx = np.where(labels == 0)[0]
    noisy_new_device = np.random.choice(
        normal_idx, size=int(len(normal_idx) * 0.03), replace=False
    )
    new_devices[noisy_new_device] = 1

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