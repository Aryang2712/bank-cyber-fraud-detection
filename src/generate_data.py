import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=50000):
    print(f"Generating {num_samples} realistic banking transactions...")
    np.random.seed(42)

    # Base Features
    data = {
        'Transaction_ID': np.arange(1, num_samples + 1),
        'Transaction_Amount': np.random.exponential(scale=5000, size=num_samples) + 100, # Ranges from 100 to large amounts
        'Transaction_Type': np.random.choice(['UPI', 'IMPS', 'NEFT', 'RTGS'], num_samples, p=[0.6, 0.2, 0.1, 0.1]),
        'Time_of_Day': np.random.randint(0, 24, num_samples),
        'New_Device_Used': np.random.choice([0, 1], num_samples, p=[0.9, 0.1]),
        'OTP_Failed_Attempts': np.random.choice([0, 1, 2, 3, 4], num_samples, p=[0.85, 0.1, 0.03, 0.01, 0.01]),
        'Active_Call_Duration_Min': np.random.choice([0, 5, 15, 60, 120, 240], num_samples, p=[0.8, 0.1, 0.05, 0.02, 0.02, 0.01]),
        'Payee_Account_Age_Days': np.random.randint(1, 3650, num_samples),
        'Is_Fraud': np.zeros(num_samples, dtype=int)
    }

    df = pd.DataFrame(data)

    # 🚨 Injecting Specific Cyber Fraud Patterns 🚨

    # 1. "Digital Arrest" Pattern: Long active call duration + Large RTGS/IMPS + Transfer to New Account
    digital_arrest_mask = (df['Active_Call_Duration_Min'] >= 60) & (df['Transaction_Amount'] > 50000) & (df['Payee_Account_Age_Days'] < 30)
    df.loc[digital_arrest_mask, 'Is_Fraud'] = 1

    # 2. "OTP Theft" Pattern: Multiple failed OTPs + New Device + Fast UPI/IMPS transfer
    otp_theft_mask = (df['OTP_Failed_Attempts'] >= 2) & (df['New_Device_Used'] == 1) & (df['Transaction_Type'].isin(['UPI', 'IMPS']))
    df.loc[otp_theft_mask, 'Is_Fraud'] = 1

    # 3. "Fake QR Code" Pattern: Small/Medium UPI scan + Brand new payee account + Unusual time (Late night)
    fake_qr_mask = (df['Transaction_Type'] == 'UPI') & (df['Payee_Account_Age_Days'] <= 2) & ((df['Time_of_Day'] > 22) | (df['Time_of_Day'] < 4))
    df.loc[fake_qr_mask, 'Is_Fraud'] = 1

    # Add some random baseline fraud (0.5% of remaining)
    normal_mask = df['Is_Fraud'] == 0
    random_fraud_indices = df[normal_mask].sample(frac=0.005).index
    df.loc[random_fraud_indices, 'Is_Fraud'] = 1

    print(f"Total Transactions: {len(df)}")
    print(f"Total Fraud Cases Simulated: {df['Is_Fraud'].sum()} ({df['Is_Fraud'].mean():.2%})")

    # Save to CSV
    os.makedirs("data/raw", exist_ok=True)
    filepath = "data/raw/synthetic_bank_fraud.csv"
    df.to_csv(filepath, index=False)
    print(f"✅ Dataset successfully saved to {filepath}")

if __name__ == "__main__":
    generate_synthetic_data()