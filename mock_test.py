import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

print("Setting up mock environment for Aryan's Priority Queue test...")

# 1. Ensure directories exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 2. Generate Dummy Data (100 rows)
np.random.seed(42)
n_samples = 100

# Create random amounts (some small, some very large to test the Expected Loss math)
mock_data = {
    'Time': np.random.uniform(0, 100000, n_samples),
    'Amount': np.random.uniform(5, 10000, n_samples), 
    'Class': np.random.choice([0, 1], p=[0.8, 0.2], size=n_samples) 
}

# Add the 28 PCA features (V1 to V28) standard in the dataset
for i in range(1, 29):
    mock_data[f'V{i}'] = np.random.normal(0, 1, n_samples)

df = pd.DataFrame(mock_data)
df.to_csv("data/raw/creditcard.csv", index=False)
print("✅ Created mock data/raw/creditcard.csv")

# 3. Create and Save Scalers
scaler_amount = StandardScaler()
df['scaled_amount'] = scaler_amount.fit_transform(df[['Amount']])
joblib.dump(scaler_amount, "models/scaler_amount.pkl")

scaler_time = StandardScaler()
df['scaled_time'] = scaler_time.fit_transform(df[['Time']])
joblib.dump(scaler_time, "models/scaler_time.pkl")
print("✅ Created mock scalers in models/")

# 4. Train a Dummy Model
X = df.drop(['Class', 'Time', 'Amount'], axis=1)
y = df['Class']

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)
joblib.dump(model, "models/fraud_model.pkl")
print("✅ Created mock models/fraud_model.pkl")

print("\nAll mock files generated successfully! You can now run your priority queue script.")