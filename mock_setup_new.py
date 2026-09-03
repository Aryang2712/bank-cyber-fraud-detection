import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("Setting up Mock ML Model for the New Synthetic Data...")

# 1. Load the new synthetic data
df = pd.read_csv("data/raw/synthetic_bank_fraud.csv")

# 2. Convert text columns (like UPI, IMPS) into numbers so the ML model can read them
df_processed = pd.get_dummies(df, columns=['Transaction_Type'])

# 3. Separate features from the target
X = df_processed.drop(['Is_Fraud', 'Transaction_ID'], axis=1)
y = df_processed['Is_Fraud']

# 4. Train a quick mock model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# 5. Save the model and the column names
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/new_fraud_model.pkl")
joblib.dump(list(X.columns), "models/model_columns.pkl")

print("✅ Mock Model trained on new cyber-fraud features!")