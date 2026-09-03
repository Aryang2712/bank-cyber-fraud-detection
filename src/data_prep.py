# -*- coding: utf-8 -*-
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def load_and_preprocess_data(filepath="data/raw/synthetic_bank_fraud.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError("Dataset not found. Please run src/generate_data.py first.")

    print("Loading synthetic cyber fraud dataset...")
    df = pd.read_csv(filepath)

    # 1. Convert Transaction_Type (UPI/IMPS/NEFT/RTGS) into one-hot columns
    df_processed = pd.get_dummies(df, columns=['Transaction_Type'])

    # 2. Scale continuous numerical features
    print("Scaling numerical features...")
    scaler = StandardScaler()
    num_cols = [
        'Transaction_Amount',
        'Active_Call_Duration_Min',
        'New_Payee_Added_Mins_Ago',
        'Account_Age_Days',
    ]
    df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])

    # Save scaler for use in explainer / future inference
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    # 3. Separate features and target
    X = df_processed.drop(['Is_Fraud', 'Transaction_ID'], axis=1)
    y = df_processed['Is_Fraud']

    # Save exact column layout so backend and priority_queue can reproduce features
    joblib.dump(list(X.columns), "models/model_columns.pkl")

    # 4. Train / Test split
    print("Splitting dataset into train (80%) and test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. SMOTE — balance fraud minority class
    print("Applying SMOTE to balance the training set...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("-" * 30)
    print("DATA PREP COMPLETE")
    print(f"Original fraud cases in train set : {sum(y_train == 1)}")
    print(f"Balanced fraud cases after SMOTE  : {sum(y_train_resampled == 1)}")
    print("-" * 30)

    return X_train_resampled, X_test, y_train_resampled, y_test, df_processed

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, df = load_and_preprocess_data()