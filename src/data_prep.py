import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def load_and_preprocess_data(filepath="data/raw/creditcard.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please add the real creditcard.csv inside data/raw/")

    print("Loading raw dataset... (This might take a few seconds for 150MB)")
    df = pd.read_csv(filepath)
    
    print("Scaling Amount and Time features...")
    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()
    
    df['scaled_amount'] = scaler_amount.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler_time.fit_transform(df['Time'].values.reshape(-1, 1))
    
    # Save scalers so Aryan's priority queue and Suyash's model can use them later
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler_amount, "models/scaler_amount.pkl")
    joblib.dump(scaler_time, "models/scaler_time.pkl")
    
    # Drop original columns and separate target (Class)
    X = df.drop(['Class', 'Time', 'Amount'], axis=1)
    y = df['Class']
    
    print("Splitting dataset into train (80%) and test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Applying SMOTE to balance the training set... (Please wait)")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print("-" * 30)
    print("✅ DATA PREP COMPLETE")
    print(f"Original fraud cases in train set: {sum(y_train == 1)}")
    print(f"Balanced fraud cases after SMOTE: {sum(y_train_resampled == 1)}")
    print("-" * 30)
    
    return X_train_resampled, X_test, y_train_resampled, y_test, df

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, raw_df = load_and_preprocess_data()