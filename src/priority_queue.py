import pandas as pd
import numpy as np
import joblib
import os

def generate_priority_queue(data_path="data/raw/creditcard.csv", 
                            model_path="models/fraud_model.pkl",
                            threshold=0.5):
    """
    Calculates expected financial loss and returns a sorted priority queue.
    """
    # 1. Dependency Checks (Ensures Ekansh and Suyash finished their modules)
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Suyash must run train.py first.")
        return None
    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}. Ekansh must add the CSV.")
        return None

    # 2. Load model, scalers, and raw data
    print("Loading predictive model and transaction records...")
    model = joblib.load(model_path)
    scaler_amount = joblib.load("models/scaler_amount.pkl")
    scaler_time = joblib.load("models/scaler_time.pkl")
    
    df = pd.read_csv(data_path)
    
    # Isolate original amounts for the business logic before scaling
    original_amounts = df['Amount'].values
    
    # Prepare features identically to Ekansh's preprocessing pipeline
    df['scaled_amount'] = scaler_amount.transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler_time.transform(df['Time'].values.reshape(-1, 1))
    
    X_inference = df.drop(['Class', 'Time', 'Amount'], axis=1)
    
    # 3. Extract exact probabilities (Suyash's output)
    print("Calculating statistical fraud probabilities...")
    probabilities = model.predict_proba(X_inference)[:, 1] 
    
    # 4. Apply Novelty 1: Expected Financial Loss
    print("Generating financial risk scores...")
    results = pd.DataFrame({
        'Transaction_ID': df.index,
        'Original_Amount': original_amounts,
        'Fraud_Probability': probabilities,
        'Expected_Loss': probabilities * original_amounts,
        'True_Class': df['Class'] # Retained for evaluation and testing
    })
    
    # 5. Filter and Sort the Queue
    # Keep transactions above the probability threshold, sorted by maximum financial exposure
    flagged_tx = results[results['Fraud_Probability'] >= threshold].copy()
    priority_queue = flagged_tx.sort_values(by='Expected_Loss', ascending=False).reset_index(drop=True)
    
    return priority_queue

if __name__ == "__main__":
    # Local execution test
    queue = generate_priority_queue()
    
    if queue is not None:
        print("\n🚨 HIGH-VALUE THREAT PRIORITIZATION QUEUE 🚨")
        # Format the output for clean terminal reading
        formatted_queue = queue[['Transaction_ID', 'Original_Amount', 'Fraud_Probability', 'Expected_Loss']].head(10)
        
        # Apply clean financial formatting
        formatted_queue['Original_Amount'] = formatted_queue['Original_Amount'].apply(lambda x: f"${x:,.2f}")
        formatted_queue['Expected_Loss'] = formatted_queue['Expected_Loss'].apply(lambda x: f"${x:,.2f}")
        formatted_queue['Fraud_Probability'] = formatted_queue['Fraud_Probability'].apply(lambda x: f"{x:.2%}")
        
        print(formatted_queue.to_string(index=False))