import pandas as pd
import joblib
import os

def generate_priority_queue(data_path="data/raw/synthetic_bank_fraud.csv", 
                            model_path="models/new_fraud_model.pkl",
                            threshold=0.5):
    """
    Calculates expected financial loss for modern cyber fraud threats.
    """
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print("Error: Missing data or model.")
        return None

   # 1. Load model, scaler, and raw data
    model = joblib.load(model_path)
    scaler = joblib.load("models/scaler.pkl") # <-- ADD THIS
    model_columns = joblib.load("models/model_columns.pkl")
    df = pd.read_csv(data_path)
    
    # 2. Process data for inference
    original_amounts = df['Transaction_Amount'].values
    df_processed = pd.get_dummies(df, columns=['Transaction_Type'])
    
    for col in model_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0
            
    X_inference = df_processed[model_columns].copy() # Use .copy() to avoid warnings
    
    # <-- ADD THESE 2 LINES TO SCALE THE DATA BEFORE INFERENCE -->
    num_cols = ['Transaction_Amount', 'Time_of_Day', 'Active_Call_Duration_Min', 'Payee_Account_Age_Days']
    X_inference[num_cols] = scaler.transform(X_inference[num_cols])
    
    # 3. Extract exact probabilities
    probabilities = model.predict_proba(X_inference)[:, 1]
    
    # 4. Apply Novelty 1: Expected Financial Loss
    results = pd.DataFrame({
        'Tx_ID': df['Transaction_ID'],
        'Amount': original_amounts,
        'Call_Mins': df['Active_Call_Duration_Min'], # Added to show Digital Arrest context!
        'OTP_Fails': df['OTP_Failed_Attempts'],      # Added to show OTP Theft context!
        'Fraud_Probability': probabilities,
        'Expected_Loss': probabilities * original_amounts,
    })
    
    # 5. Filter and Sort the Queue
    flagged_tx = results[results['Fraud_Probability'] >= threshold].copy()
    priority_queue = flagged_tx.sort_values(by='Expected_Loss', ascending=False).reset_index(drop=True)
    
    return priority_queue

if __name__ == "__main__":
    queue = generate_priority_queue()
    
    if queue is not None:
        print("\n🚨 HIGH-VALUE CYBER THREAT PRIORITIZATION QUEUE 🚨")
        formatted_queue = queue.head(10).copy()
        
        # Insert Priority Number as the very first column
        formatted_queue.insert(0, 'Priority', range(1, len(formatted_queue) + 1))
        
        # Apply clean formatting with Rupees (₹)
        formatted_queue['Amount'] = formatted_queue['Amount'].apply(lambda x: f"₹{x:,.2f}")
        formatted_queue['Expected_Loss'] = formatted_queue['Expected_Loss'].apply(lambda x: f"₹{x:,.2f}")
        formatted_queue['Fraud_Probability'] = formatted_queue['Fraud_Probability'].apply(lambda x: f"{x:.2%}")
        
        print(formatted_queue.to_string(index=False))