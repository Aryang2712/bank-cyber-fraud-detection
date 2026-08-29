import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data_prep import load_and_preprocess_data

def train_model():
    print("🚀 Starting the XGBoost AI Training Pipeline...")

    # 1. Load the data
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()

    print("\n🧠 Training the XGBoost Machine Learning Model...")
    
    # 2. Initialize XGBoost
    # scale_pos_weight heavily penalizes the model for missing a fraud case
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)

    # 3. Evaluate the AI
    print("\n📊 Evaluating AI against unseen Test Data...")
    y_pred = model.predict(X_test)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))

    # 4. Save the model
    os.makedirs("models", exist_ok=True)
    model_path = "models/new_fraud_model.pkl"
    joblib.dump(model, model_path)
    
    print(f"\n✅ XGBoost Model successfully trained and saved to {model_path}")

if __name__ == "__main__":
    train_model()