# -*- coding: utf-8 -*-
import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data_prep import load_and_preprocess_data

def train_model():
    print("Starting the XGBoost AI Training Pipeline...")

    # 1. Load and preprocess data
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()

    print("\nTraining the XGBoost Machine Learning Model...")

    # 2. XGBoost — scale_pos_weight penalises missing a fraud case
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # 3. Evaluate
    print("\nEvaluating AI against unseen Test Data...")
    y_pred = model.predict(X_test)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))

    # 4. Persist model
    os.makedirs("models", exist_ok=True)
    model_path = "models/new_fraud_model.pkl"
    joblib.dump(model, model_path)

    print(f"\nXGBoost Model saved to {model_path}")

if __name__ == "__main__":
    train_model()