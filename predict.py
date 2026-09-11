"""
Prediction Script
Make predictions on new credit card transactions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle

class FraudDetector:
    """Class to handle fraud detection predictions"""
    
    def __init__(self, data_path='data/creditcard.csv'):
        """Initialize the fraud detector with trained model"""
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
    def train_model(self, n_neighbors=5):
        """Train the model"""
        print("Training model...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Prepare features
        X = df.drop('Class', axis=1)
        y = df['Class']
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean', n_jobs=-1)
        self.model.fit(X_train_scaled, y_train)
        
        print(f"Model trained successfully with k={n_neighbors}")
        
    def predict_single(self, transaction):
        """Predict if a single transaction is fraud"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Prepare transaction data
        transaction_df = pd.DataFrame([transaction])
        transaction_scaled = self.scaler.transform(transaction_df)
        
        # Predict
        prediction = self.model.predict(transaction_scaled)[0]
        probability = self.model.predict_proba(transaction_scaled)[0]
        
        return {
            'prediction': 'FRAUD' if prediction == 1 else 'NORMAL',
            'fraud_probability': probability[1],
            'normal_probability': probability[0]
        }
    
    def predict_batch(self, transactions_df):
        """Predict on multiple transactions"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Scale transactions
        transactions_scaled = self.scaler.transform(transactions_df)
        
        # Predict
        predictions = self.model.predict(transactions_scaled)
        probabilities = self.model.predict_proba(transactions_scaled)
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({
                'prediction': 'FRAUD' if pred == 1 else 'NORMAL',
                'fraud_probability': prob[1],
                'normal_probability': prob[0]
            })
        
        return results
    
    def save_model(self, model_path='models/fraud_detector.pkl', scaler_path='models/scaler.pkl'):
        """Save trained model and scaler"""
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path='models/fraud_detector.pkl', scaler_path='models/scaler.pkl'):
        """Load trained model and scaler"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")

def example_predictions():
    """Run example predictions"""
    print("\n" + "="*60)
    print("CREDIT CARD FRAUD DETECTION - PREDICTION EXAMPLES")
    print("="*60)
    
    # Initialize detector
    detector = FraudDetector()
    detector.train_model(n_neighbors=5)
    
    # Load dataset for examples
    df = pd.read_csv('data/creditcard.csv')
    
    # Get one normal and one fraud transaction
    normal_tx = df[df['Class'] == 0].iloc[0].drop('Class')
    fraud_tx = df[df['Class'] == 1].iloc[0].drop('Class')
    
    print("\n" + "-"*60)
    print("Example 1: Normal Transaction")
    print("-"*60)
    result = detector.predict_single(normal_tx)
    print(f"Prediction: {result['prediction']}")
    print(f"Fraud Probability: {result['fraud_probability']:.4f}")
    print(f"Normal Probability: {result['normal_probability']:.4f}")
    
    print("\n" + "-"*60)
    print("Example 2: Fraudulent Transaction")
    print("-"*60)
    result = detector.predict_single(fraud_tx)
    print(f"Prediction: {result['prediction']}")
    print(f"Fraud Probability: {result['fraud_probability']:.4f}")
    print(f"Normal Probability: {result['normal_probability']:.4f}")
    
    # Batch prediction
    print("\n" + "-"*60)
    print("Batch Prediction (First 5 transactions)")
    print("-"*60)
    test_batch = df.drop('Class', axis=1).iloc[:5]
    batch_results = detector.predict_batch(test_batch)
    
    for i, result in enumerate(batch_results):
        print(f"Transaction {i+1}: {result['prediction']} (Fraud: {result['fraud_probability']:.4f})")
    
    # Save model
    detector.save_model()
    
    print("\n" + "="*60)
    print("PREDICTIONS COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    example_predictions()
