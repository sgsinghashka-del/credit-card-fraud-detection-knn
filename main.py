import os
from data_exploration import main as eda_main
from predict import FraudDetector


def ensure_dirs():
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

def run_all():
    ensure_dirs()
    # Run EDA (saves plots to results/)
    eda_main()

    # Train and save model
    detector = FraudDetector(data_path='data/creditcard.csv')
    detector.train_model(n_neighbors=5)
    detector.save_model(model_path='models/fraud_detector.pkl', scaler_path='models/scaler.pkl')


if __name__ == "__main__":
    run_all()
