# Folder: src
# File: main.py

import sys
import pandas as pd
from data_preprocessing import clean_data
from feature_engineering import create_features
from model_training import train_model
from fraud_detection import detect_fraud

def main():
    try:
        # Load dataset
        data = pd.read_csv("data/california_health_services.csv")
        
        # Data preprocessing
        clean_data(data)
        
        # Feature engineering
        create_features(data)
        
        # Model training
        model = train_model(data)
        
        # Check if model is None
        if model is None:
            print("Error: Model training failed.")
            sys.exit(1)
        
        # Fraud detection
        fraud_results = detect_fraud(data, model)
        
        print("Fraud detection results:", fraud_results)
    
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        sys.exit(1)
    except Exception as e:
        print("An unexpected error occurred:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
