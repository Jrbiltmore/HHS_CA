# Folder: src/data
# File: model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(data_file):
    try:
        # Load the data
        logger.info("Loading data...")
        data = pd.read_csv(data_file)
        
        # Split features and target variable
        X = data.drop(columns=['target_column'])
        y = data['target_column']
        
        # Split the data into training and testing sets
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize the model
        logger.info("Initializing Random Forest Classifier model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train the model
        logger.info("Training the model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        logger.info("Making predictions on the test set...")
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy: {accuracy}")
        
        # Save the trained model
        logger.info("Saving the trained model...")
        joblib.dump(model, 'fraud_detection_model.pkl')
        
        logger.info("Model training completed successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")

if __name__ == "__main__":
    # Path to the data file
    data_file = "california_health_services.csv"
    
    # Train the model
    train_model(data_file)
