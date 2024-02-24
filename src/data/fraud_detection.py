import pandas as pd
from urllib.parse import urlparse
import logging
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_file):
    """
    Load and return the dataset from the specified file path.
    """
    logger.info("Loading data...")
    return pd.read_csv(data_file)

def preprocess_data(data):
    """
    Perform preprocessing steps such as handling missing values, encoding categorical variables,
    and feature scaling.
    """
    logger.info("Preprocessing data...")
    # Identify numeric and categorical features
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = data.select_dtypes(include=['object']).columns
    
    # Define transformers for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def feature_selection(X_train, y_train):
    """
    Perform feature selection to reduce dimensionality and improve model interpretability.
    """
    logger.info("Performing feature selection...")
    sel = SelectFromModel(RandomForestClassifier(n_estimators=100))
    sel.fit(X_train, y_train)
    selected_features = X_train.columns[(sel.get_support())]
    logger.info(f"Selected features: {list(selected_features)}")
    return selected_features

def build_model(preprocessor, selected_features):
    """
    Build a machine learning pipeline that includes preprocessing, feature selection,
    and a classifier.
    """
    # Define the classifier
    classifier = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100))),
                                 ('classifier', GradientBoostingClassifier())])
    
    return classifier

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance on the test set using various metrics.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    logger.info("Model Evaluation Report:\n" + report)
    logger.info(f"ROC-AUC Score: {roc_auc}")

if __name__ == "__main__":
    try:
        data_file = "california_health_services.csv"
        data = load_data(data_file)
        
        # Assume 'NPI' column contains URLs to be analyzed
        hyperlinks = [urlparse(url).geturl() for url in data['NPI'] if urlparse(url).scheme]
        logger.info(f"Analyzed {len(hyperlinks)} hyperlinks.")
        
        # Preprocess data and select features
        preprocessor = preprocess_data(data)
        X = data.drop('fraud_label', axis=1)  # Assuming 'fraud_label' is the target column
        y = data['fraud_label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        selected_features = feature_selection(X_train, y_train)
        
        # Build and evaluate the model
        model = build_model(preprocessor, selected_features)
        model.fit(X_train, y_train)
        evaluate_model(model, X_test, y_test)
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
