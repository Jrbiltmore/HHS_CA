import os

# Define the directory structure
folders = [
    "src",
    "src/data",
    "src/models",
    "src/utils",
    "src/tests",
    "lib",
    "lib/screens",
    "lib/widgets",
    "lib/services",
    "lib/models",
    "lib/utils",
    "lib/assets"
]

files = [
    "src/main.py",
    "src/data/data_preprocessing.py",
    "src/data/feature_engineering.py",
    "src/data/model_training.py",
    "src/data/fraud_detection.py",
    "src/utils/helper_functions.py",
    "src/tests/test_data_preprocessing.py",
    "src/tests/test_feature_engineering.py",
    "src/tests/test_model_training.py",
    "src/tests/test_fraud_detection.py",
    "src/california_health_services.csv",
    "src/fraud_detection_model.pkl",
    "src/requirements.txt",
    "src/README.md",
    "lib/main.dart",
    "lib/pubspec.yaml",
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for file in files:
    open(file, 'a').close()  # Create an empty file
