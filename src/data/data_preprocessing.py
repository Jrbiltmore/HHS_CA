# Folder: src
# File: data_preprocessing.py

import pandas as pd
from typing import Optional, Union

def clean_data(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        # Drop duplicates
        data.drop_duplicates(inplace=True)
        
        # Handle missing values using forward and backward filling
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)
        
        # Convert date columns to datetime
        date_columns = [col for col in data.columns if 'date' in col.lower()]
        for col in date_columns:
            data[col] = pd.to_datetime(data[col])
        
        # Remove outliers
        numerical_columns = data.select_dtypes(include=['number']).columns
        for col in numerical_columns:
            data = remove_outliers(data, col)
        
        # Remove irrelevant columns
        irrelevant_columns = ['id', 'unnecessary_column']
        data.drop(columns=irrelevant_columns, inplace=True)
        
        # Other cleaning steps...
        
        return data
    
    except Exception as e:
        print("An error occurred during data preprocessing:", e)
        return None

def remove_outliers(data: pd.DataFrame, column: Union[str, int]) -> pd.DataFrame:
    # Remove outliers using Z-score method
    z_scores = (data[column] - data[column].mean()) / data[column].std()
    data = data[(z_scores.abs() < 3)]
    return data
