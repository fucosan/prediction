"""
Data preparation functions for model training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Tuple, List, Dict, Optional, Union
from datetime import datetime

from config.config import DATE_COLUMNS, TRAIN_TEST_SPLIT_PARAMS, MODEL_DIR

def prepare_model_data(
    df: pd.DataFrame,
    numerical_columns: List[str],
    categorical_columns: List[str],
    model_type: str = 'tree',
    encoding_type: str = 'label',
    target_column: str = 'Quantity'
) -> Tuple[pd.DataFrame, pd.Series, object, object]:
    """
    Prepare data for modeling by:
    1. Removing date columns
    2. Encoding categorical features
    3. Scaling numerical features if needed
    4. Creating feature matrix X and target vector y
    
    Args:
        df: Processed DataFrame
        numerical_columns: List of numerical feature columns
        categorical_columns: List of categorical feature columns
        model_type: 'tree' or 'linear' (determines if scaling is applied)
        encoding_type: 'label' or 'onehot'
        target_column: Name of target column
        
    Returns:
        Tuple of (X, y, scaler, encoder)
    """
    print("Preparing data for modeling...")
    df = df.copy()
    
    # Remove date columns
    for col in DATE_COLUMNS:
        if col in df.columns:
            print(f"Removing {col} column")
            df = df.drop(columns=[col])
    
    # Initialize encoders and scalers
    scaler = StandardScaler()
    encoder = LabelEncoder()
    
    # Create feature matrix X and target vector y
    y = df[target_column]
    
    # Process categorical features
    for col in categorical_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                if encoding_type == 'label':
                    df[col] = encoder.fit_transform(df[col].astype(str))
                    print(f"Label encoded: {col}")
    
    # Process numerical features
    if model_type == 'linear':
        # Only scale for linear models, tree-based models don't need scaling
        numerical_features = [col for col in numerical_columns if col in df.columns and col != target_column]
        if numerical_features:
            df[numerical_features] = scaler.fit_transform(df[numerical_features])
            print(f"Scaled {len(numerical_features)} numerical features")
    
    # Get all features excluding target
    X = df.drop(columns=[target_column])
    
    return X, y, scaler, encoder

def split_time_series_data(
    df: pd.DataFrame, 
    X: pd.DataFrame, 
    y: pd.Series,
    use_all_for_train: bool = TRAIN_TEST_SPLIT_PARAMS['use_all_for_train'],
    test_size: float = TRAIN_TEST_SPLIT_PARAMS['test_size'],
    validation_size: float = TRAIN_TEST_SPLIT_PARAMS['validation_size'],
    random_state: int = TRAIN_TEST_SPLIT_PARAMS['random_state']
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.
    For time series data, this will use the most recent data as test set.
    
    Args:
        df: Original DataFrame with date information
        X: Feature matrix
        y: Target vector
        use_all_for_train: If True, use all data for training
        test_size: Fraction of data to use for testing
        validation_size: Fraction of training data to use for validation
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_valid, X_test, y_train, y_valid, y_test)
    """
    print("Creating train/test/validation splits...")
    
    if 'End_Date' in df.columns:
        # Sort by End_Date to ensure chronological order
        df['End_Date'] = pd.to_datetime(df['End_Date'])
        sorted_indices = df.sort_values('End_Date').index
        X = X.loc[sorted_indices]
        y = y.loc[sorted_indices]
        
        # Calculate split points
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        if use_all_for_train:
            # Use all data for training (for final model)
            X_train = X
            y_train = y
            # Use a small portion for validation during training
            n_valid = int(n_samples * validation_size)
            
            X_valid = X.iloc[-n_valid:]
            y_valid = y.iloc[-n_valid:]
            
            # Use the most recent data for testing
            X_test = X.iloc[-n_test:]
            y_test = y.iloc[-n_test:]
        else:
            # Split into train and test
            X_train = X.iloc[:-n_test]
            X_test = X.iloc[-n_test:]
            y_train = y.iloc[:-n_test]
            y_test = y.iloc[-n_test:]
            
            # Further split train into train and validation
            n_valid = int(len(X_train) * validation_size / (1 - test_size))
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, y_train, test_size=n_valid, random_state=random_state
            )
    else:
        # Fallback to random splits if no date column
        print("Warning: No 'End_Date' column found, using random splits instead of time-based")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=validation_size, random_state=random_state
        )
    
    print(f"Split sizes: Train={len(X_train)}, Validation={len(X_valid)}, Test={len(X_test)}")
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test