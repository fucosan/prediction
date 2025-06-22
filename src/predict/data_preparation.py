"""
Data preparation functions for sales predictions
"""

import pandas as pd
import numpy as np
import os
import joblib
from typing import Tuple, Dict, Optional, List

from src.preprocess import process_sales_data, load_data
from config.config import MODEL_DIR, SCALER_PATH, ENCODER_PATH

def prepare_prediction_data(
    data_path: str,
    numerical_columns: List[str],
    categorical_columns: List[str],
    scaler_path: str = SCALER_PATH,
    encoder_path: str = ENCODER_PATH
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Prepare data for making predictions using bi-weekly first approach
    
    Args:
        data_path: Path to input data file
        numerical_columns: List of numerical columns
        categorical_columns: List of categorical columns
        scaler_path: Path to the saved scaler
        encoder_path: Path to the saved encoder
    
    Returns:
        Tuple of (X for prediction, metadata DataFrame, feature metadata)
    """
    print(f"Preparing prediction data from {data_path}")
    
    # Load data
    data = load_data(data_path)
    print(f"Loaded {len(data)} records")
    
    # Import bi-weekly specific parameters
    from config.config import BI_WEEKLY_LAG_PERIODS, BI_WEEKLY_WINDOW_SIZES
    
    # Process using bi-weekly first approach
    processed_data, feature_metadata = process_sales_data(
        data,
        bi_weekly_lag_periods=BI_WEEKLY_LAG_PERIODS,
        bi_weekly_window_sizes=BI_WEEKLY_WINDOW_SIZES,
        save_metadata=False  # No need to save metadata during prediction
    )
    print(f"Processed data shape: {processed_data.shape}")
    
    # Extract metadata for later
    metadata_cols = ['Site_No', 'Item_No']
    if 'Start_Date' in processed_data.columns:
        metadata_cols.append('Start_Date')
    if 'End_Date' in processed_data.columns:
        metadata_cols.append('End_Date')
    
    metadata = processed_data[metadata_cols].copy()
    
    # Prepare features for prediction
    X = processed_data.copy()
    
    # Remove date columns and target column if present
    for col in ['Start_Date', 'End_Date', 'Date', 'Quantity']:
        if col in X.columns:
            X = X.drop(columns=[col])
    
    # Load scaler and encoder if they exist
    scaler = None
    encoder = None
    
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"Loaded scaler from {scaler_path}")
    
    if os.path.exists(encoder_path):
        encoder = joblib.load(encoder_path)
        print(f"Loaded encoder from {encoder_path}")
    
    # Apply transformations
    if scaler is not None:
        # Apply only to numerical columns that exist in both X and were used to fit the scaler
        scale_cols = [col for col in numerical_columns if col in X.columns and col in scaler.feature_names_in_]
        if scale_cols:
            X[scale_cols] = scaler.transform(X[scale_cols])
    
    # Handle categorical columns
    print("Converting categorical columns to numeric...")
    for col in X.select_dtypes(include=['object']).columns:
        if col in categorical_columns:
            # Try to convert directly to numeric first
            try:
                X[col] = pd.to_numeric(X[col], errors='raise')
                print(f"  - Converted {col} to numeric directly")
                continue
            except:
                pass
                
            # If specific encoder exists, use it
            if encoder is not None and isinstance(encoder, dict) and col in encoder:
                X[col] = X[col].astype(str).map(encoder[col]).fillna(-1).astype(int)
                print(f"  - Encoded {col} using saved encoder mapping")
            else:
                # Fallback: use label encoding on the fly
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                print(f"  - Encoded {col} using new LabelEncoder (no saved mapping found)")
    
    # Final check for any remaining object columns
    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        print(f"Warning: Converting remaining object columns to numeric: {object_cols}")
        for col in object_cols:
            X[col] = pd.factorize(X[col].astype(str))[0]
    
    # Final verification - ensure no object types remain
    final_object_cols = X.select_dtypes(include=['object']).columns.tolist()
    if final_object_cols:
        raise ValueError(f"Failed to convert columns to numeric: {final_object_cols}")
    
    print(f"Prepared feature matrix with shape {X.shape}, all numeric data types")
    return X, metadata, feature_metadata