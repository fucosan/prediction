"""
Functions to load prediction and actual data for comparison
"""

import pandas as pd
from typing import Optional, List, Tuple
import os

def load_prediction_data(
    prediction_path: str,
    required_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load prediction data from file
    
    Args:
        prediction_path: Path to prediction file
        required_columns: List of required columns (validates data has these)
    
    Returns:
        DataFrame containing predictions
    """
    print(f"Loading prediction data from {prediction_path}")
    
    # Determine file type and load
    if prediction_path.endswith('.csv'):
        predictions = pd.read_csv(prediction_path)
    elif prediction_path.endswith('.xlsx') or prediction_path.endswith('.xls'):
        predictions = pd.read_excel(prediction_path)
    elif prediction_path.endswith('.parquet'):
        predictions = pd.read_parquet(prediction_path)
    else:
        raise ValueError(f"Unsupported file format: {prediction_path}")
    
    print(f"Loaded {len(predictions)} prediction records")
    
    # Handle Date vs Start_Date/End_Date mapping
    if 'Date' not in predictions.columns and 'Start_Date' in predictions.columns:
        if required_columns and 'Date' in required_columns:
            # If Date is required but we have Start_Date/End_Date, modify required_columns
            modified_required = [col for col in required_columns if col != 'Date']
            if 'Start_Date' not in modified_required:
                modified_required.append('Start_Date')
            required_columns = modified_required
            print("Using 'Start_Date' instead of 'Date' for predictions")
    
    # Validate required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in predictions.columns]
        if missing_cols:
            raise ValueError(f"Prediction data missing required columns: {missing_cols}")
    
    # Convert date columns if present
    for date_col in ['Start_Date', 'End_Date', 'Date']:
        if date_col in predictions.columns:
            predictions[date_col] = pd.to_datetime(predictions[date_col])
    
    return predictions

def load_actual_data(
    actual_path: str,
    required_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load actual data from file
    
    Args:
        actual_path: Path to actual data file
        required_columns: List of required columns (validates data has these)
    
    Returns:
        DataFrame containing actual data
    """
    print(f"Loading actual data from {actual_path}")
    
    # Determine file type and load
    if actual_path.endswith('.csv'):
        actuals = pd.read_csv(actual_path)
    elif actual_path.endswith('.xlsx') or actual_path.endswith('.xls'):
        actuals = pd.read_excel(actual_path)
    elif actual_path.endswith('.parquet'):
        actuals = pd.read_parquet(actual_path)
    else:
        raise ValueError(f"Unsupported file format: {actual_path}")
    
    print(f"Loaded {len(actuals)} actual data records")
    
    # Validate required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in actuals.columns]
        if missing_cols:
            raise ValueError(f"Actual data missing required columns: {missing_cols}")
    
    # Convert date columns if present
    for date_col in ['Start_Date', 'End_Date', 'Date']:
        if date_col in actuals.columns:
            actuals[date_col] = pd.to_datetime(actuals[date_col])
    
    return actuals

def preprocess_data_for_comparison(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess prediction and actual data for comparison
    
    Args:
        predictions: DataFrame with predictions
        actuals: DataFrame with actual values
    
    Returns:
        Tuple of (processed predictions, processed actuals)
    """
    print("Preprocessing data for comparison")
    
    proc_predictions = predictions.copy()
    proc_actuals = actuals.copy()
    
    # Handle date column differences
    if 'Date' in proc_actuals.columns and 'Date' not in proc_predictions.columns:
        if 'End_Date' in proc_predictions.columns:
            print("Using End_Date from predictions to match with Date in actuals")
            proc_predictions['Date'] = proc_predictions['End_Date']
    
    if 'Date' in proc_actuals.columns and 'Start_Date' in proc_predictions.columns and 'End_Date' in proc_predictions.columns:
        print("Filtering actuals to match prediction date range")
        min_date = proc_predictions['Start_Date'].min()
        max_date = proc_predictions['End_Date'].max()
        proc_actuals = proc_actuals[(proc_actuals['Date'] >= min_date) & (proc_actuals['Date'] <= max_date)]
    
    return proc_predictions, proc_actuals