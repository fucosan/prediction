"""
Functions to load prediction and actual data for comparison
"""

import pandas as pd
from typing import Optional, List, Tuple
import os
from datetime import datetime, timedelta

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
    
    # Bi-weekly first expects Start_Date and End_Date
    if 'Start_Date' not in predictions.columns and 'Date' in predictions.columns:
        print("Converting 'Date' to 'Start_Date'/'End_Date' for bi-weekly compatibility")
        predictions['Start_Date'] = predictions['Date']
        predictions['End_Date'] = predictions['Date'] + pd.Timedelta(days=13)  # Assume 2-week periods
    
    # Convert date columns to datetime
    for date_col in ['Start_Date', 'End_Date', 'Date']:
        if date_col in predictions.columns:
            predictions[date_col] = pd.to_datetime(predictions[date_col])
    
    # Validate required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in predictions.columns]
        if missing_cols:
            raise ValueError(f"Prediction data missing required columns: {missing_cols}")
    
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
    
    # Convert date columns to datetime
    for date_col in ['Start_Date', 'End_Date', 'Date']:
        if date_col in actuals.columns:
            actuals[date_col] = pd.to_datetime(actuals[date_col])
    
    return actuals

def aggregate_actuals_by_date_range(
    actuals: pd.DataFrame,
    predictions: pd.DataFrame,
    key_columns: List[str] = ['Site_No', 'Item_No'],
    date_column: str = 'Date',
    actual_value_column: str = 'Quantity'
) -> pd.DataFrame:
    """
    Aggregate actual data to match prediction date ranges
    
    Args:
        actuals: DataFrame with actual data
        predictions: DataFrame with prediction data containing date ranges
        key_columns: List of columns to use as keys (Site_No, Item_No)
        date_column: Column name for date in actuals
        actual_value_column: Column name for values to be aggregated
    
    Returns:
        DataFrame with actual data aggregated to match prediction date ranges
    """
    print("Aggregating actual data to match prediction date ranges")
    
    # Check if predictions contain Start_Date and End_Date
    if 'Start_Date' not in predictions.columns or 'End_Date' not in predictions.columns:
        print("Warning: Predictions do not contain Start_Date/End_Date columns. Skipping aggregation.")
        return actuals
    
    # Ensure date columns are datetime
    actuals[date_column] = pd.to_datetime(actuals[date_column])
    predictions['Start_Date'] = pd.to_datetime(predictions['Start_Date'])
    predictions['End_Date'] = pd.to_datetime(predictions['End_Date'])
    
    # Extract unique date ranges and keys from predictions
    date_ranges = predictions[key_columns + ['Start_Date', 'End_Date']].drop_duplicates()
    print(f"Found {len(date_ranges)} unique date ranges in prediction data")
    
    # Initialize empty DataFrame for aggregated results
    aggregated = []
    
    # Process each date range
    for _, row in date_ranges.iterrows():
        # Extract key values and date range
        key_values = {col: row[col] for col in key_columns}
        start_date = row['Start_Date']
        end_date = row['End_Date']
        
        # Filter actuals for this key and date range
        filter_conditions = [
            (actuals[date_column] >= start_date),
            (actuals[date_column] <= end_date)
        ]
        
        for col in key_columns:
            filter_conditions.append(actuals[col] == key_values[col])
        
        matching_actuals = actuals[pd.concat(filter_conditions, axis=1).all(axis=1)]
        
        # If there are matching records, aggregate them
        if len(matching_actuals) > 0:
            agg_value = matching_actuals[actual_value_column].sum()
            
            # Create result row
            result = dict(key_values)
            result['Start_Date'] = start_date
            result['End_Date'] = end_date
            result[date_column] = end_date  # Use end date as the reference date
            result[actual_value_column] = agg_value
            aggregated.append(result)
    
    if not aggregated:
        print("Warning: No matching records found in actual data for prediction date ranges")
        return pd.DataFrame(columns=actuals.columns)
    
    # Convert results to DataFrame
    aggregated_df = pd.DataFrame(aggregated)
    
    print(f"Aggregated actual data into {len(aggregated_df)} records matching prediction date ranges")
    
    return aggregated_df

def preprocess_data_for_comparison(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    key_columns: List[str] = ['Site_No', 'Item_No']
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
    
    # Check if predictions contain Start_Date/End_Date and actuals contain Date
    has_date_ranges = 'Start_Date' in proc_predictions.columns and 'End_Date' in proc_predictions.columns
    has_date = 'Date' in proc_actuals.columns
    
    if has_date_ranges and has_date:
        print("Aggregating actual data to match prediction date ranges")
        proc_actuals = aggregate_actuals_by_date_range(
            proc_actuals, 
            proc_predictions,
            key_columns=key_columns
        )
    else:
        print("Warning: Cannot aggregate actuals by date range. Missing required date columns.")
    
    return proc_predictions, proc_actuals