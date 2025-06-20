"""
Functions to compare predictions with actual data
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List
import os
from datetime import datetime

def compare_predictions(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    join_columns: List[str] = ['Site_No', 'Item_No', 'Date'],
    prediction_column: str = 'Predicted_Quantity',
    actual_column: str = 'Quantity',
    include_all: bool = True
) -> pd.DataFrame:
    """
    Compare predictions with actual data
    
    Args:
        predictions: DataFrame with predictions
        actuals: DataFrame with actual values
        join_columns: Columns to use for matching predictions and actuals
        prediction_column: Column name for predictions
        actual_column: Column name for actual values
        include_all: Whether to include all records (True) or only matched (False)
    
    Returns:
        DataFrame with predictions and actual values
    """
    print(f"Comparing predictions with actual data using {join_columns}")
    
    # Verify join columns exist in both datasets
    for col in join_columns:
        if col not in predictions.columns:
            raise ValueError(f"Join column {col} not found in predictions DataFrame")
        if col not in actuals.columns:
            raise ValueError(f"Join column {col} not found in actuals DataFrame")
    
    # If Date is used for joining and date formats are different, normalize
    for date_col in ['Date', 'Start_Date', 'End_Date']:
        if date_col in join_columns:
            if date_col in predictions.columns:
                predictions[date_col] = pd.to_datetime(predictions[date_col]).dt.date
            if date_col in actuals.columns:  
                actuals[date_col] = pd.to_datetime(actuals[date_col]).dt.date
    
    # Select required columns from both dataframes
    pred_cols = join_columns + [prediction_column] + [col for col in predictions.columns if col not in join_columns and col != prediction_column]
    act_cols = join_columns + [actual_column]
    
    pred_data = predictions[pred_cols].copy()
    act_data = actuals[act_cols].copy()
    
    # Determine merge strategy based on include_all parameter
    if include_all:
        # Full outer join to keep all records
        merged = pred_data.merge(
            act_data,
            on=join_columns,
            how='outer',
            suffixes=('', '_act')
        )
    else:
        # Inner join to keep only matched records
        merged = pred_data.merge(
            act_data,
            on=join_columns,
            how='inner',
            suffixes=('', '_act')
        )
    
    # Calculate error metrics
    merged['Error'] = merged[prediction_column].fillna(0) - merged[actual_column].fillna(0)
    merged['Abs_Error'] = np.abs(merged['Error'])
    merged['Pct_Error'] = np.where(
        merged[actual_column] != 0,
        100 * merged['Error'] / merged[actual_column],
        np.nan
    )
    merged['Abs_Pct_Error'] = np.abs(merged['Pct_Error'])
    
    # Create record status column
    merged['Status'] = 'matched'
    if include_all:
        merged.loc[merged[prediction_column].isna(), 'Status'] = 'actual_only'
        merged.loc[merged[actual_column].isna(), 'Status'] = 'prediction_only'
    
    print(f"Created comparison with {len(merged)} rows")
    print(f"  - Matched records: {(merged['Status'] == 'matched').sum()}")
    
    if include_all:
        print(f"  - Prediction only: {(merged['Status'] == 'prediction_only').sum()}")
        print(f"  - Actual only: {(merged['Status'] == 'actual_only').sum()}")
    
    return merged

def save_comparison_results(
    comparison_df: pd.DataFrame,
    output_path: Optional[str] = None,
    output_dir: str = 'output/compare',
    formats: List[str] = ['csv', 'excel']
) -> Dict[str, str]:
    """
    Save comparison results to file(s)
    
    Args:
        comparison_df: Comparison DataFrame
        output_path: Path for output file (without extension)
        output_dir: Directory for output files if path not specified
        formats: List of output formats ('csv', 'excel', 'parquet')
    
    Returns:
        Dictionary of output paths by format
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate default output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'comparison_{timestamp}')
    elif not os.path.isabs(output_path):
        # If relative path provided, prepend with output_dir
        output_path = os.path.join(output_dir, output_path)
    
    # Save in requested formats
    output_files = {}
    
    for fmt in formats:
        if fmt.lower() == 'csv':
            file_path = f"{output_path}.csv"
            comparison_df.to_csv(file_path, index=False)
            output_files['csv'] = file_path
            print(f"Comparison saved to {file_path}")
        
        elif fmt.lower() == 'excel':
            file_path = f"{output_path}.xlsx"
            comparison_df.to_excel(file_path, index=False, sheet_name='Comparison')
            output_files['excel'] = file_path
            print(f"Comparison saved to {file_path}")
        
        elif fmt.lower() == 'parquet':
            file_path = f"{output_path}.parquet"
            comparison_df.to_parquet(file_path, index=False)
            output_files['parquet'] = file_path
            print(f"Comparison saved to {file_path}")
    
    return output_files