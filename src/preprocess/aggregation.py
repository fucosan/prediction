"""
Aggregation functions for sales prediction
"""

import pandas as pd
from typing import List

def aggregate_biweekly(df: pd.DataFrame, period_days: int = 14) -> pd.DataFrame:
    """
    Aggregate the enriched daily dataset into non-overlapping periods.
    
    Args:
        df: DataFrame with Date, Site_No, Item_No, and enriched features
        period_days: Number of days in each period (default: 14)
        
    Returns:
        Aggregated DataFrame with one row per (Site_No, Item_No, period)
    """
    df = df.copy()
    
    # Define a reference date (earliest date in the dataset)
    reference_date = df['Date'].min()
    
    # Calculate days since the reference date
    df['days_since_reference'] = (df['Date'] - reference_date).dt.days
    
    # Create non-overlapping period IDs (integer division)
    df['period_id'] = df['days_since_reference'] // period_days
    
    # Calculate the start and end dates for each period
    df['Start_Date'] = reference_date + pd.to_timedelta(df['period_id'] * period_days, unit='D')
    df['End_Date'] = df['Start_Date'] + pd.to_timedelta(period_days - 1, unit='D')
    
    # Identify feature columns (excluding key columns and date-related columns)
    key_cols = ['Site_No', 'Item_No', 'Date', 'period_id', 'Start_Date', 'End_Date', 'days_since_reference']
    feature_cols = [col for col in df.columns if col not in key_cols]
    
    # Define aggregation dictionary
    agg_dict = {
        'Start_Date': 'first',
        'End_Date': 'first',
        'Quantity': 'sum'
    }
    
    # For feature columns, use mean aggregation (or appropriate aggregation)
    for col in feature_cols:
        if col != 'Quantity':
            if col == 'IsHoliday':
                # For boolean holiday flag, use max (True if any day in period is holiday)
                agg_dict[col] = 'max'
            elif col.startswith(('lag_', 'rolling_', 'days_since_last_sale')):
                agg_dict[col] = 'mean'
            else:
                # For other columns, use last value or appropriate aggregation
                agg_dict[col] = 'last'
    
    # Aggregate by (Site_No, Item_No, period_id)
    aggregated = df.groupby(['Site_No', 'Item_No', 'period_id']).agg(agg_dict).reset_index()
    
    # Drop intermediate columns (safely)
    columns_to_drop = []
    if 'period_id' in aggregated.columns:
        columns_to_drop.append('period_id')
    if 'days_since_reference' in aggregated.columns:
        columns_to_drop.append('days_since_reference')
    
    if columns_to_drop:
        aggregated = aggregated.drop(columns=columns_to_drop)
    
    # Keep dates as proper datetime objects instead of converting to strings
    # This ensures they're handled consistently throughout the pipeline
    aggregated['End_Date'] = pd.to_datetime(aggregated['End_Date'])
    aggregated['Start_Date'] = pd.to_datetime(aggregated['Start_Date'])
    
    return aggregated