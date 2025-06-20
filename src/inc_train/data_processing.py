"""
Data processing functions for incremental training
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Tuple, Dict, Optional

# Import preprocessing functions for consistency
from src.preprocess import process_sales_data, load_data, preprocess_raw_data
from config.config import FEATURE_STORE_PATH, PREPROCESS_DIR, LOOKBACK_DAYS

def process_new_data(
    new_data_path: str,
    prefilter: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Process new sales data for incremental training
    
    Args:
        new_data_path: Path to the new data file
        prefilter: Whether to prefilter the data using data_mapper
    
    Returns:
        Tuple of (processed new data, feature metadata)
    """
    print(f"Processing new data from {new_data_path}")
    
    # Prefilter if requested
    if prefilter:
        filtered_path = preprocess_raw_data(new_data_path)
        new_data = load_data(filtered_path)
    else:
        new_data = load_data(new_data_path)
    
    # Process using the same pipeline as initial training
    processed_new_data, feature_metadata = process_sales_data(new_data)
    
    # Save as temporary file for inspection
    temp_path = os.path.join(PREPROCESS_DIR, 'new_processed_data.parquet')
    processed_new_data.to_parquet(temp_path, index=False)
    print(f"New processed data saved to {temp_path}")
    
    return processed_new_data, feature_metadata

def update_feature_store(
    new_processed_data: pd.DataFrame,
    feature_store_path: str = FEATURE_STORE_PATH,
    lookback_days: int = LOOKBACK_DAYS
) -> pd.DataFrame:
    """
    Update the feature store with new data
    
    Args:
        new_processed_data: Processed new data
        feature_store_path: Path to the feature store
        lookback_days: Number of days to keep in the feature store
    
    Returns:
        Updated feature store
    """
    print(f"Updating feature store at {feature_store_path}")
    
    # Check if feature store exists
    if not os.path.exists(feature_store_path):
        print("Feature store not found. Creating new feature store.")
        updated_store = new_processed_data.copy()
    else:
        # Load existing feature store
        existing_store = pd.read_parquet(feature_store_path)
        print(f"Loaded existing feature store with {len(existing_store)} records")
        
        # Convert date columns to datetime if needed
        for date_col in ['Start_Date', 'End_Date']:
            if date_col in existing_store.columns:
                existing_store[date_col] = pd.to_datetime(existing_store[date_col])
            if date_col in new_processed_data.columns:
                new_processed_data[date_col] = pd.to_datetime(new_processed_data[date_col])
        
        # Identify duplicate records (same Site_No, Item_No, and date)
        if 'End_Date' in existing_store.columns and 'End_Date' in new_processed_data.columns:
            # Identify records in existing store that also exist in new data (by Site_No, Item_No, End_Date)
            merge_keys = ['Site_No', 'Item_No', 'End_Date']
            duplicates = existing_store.merge(
                new_processed_data[merge_keys], 
                on=merge_keys, 
                how='inner'
            )
            
            # Remove duplicates from existing store
            if not duplicates.empty:
                print(f"Found {len(duplicates)} duplicate records. Removing from existing store.")
                existing_store = existing_store.merge(
                    duplicates[merge_keys], 
                    on=merge_keys, 
                    how='left', 
                    indicator=True
                )
                existing_store = existing_store[existing_store['_merge'] == 'left_only'].drop('_merge', axis=1)
        
        # Concatenate with new data
        updated_store = pd.concat([existing_store, new_processed_data], ignore_index=True)
        
        # Filter to keep only recent data if lookback_days > 0
        if lookback_days > 0 and 'End_Date' in updated_store.columns:
            latest_date = updated_store['End_Date'].max()
            cutoff_date = latest_date - pd.Timedelta(days=lookback_days)
            old_len = len(updated_store)
            updated_store = updated_store[updated_store['End_Date'] >= cutoff_date]
            print(f"Filtered feature store to keep {lookback_days} days of history.")
            print(f"Removed {old_len - len(updated_store)} old records.")
    
    # Sort by date
    if 'End_Date' in updated_store.columns:
        updated_store = updated_store.sort_values('End_Date')
    
    # Fix data type issues for problematic columns
    problematic_cols = [
        'Periodic_Disc_Amount_incl_PPN_IDR', 
        'Amount_Discount_Tambahan_incl_PPN_IDR',
        'Line_Disc_Amount_Approval_IDR',
        'Total_Sales_Incl_PPN_IDR',
        'Add_Periodic_Disc_Amount_IDR'
    ]
    
    for col in problematic_cols:
        if col in updated_store.columns:
            try:
                # First, try to convert to float - handles most numeric cases
                updated_store[col] = pd.to_numeric(updated_store[col], errors='coerce')
                print(f"Converted {col} to numeric type")
            except:
                # If that fails, convert to string as a fallback
                updated_store[col] = updated_store[col].astype(str)
                print(f"Converted {col} to string type")
    
    # Save updated feature store
    try:
        updated_store.to_parquet(feature_store_path, index=False)
        print(f"Updated feature store saved with {len(updated_store)} records.")
    except Exception as e:
        # If saving fails, try more aggressive handling
        print(f"Error saving feature store: {e}")
        print("Attempting more aggressive data type conversion...")
        
        # For all object columns, convert to string
        for col in updated_store.select_dtypes(include=['object']).columns:
            updated_store[col] = updated_store[col].astype(str)
            print(f"Converted {col} to string type")
        
        # Try saving again
        updated_store.to_parquet(feature_store_path, index=False)
        print(f"Updated feature store saved with {len(updated_store)} records after type conversion.")
    
    return updated_store