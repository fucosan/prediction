"""
Data cleaning functions for sales prediction preprocessing
"""

import pandas as pd
from typing import List, Dict
from config.config import KEEP_COLS, PCT_COLUMNS, COLUMN_PAIRS

def clean_initial_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the specified columns and convert percentage columns from strings (e.g., '3%') to float values (e.g., 0.03).
    
    Args:
        df: Raw DataFrame with sales data
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Keep only the specified columns that exist in the DataFrame
    existing_cols = [col for col in KEEP_COLS if col in df.columns]
    df = df[existing_cols]
    
    # Convert percentage columns to proper float values
    for col in PCT_COLUMNS:
        if col in df.columns:
            # Handle string percentages like '3%'
            if df[col].dtype == 'object':
                df[col] = df[col].replace('', '0%')  # Handle empty strings
                df[col] = df[col].str.rstrip('%').astype(float) / 100
            else:
                df[col] = df[col] / 100
    
    return df

def normalize_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize names for key identifier columns:
    - Map Item_No to the most common Item_Name
    - Map Site_No to the most common Site_Name
    - Map Vendor_No to the most common Vendor_Name
    
    Args:
        df: DataFrame with identifier columns and corresponding name columns
        
    Returns:
        DataFrame with normalized name columns
    """
    df = df.copy()
    
    for code_col, name_col in COLUMN_PAIRS:
        # Only process if both columns exist in the DataFrame
        if code_col in df.columns and name_col in df.columns:
            print(f"Normalizing {name_col} based on {code_col}")
            
            # Create mapping from code to most common name
            name_mapping = df.groupby(code_col)[name_col].agg(
                lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else None
            ).to_dict()
            
            # Apply mapping to standardize names
            df[name_col] = df[code_col].map(name_mapping)
            
    return df

def normalize_daily_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the daily time series data by:
    1. Ensuring each (Site_No, Item_No, Date) combination is unique
    2. Creating a complete daily date range for each (Site_No, Item_No) pair
    3. Filling missing dates with Quantity = 0
    4. Properly handling other columns for newly generated dates
    
    Args:
        df: DataFrame with at least Date, Site_No, Item_No, and Quantity columns
        
    Returns:
        Normalized DataFrame with complete daily time series
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Identify all columns except for Quantity that need to be preserved
    preserved_cols = [col for col in df.columns if col != 'Quantity']
    
    # Aggregate by (Site_No, Item_No, Date)
    # For Quantity, use sum
    # For other columns, use first value (assuming they're constant within a day)
    agg_dict = {'Quantity': 'sum'}
    for col in preserved_cols:
        if col not in ['Site_No', 'Item_No', 'Date']:
            agg_dict[col] = 'first'
    
    df_agg = df.groupby(['Site_No', 'Item_No', 'Date']).agg(agg_dict).reset_index()
    
    # Get min and max dates for each (Site_No, Item_No) pair
    date_ranges = df_agg.groupby(['Site_No', 'Item_No']).agg({'Date': ['min', 'max']})
    date_ranges.columns = ['min_date', 'max_date']
    date_ranges = date_ranges.reset_index()
    
    # Create a complete daily time series for each (Site_No, Item_No) pair
    normalized_dfs = []
    
    for _, row in date_ranges.iterrows():
        site = row['Site_No']
        item = row['Item_No']
        min_date = row['min_date']
        max_date = row['max_date']
        
        # Create date range
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        # Create a DataFrame with the complete date range
        complete_df = pd.DataFrame({'Date': date_range})
        complete_df['Site_No'] = site
        complete_df['Item_No'] = item
        
        # Get data for this site-item pair
        site_item_data = df_agg[(df_agg['Site_No'] == site) & (df_agg['Item_No'] == item)]
        
        # Merge with the actual data
        complete_df = pd.merge(complete_df, site_item_data, on=['Site_No', 'Item_No', 'Date'], how='left')
        
        # Fill missing quantities with 0
        complete_df['Quantity'] = complete_df['Quantity'].fillna(0)
        
        # Handle other columns: forward fill first, then backfill if needed
        for col in complete_df.columns:
            if col not in ['Site_No', 'Item_No', 'Date', 'Quantity'] and complete_df[col].isna().any():
                # Try forward fill first
                complete_df[col] = complete_df[col].ffill()
                
                # If still has NaN values, try backfill
                if complete_df[col].isna().any():
                    complete_df[col] = complete_df[col].bfill()
                
                # If still has NaN, use first non-NaN value in the group
                if complete_df[col].isna().any() and not site_item_data[col].isna().all():
                    first_value = site_item_data[col].dropna().iloc[0] if len(site_item_data) > 0 else None
                    if first_value is not None:
                        complete_df[col] = complete_df[col].fillna(first_value)
        
        normalized_dfs.append(complete_df)
    
    # Combine all normalized DataFrames
    result = pd.concat(normalized_dfs, ignore_index=True)
    
    return result