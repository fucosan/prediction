"""
Feature engineering functions for sales prediction
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from config.config import LAG_PERIODS, WINDOW_SIZES, ROLLING_METRICS, TIME_CATEGORICALS, DATE_FIELDS, ID_COLUMNS

def add_lag_features(df: pd.DataFrame, lag_periods: List[int] = LAG_PERIODS) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add lag features to the DataFrame.
    
    Args:
        df: DataFrame with Date, Site_No, Item_No, and Quantity columns
        lag_periods: List of lag periods to create features for
        
    Returns:
        Tuple of (DataFrame with lag features, List of feature column names)
    """
    df = df.copy()
    feature_cols = []
    
    # Sort by (Site_No, Item_No, Date) to ensure correct lag calculation
    df = df.sort_values(['Site_No', 'Item_No', 'Date'])
    
    # Create lag features
    for lag in lag_periods:
        col_name = f'lag_{lag}'
        df[col_name] = df.groupby(['Site_No', 'Item_No'])['Quantity'].shift(lag)
        # Fill NaN values with 0 as required by the prompt
        df[col_name] = df[col_name].fillna(0)
        feature_cols.append(col_name)
    
    return df, feature_cols

def add_rolling_window_features(df: pd.DataFrame, 
                               window_sizes: List[int] = WINDOW_SIZES, 
                               metrics: List[str] = ROLLING_METRICS) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add rolling window features to the DataFrame, using lagged values to prevent data leakage.
    
    Args:
        df: DataFrame with Date, Site_No, Item_No, and Quantity columns
        window_sizes: List of window sizes to create features for
        metrics: List of metrics to calculate (default: mean, sum, std)
        
    Returns:
        Tuple of (DataFrame with rolling features, List of feature column names)
    """
    df = df.copy()
    feature_cols = []
    
    # Sort by (Site_No, Item_No, Date) to ensure correct rolling calculation
    df = df.sort_values(['Site_No', 'Item_No', 'Date'])
    
    # Shift Quantity by 1 to prevent using current day's value in rolling calculations
    df['Quantity_lagged'] = df.groupby(['Site_No', 'Item_No'])['Quantity'].shift(1)
    # Fill NaN values with 0 as required by the prompt
    df['Quantity_lagged'] = df['Quantity_lagged'].fillna(0)
    
    # Create rolling window features
    for window in window_sizes:
        for metric in metrics:
            if metric == 'mean':
                col_name = f'rolling_mean_{window}'
                df[col_name] = df.groupby(['Site_No', 'Item_No'])['Quantity_lagged'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
            elif metric == 'sum':
                col_name = f'rolling_sum_{window}'
                df[col_name] = df.groupby(['Site_No', 'Item_No'])['Quantity_lagged'].transform(
                    lambda x: x.rolling(window, min_periods=1).sum()
                )
            elif metric == 'std':
                col_name = f'rolling_std_{window}'
                df[col_name] = df.groupby(['Site_No', 'Item_No'])['Quantity_lagged'].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )
            
            # Fill any NaN values with 0 (this may happen with std calculation when all values are the same)
            df[col_name] = df[col_name].fillna(0)
            feature_cols.append(col_name)
    
    # Drop the temporary column
    df = df.drop(columns=['Quantity_lagged'])
    
    return df, feature_cols

def add_days_since_last_sale(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add a feature that tracks days since last sale.
    
    Args:
        df: DataFrame with Date, Site_No, Item_No, and Quantity columns
        
    Returns:
        Tuple of (DataFrame with days_since_last_sale feature, List of feature column names)
    """
    df = df.copy()
    feature_cols = ['days_since_last_sale']
    
    # Sort by (Site_No, Item_No, Date)
    df = df.sort_values(['Site_No', 'Item_No', 'Date'])
    
    # Initialize days_since_last_sale
    df['days_since_last_sale'] = 0
    
    # Calculate days_since_last_sale for each (Site_No, Item_No) group
    for (site, item), group in df.groupby(['Site_No', 'Item_No']):
        days_count = 0
        days_since_list = []
        
        for quantity in group['Quantity']:
            if quantity > 0:
                days_count = 0
            else:
                days_count += 1
            days_since_list.append(days_count)
        
        df.loc[(df['Site_No'] == site) & (df['Item_No'] == item), 'days_since_last_sale'] = days_since_list
    
    return df, feature_cols

def add_holiday_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add Indonesia-specific holiday features to the DataFrame.
    
    Features added:
    - IsHoliday: Boolean indicating if the date is a holiday
    - HolidayName: Normalized holiday label (e.g., "Christmas" instead of "Christmas 2025")
    - DaysToNextHoliday: Number of days to the next known holiday
    - DaysSinceLastHoliday: Number of days since the last known holiday
    
    Args:
        df: DataFrame with Date column
        
    Returns:
        Tuple of (DataFrame with holiday features, List of feature column names)
    """
    df = df.copy()
    feature_cols = ['IsHoliday', 'HolidayName', 'DaysToNextHoliday', 'DaysSinceLastHoliday']
    
    # Get the first and last year from the dataset
    min_year = df['Date'].dt.year.min()
    max_year = df['Date'].dt.year.max()
    
    # Try to use the holidays package
    try:
        from holidays import Indonesia
        
        # Create a dictionary of Indonesian holidays for the relevant years
        indonesian_holidays = Indonesia(years=range(min_year, max_year + 1))
        
        # Create a list of all holidays in the range
        holiday_dates = []
        holiday_names = {}
        
        for date, name in indonesian_holidays.items():
            # Convert to pandas Timestamp for consistent handling
            date_ts = pd.Timestamp(date)
            holiday_dates.append(date_ts)
            
            # Normalize holiday names by removing the year part
            normalized_name = name.split(' ')
            # If the last part is a year, remove it
            if normalized_name[-1].isdigit():
                normalized_name = ' '.join(normalized_name[:-1])
            else:
                normalized_name = name
                
            holiday_names[date_ts] = normalized_name
        
        # Add IsHoliday and HolidayName columns - use vectorized operations for speed
        df['IsHoliday'] = df['Date'].isin(holiday_dates)
        
        # Create a lookup dictionary for faster mapping
        date_to_name = {pd.Timestamp(date): name for date, name in holiday_names.items()}
        df['HolidayName'] = df['Date'].map(date_to_name).fillna('not_holiday')
        
    except ImportError:
        # Make the holidays package a strict requirement
        raise ImportError(
            "The 'holidays' package is required for adding holiday features. "
            "Please install it using 'pip install holidays'."
        )
    
    # Sort all holiday dates for calculating days to/from holidays
    sorted_holiday_dates = sorted(holiday_dates)
    
    # Create dictionaries to map dates to days-to-next and days-since-last holiday
    next_holiday_map = {}
    prev_holiday_map = {}
    
    # Get unique dates in the dataframe to avoid redundant calculations
    unique_dates = df['Date'].unique()
    
    # Calculate for each unique date
    for date in unique_dates:
        # Find next holiday
        next_holidays = [h for h in sorted_holiday_dates if h > date]
        next_holiday_map[date] = (min(next_holidays) - date).days if next_holidays else 0
        
        # Find previous holiday  
        prev_holidays = [h for h in sorted_holiday_dates if h < date]
        prev_holiday_map[date] = (date - max(prev_holidays)).days if prev_holidays else 0
    
    # Map the dictionaries to create features
    df['DaysToNextHoliday'] = df['Date'].map(next_holiday_map)
    df['DaysSinceLastHoliday'] = df['Date'].map(prev_holiday_map)
    
    return df, feature_cols

def add_calendar_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add calendar-related features to the DataFrame.
    
    Features added:
    - Month: numeric month (1-12)
    - Week: ISO week number
    - Year: the year of data
    - IsPaydayWindow: Boolean - True if date falls within 25th to 1st (inclusive)
    
    Args:
        df: DataFrame with Date column
        
    Returns:
        Tuple of (DataFrame with calendar features, List of feature column names)
    """
    df = df.copy()
    feature_cols = ['Month', 'Week', 'Year', 'IsPaydayWindow']
    
    # Extract month, week, and year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Year'] = df['Date'].dt.year
    
    # Create IsPaydayWindow feature (True if day of month is between 25 and last day, or day 1)
    df['IsPaydayWindow'] = ((df['Date'].dt.day >= 25) | (df['Date'].dt.day == 1))
    
    return df, feature_cols

def add_promo_feature(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create isPromo feature to capture unusually high sales activity.
    
    For each (Site_No, Item_No) group:
    - Calculate Q3 (75th percentile) of daily Quantity
    - For each day, if Quantity > Q3, mark as promo (1), otherwise 0
    
    Args:
        df: DataFrame with Date, Site_No, Item_No, and Quantity columns
        
    Returns:
        Tuple of (DataFrame with isPromo feature, List of feature column names)
    """
    df = df.copy()
    feature_cols = ['isPromo']
    
    # Calculate Q3 for each (Site_No, Item_No) group
    q3_by_group = df.groupby(['Site_No', 'Item_No'])['Quantity'].quantile(0.75)
    
    # Map each record to its group's Q3 value
    df['q3_threshold'] = df.set_index(['Site_No', 'Item_No']).index.map(q3_by_group)
    
    # Create isPromo feature
    df['isPromo'] = (df['Quantity'] > df['q3_threshold']).astype(int)
    
    # Drop temporary column
    df = df.drop(columns=['q3_threshold'])
    
    return df, feature_cols

def classify_data_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Classify columns into numerical and categorical types.
    
    Args:
        df: DataFrame to classify columns from
        
    Returns:
        Tuple of (List of numerical columns, List of categorical columns)
    """
    # Find numeric columns (float and int), exclude dates and IDs
    numeric_cols = []
    
    for col in df.columns:
        if col not in DATE_FIELDS and col not in ID_COLUMNS:
            try:
                # Check if column can be converted to float
                pd.to_numeric(df[col]).dtype
                numeric_cols.append(col)
            except:
                pass
    
    # Correct vendor_no if mistakenly classified as numeric
    if 'Vendor_No' in numeric_cols:
        numeric_cols.remove('Vendor_No')
    
    # Add specific generated feature columns that should be numeric
    for col in df.columns:
        if col not in numeric_cols and col not in ID_COLUMNS and (
            col.startswith(('lag_', 'rolling_', 'days_since_', 'DaysTo', 'DaysSince')) or
            col == 'Quantity'
        ):
            numeric_cols.append(col)
    
    # All other columns are categorical, explicitly including Site_No and Item_No
    categorical_cols = [col for col in df.columns if col not in numeric_cols and col not in DATE_FIELDS]
    
    # Explicitly add time-based features as categorical
    for col in TIME_CATEGORICALS:
        if col in df.columns:
            if col in numeric_cols:
                numeric_cols.remove(col)
            if col not in categorical_cols:
                categorical_cols.append(col)
    
    # Make absolutely sure identifier columns are in categorical columns
    for id_col in ID_COLUMNS:
        if id_col in df.columns and id_col not in categorical_cols:
            categorical_cols.append(id_col)
    
    # Explicitly exclude date fields from both lists
    for date_field in DATE_FIELDS:
        if date_field in numeric_cols:
            numeric_cols.remove(date_field)
        if date_field in categorical_cols:
            categorical_cols.remove(date_field)
    
    # Log the classifications for visibility and debugging
    print(f"\nNumerical columns ({len(numeric_cols)}):")
    print(numeric_cols)
    print(f"\nCategorical columns ({len(categorical_cols)}):")
    print(categorical_cols)
    
    return numeric_cols, categorical_cols