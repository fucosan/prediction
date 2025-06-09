import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from dateutil.relativedelta import relativedelta


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
    # Convert Date to datetime if it's not already
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


def add_lag_features(df: pd.DataFrame, lag_periods: List[int]) -> Tuple[pd.DataFrame, List[str]]:
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


def add_rolling_window_features(df: pd.DataFrame, window_sizes: List[int], 
                               metrics: List[str] = ['mean', 'sum', 'std']) -> Tuple[pd.DataFrame, List[str]]:
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


def clean_initial_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial cleaning operations on the dataset:
    1. Drop specified columns if they exist
    2. Convert percentage columns from strings (e.g., '3%') to float values (e.g., 0.03)
    
    Args:
        df: Raw DataFrame with sales data
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Drop specified columns if they exist
    drop_cols = [
        'Total_Sales_Incl_PPN_IDR',
        'Selling_Price_Incl_PPN_IDR',
        'Periodic_Disc_Amount_incl_PPN_IDR',
        'Add_Periodic_Disc_Amount_IDR',
        'Amount_Discount_Tambahan_incl_PPN_IDR',
        'Vendor_No', 'Master_Brand_Name', 'Unit_of_Measure',
        'item_mitra_10', 'kode_item_m10', 'kode_item_m10(cons)',
        'item_mitra_10(cons)', 'kode_item_onda', 'mapping_nama_item_onda',
        'produksi', 'benchmark_new_item', 'dus', 'category'
    ]
    
    existing_cols = [col for col in drop_cols if col in df.columns]
    if existing_cols:
        df = df.drop(columns=existing_cols)
    
    # Convert percentage columns to proper float values
    pct_columns = [
        'Periodic_Disc_Percentage',
        'Periodic_Disc_CC_Percentage',
        'Discount_Tambahan_Percentage'
    ]
    
    for col in pct_columns:
        if col in df.columns:
            # Handle string percentages like '3%'
            if df[col].dtype == 'object':
                df[col] = df[col].replace('', '0%')  # Handle empty strings
                df[col] = df[col].str.rstrip('%').astype(float) / 100
            # Handle raw numeric values
            else:
                df[col] = df[col] / 100
    
    return df


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


def classify_data_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Separate DataFrame columns into numerical and categorical types.
    
    Args:
        df: DataFrame to classify
        
    Returns:
        Tuple of (numerical_columns, categorical_columns)
    """
    # Date fields to exclude from both numerical and categorical columns
    date_fields = ['Date', 'Start_Date', 'End_Date']
    
    # Basic datatypes that are always numeric
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    # Get columns with numeric datatypes
    numeric_cols = list(df.select_dtypes(include=numeric_dtypes).columns)
    
    # Remove Site_No and Item_No from numeric columns if they are there
    # This ensures they're treated as categorical even if they contain only numbers
    if 'Site_No' in numeric_cols:
        numeric_cols.remove('Site_No')
    if 'Item_No' in numeric_cols:
        numeric_cols.remove('Item_No')
    if 'Vendor_No' in numeric_cols:
        numeric_cols.remove('Vendor_No')
    
    # Add specific generated feature columns that should be numeric
    for col in df.columns:
        if col not in numeric_cols and col not in ['Site_No', 'Item_No', 'Vendor_No'] and (
            col.startswith(('lag_', 'rolling_', 'days_since_', 'DaysTo', 'DaysSince')) or
            col in ['Quantity', 'Month', 'Week', 'Year', 'isPromo']
        ):
            numeric_cols.append(col)
    
    # All other columns are categorical, explicitly including Site_No and Item_No
    categorical_cols = [col for col in df.columns if col not in numeric_cols]
    
    # Make absolutely sure identifier columns are in categorical columns
    for id_col in ['Site_No', 'Item_No', 'Vendor_No']:
        if id_col in df.columns and id_col not in categorical_cols:
            categorical_cols.append(id_col)
    
    # Explicitly exclude date fields from both lists as they should be used only for
    # filtering, grouping, and feature generation, not as direct model inputs
    for date_field in date_fields:
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



def prepare_for_training(df: pd.DataFrame, numerical_cols: List[str], categorical_cols: List[str], 
                         target_col: str = 'Quantity', scaler_type: str = 'robust',
                         encoding_type: str = 'onehot', model_type: str = 'linear') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare the processed dataset for model training:
    1. Normalize numerical features (except target_col)
    2. Encode categorical features (except Site_No and Item_No)
    3. Create feature matrix X and target variable y
    
    Args:
        df: Processed dataframe with all features
        numerical_cols: List of numerical columns
        categorical_cols: List of categorical columns
        target_col: Target column name (default: 'Quantity')
        scaler_type: Type of scaler to use ('minmax', 'standard', 'robust')
        encoding_type: Type of categorical encoding ('onehot', 'label')
        model_type: Type of model to prepare for ('linear', 'tree')
        
    Returns:
        Tuple of (Feature matrix X, Target variable y)
    """
    df = df.copy()
    
    # Separate target variable
    y = df[target_col].copy()
    
    # 1. Normalize numerical features (except target)
    # Remove target_col from normalization if it's in numerical_cols
    numerical_for_scaling = [col for col in numerical_cols if col != target_col]
    
    # Choose scaler based on parameter
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}. Use 'minmax', 'standard', or 'robust'.")
    
    # Scale numerical features if there are any
    if numerical_for_scaling:
        df[numerical_for_scaling] = scaler.fit_transform(df[numerical_for_scaling])
    
    # 2. Encode categorical features (except Site_No and Item_No)
    # Remove identifiers from encoding
    cat_for_encoding = [col for col in categorical_cols if col not in ['Site_No', 'Item_No']]
    
    # Process categorical variables based on model type and encoding preference
    if encoding_type == 'onehot' or model_type == 'linear':
        # One-hot encoding for categorical variables (preferable for linear models)
        encoded_df = pd.get_dummies(df[cat_for_encoding], drop_first=True)
        
        # Drop original categorical columns and add encoded ones
        df = df.drop(columns=cat_for_encoding)
        df = pd.concat([df, encoded_df], axis=1)
        
    elif encoding_type == 'label' or model_type == 'tree':
        # Label encoding for categorical variables (suitable for tree-based models)
        for col in cat_for_encoding:
            if df[col].dtype == 'object' or df[col].dtype == 'category':
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])
    
    # 3. Remove Site_No and Item_No from features
    X = df.drop(columns=['Site_No', 'Item_No', target_col], errors='ignore')
    
    # Check for any remaining NaNs and fill them
    if X.isna().any().any():
        print(f"Warning: NaN values found in {X.columns[X.isna().any()].tolist()}. Filling with zeros.")
        X = X.fillna(0)
    
    # Final check - remove any remaining categorical columns that might not be encoded
    object_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(object_cols) > 0:
        print(f"Warning: Non-encoded object columns found: {object_cols.tolist()}. Converting to category codes.")
        for col in object_cols:
            X[col] = X[col].astype('category').cat.codes
    
    return X, y


def split_time_series_data(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, 
                           test_months: int = 2, valid_months: Optional[int] = None,
                           date_col: str = 'End_Date') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                               pd.Series, pd.Series, pd.Series]:
    """
    Split time series data into training, validation, and test sets.
    Uses fixed time windows rather than proportions to prevent data leakage.
    
    Args:
        df: Original dataframe with date column
        X: Feature matrix
        y: Target variable
        test_months: Number of months at the end to use as test set (default: 2)
        valid_months: Number of months before test set to use as validation set (default: None)
                      If None, no validation set is created (only train and test)
        date_col: Name of date column to use for splitting
        
    Returns:
        Tuple of (X_train, X_valid, X_test, y_train, y_valid, y_test)
    """
    # Ensure the date column exists
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe")
        
    # Convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Get the last date in the dataset
    last_date = df[date_col].max()
    
    # Calculate the test cutoff date (last date minus test_months)
    test_cutoff_date = last_date - relativedelta(months=test_months)
    
    # Calculate the validation cutoff date if requested
    if valid_months is not None:
        valid_cutoff_date = test_cutoff_date - relativedelta(months=valid_months)
        # Create masks for each split
        train_mask = df[date_col] < valid_cutoff_date
        valid_mask = (df[date_col] >= valid_cutoff_date) & (df[date_col] < test_cutoff_date)
        test_mask = df[date_col] >= test_cutoff_date
        
        # Split based on masks
        X_train = X.loc[df[train_mask].index]
        X_valid = X.loc[df[valid_mask].index]
        X_test = X.loc[df[test_mask].index]
        
        y_train = y.loc[df[train_mask].index]
        y_valid = y.loc[df[valid_mask].index]
        y_test = y.loc[df[test_mask].index]
        
        # Print information about the split
        print(f"Training set: {len(X_train)} samples (data before {valid_cutoff_date.date()})")
        print(f"Validation set: {len(X_valid)} samples (data from {valid_cutoff_date.date()} to {test_cutoff_date.date()})")
        print(f"Test set: {len(X_test)} samples (last {test_months} months, from {test_cutoff_date.date()} to {last_date.date()})")
    
    else:
        # No validation set, only train and test
        train_mask = df[date_col] < test_cutoff_date
        test_mask = df[date_col] >= test_cutoff_date
        
        # Split based on masks
        X_train = X.loc[df[train_mask].index]
        X_valid = pd.DataFrame()  # Empty DataFrame
        X_test = X.loc[df[test_mask].index]
        
        y_train = y.loc[df[train_mask].index]
        y_valid = pd.Series()  # Empty Series
        y_test = y.loc[df[test_mask].index]
        
        # Print information about the split
        print(f"Training set: {len(X_train)} samples (data before {test_cutoff_date.date()})")
        print(f"Test set: {len(X_test)} samples (last {test_months} months, from {test_cutoff_date.date()} to {last_date.date()})")
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def process_sales_data(df: pd.DataFrame, 
                       lag_periods: List[int] = [1, 2, 3, 7, 14],
                       window_sizes: List[int] = [3, 7, 14, 28],
                       rolling_metrics: List[str] = ['mean', 'sum', 'std'],
                       prepare_model_data: bool = False,
                       target_col: str = 'Quantity',
                       scaler_type: str = 'robust',
                       encoding_type: str = 'onehot',
                       model_type: str = 'linear') -> Union[
                           Tuple[pd.DataFrame, List[str], List[str], List[str]],
                           Tuple[pd.DataFrame, List[str], List[str], List[str], pd.DataFrame, pd.Series]
                       ]:
    """
    Main processing function that orchestrates the full workflow:
    1. Initial cleaning
    1b. Name normalization
    2. Normalize daily time series
    3. Add lag features
    4. Add rolling window features
    5. Add days since last sale
    6. Add holiday features
    7. Add calendar features
    8. Add promo feature
    9. Aggregate to biweekly periods
    10. Classify data types
    11. (Optional) Prepare data for modeling
    
    Args:
        df: DataFrame with at least Date, Site_No, Item_No, and Quantity columns
        lag_periods: List of lag periods to create features for
        window_sizes: List of window sizes for rolling calculations
        rolling_metrics: List of metrics to calculate for rolling windows
        prepare_model_data: Whether to prepare data for modeling
        target_col: Target column for modeling (default: Quantity)
        scaler_type: Type of scaler for numerical features
        encoding_type: Type of encoding for categorical features
        model_type: Type of model to prepare for
        
    Returns:
        If prepare_model_data is False:
            Tuple of (Processed DataFrame, Generated feature columns, Numerical columns, Categorical columns)
        If prepare_model_data is True:
            Tuple of (Processed DataFrame, Generated feature columns, Numerical columns, Categorical columns, X, y)
    """
    # Track all generated feature columns
    generated_feature_columns = []
    
    # Step 1: Initial cleaning
    df_cleaned = clean_initial_dataset(df)
    
    # Step 1b: Name normalization (new)
    if any(col in df_cleaned.columns for col in ['Item_Name', 'Site_Name', 'Vendor_Name']):
        df_cleaned = normalize_names(df_cleaned)
    
    # Step 2: Normalize daily time series
    df_normalized = normalize_daily_time_series(df_cleaned)
    
    # Step 3: Add lag features
    df_with_lags, lag_cols = add_lag_features(df_normalized, lag_periods)
    generated_feature_columns.extend(lag_cols)
    
    # Step 4: Add rolling window features
    df_with_rolling, rolling_cols = add_rolling_window_features(df_with_lags, window_sizes, rolling_metrics)
    generated_feature_columns.extend(rolling_cols)
    
    # Step 5: Add days since last sale
    df_enriched, days_since_cols = add_days_since_last_sale(df_with_rolling)
    generated_feature_columns.extend(days_since_cols)
    
    # Step 6: Add holiday features
    df_with_holidays, holiday_cols = add_holiday_features(df_enriched)
    generated_feature_columns.extend(holiday_cols)
    
    # Step 7: Add calendar features
    df_with_calendar, calendar_cols = add_calendar_features(df_with_holidays)
    generated_feature_columns.extend(calendar_cols)
    
    # Step 8: Add promo feature (new)
    df_with_promo, promo_cols = add_promo_feature(df_with_calendar)
    generated_feature_columns.extend(promo_cols)
    
    # Step 9: Aggregate to biweekly periods
    df_aggregated = aggregate_biweekly(df_with_promo)
    
    # Step 10: Classify data types
    numerical_columns, categorical_columns = classify_data_types(df_aggregated)
    
    # Step 11: Optionally prepare for modeling
    if prepare_model_data:
        X, y = prepare_for_training(
            df_aggregated,
            numerical_columns,
            categorical_columns,
            target_col=target_col,
            scaler_type=scaler_type,
            encoding_type=encoding_type,
            model_type=model_type
        )
        return df_aggregated, generated_feature_columns, numerical_columns, categorical_columns, X, y
    
    return df_aggregated, generated_feature_columns, numerical_columns, categorical_columns

def normalize_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize names for key identifier columns:
    - Map Item_No to the most common Item_Name
    - Map Site_No to the most common Site_Name
    - Map Vendor_No to the most common Vendor_Name (if Vendor_No is not dropped)
    
    Args:
        df: DataFrame with identifier columns and corresponding name columns
        
    Returns:
        DataFrame with normalized name columns
    """
    df = df.copy()
    
    # Define pairs of code-name columns to normalize
    column_pairs = [
        ('Item_No', 'Item_Name'),
        ('Site_No', 'Site_Name'),
        ('Vendor_No', 'Vendor_Name')
    ]
    
    for code_col, name_col in column_pairs:
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
