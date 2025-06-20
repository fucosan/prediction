"""
Preprocessing module for sales prediction system
"""

import pandas as pd
import os
import json
import joblib
from typing import List, Tuple, Dict, Optional, Union
from datetime import datetime

# Import all preprocessing functions
from .cleaning import clean_initial_dataset, normalize_names, normalize_daily_time_series
from .feature_engineering import (
    add_lag_features, add_rolling_window_features, add_days_since_last_sale,
    add_holiday_features, add_calendar_features, add_promo_feature,
    classify_data_types
)
from .aggregation import aggregate_biweekly
from .data_loader import preprocess_raw_data, load_data

from config.config import (
    PREPROCESS_DIR, LAG_PERIODS, WINDOW_SIZES, ROLLING_METRICS, 
)

def process_sales_data(
    df: pd.DataFrame,
    lag_periods: List[int] = LAG_PERIODS,
    window_sizes: List[int] = WINDOW_SIZES,
    rolling_metrics: List[str] = ROLLING_METRICS,
    save_metadata: bool = True
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Main processing function that orchestrates the full preprocessing workflow.
    
    Args:
        df: DataFrame with sales data
        lag_periods: List of lag periods for feature creation
        window_sizes: List of window sizes for rolling features
        rolling_metrics: Metrics to calculate for rolling windows
        save_metadata: Whether to save metadata files
    
    Returns:
        Tuple of (Processed DataFrame, Dictionary of feature metadata)
    """
    print("Starting sales data preprocessing...")
    # Track features and execution time
    start_time = datetime.now()
    feature_metadata = {}
    all_generated_features = []
    
    # Step 1: Initial cleaning
    print("Step 1: Cleaning initial dataset...")
    df_cleaned = clean_initial_dataset(df)
    
    # Step 1b: Name normalization
    if any(col in df_cleaned.columns for col in ['Item_Name', 'Site_Name', 'Vendor_Name']):
        print("Step 1b: Normalizing names...")
        df_cleaned = normalize_names(df_cleaned)
    
    # Step 2: Normalize daily time series
    print("Step 2: Normalizing daily time series...")
    df_normalized = normalize_daily_time_series(df_cleaned)
    
    # Step 3: Add lag features
    print(f"Step 3: Adding lag features (periods: {lag_periods})...")
    df_with_lags, lag_cols = add_lag_features(df_normalized, lag_periods)
    feature_metadata['lag_features'] = lag_cols
    all_generated_features.extend(lag_cols)
    
    # Step 4: Add rolling window features
    print(f"Step 4: Adding rolling window features (windows: {window_sizes}, metrics: {rolling_metrics})...")
    df_with_rolling, rolling_cols = add_rolling_window_features(
        df_with_lags, window_sizes, rolling_metrics)
    feature_metadata['rolling_features'] = rolling_cols
    all_generated_features.extend(rolling_cols)
    
    # Step 5: Add days since last sale
    print("Step 5: Adding days since last sale feature...")
    df_enriched, days_since_cols = add_days_since_last_sale(df_with_rolling)
    feature_metadata['days_since_features'] = days_since_cols
    all_generated_features.extend(days_since_cols)
    
    # Step 6: Add holiday features
    print("Step 6: Adding holiday features...")
    df_with_holidays, holiday_cols = add_holiday_features(df_enriched)
    feature_metadata['holiday_features'] = holiday_cols
    all_generated_features.extend(holiday_cols)
    
    # Step 7: Add calendar features
    print("Step 7: Adding calendar features...")
    df_with_calendar, calendar_cols = add_calendar_features(df_with_holidays)
    feature_metadata['calendar_features'] = calendar_cols
    all_generated_features.extend(calendar_cols)
    
    # Step 8: Add promo feature
    print("Step 8: Adding promotion features...")
    df_with_promo, promo_cols = add_promo_feature(df_with_calendar)
    feature_metadata['promo_features'] = promo_cols
    all_generated_features.extend(promo_cols)
    
    # Step 9: Aggregate to biweekly periods
    print("Step 9: Aggregating to biweekly periods...")
    df_aggregated = aggregate_biweekly(df_with_promo)
    
    # Step 10: Classify data types
    print("Step 10: Classifying data types...")
    numerical_columns, categorical_columns = classify_data_types(df_aggregated)
    feature_metadata['numerical_columns'] = numerical_columns
    feature_metadata['categorical_columns'] = categorical_columns
    
    # Save metadata if requested
    if save_metadata:
        print("Saving feature metadata...")
        metadata_path = os.path.join(PREPROCESS_DIR, "feature_metadata.json")
        with open(metadata_path, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_metadata = {
                k: [str(item) for item in v] if isinstance(v, list) else str(v)
                for k, v in feature_metadata.items()
            }
            json.dump(serializable_metadata, f, indent=2)
        
        # Save column lists for later use in model training
        joblib.dump(numerical_columns, os.path.join(PREPROCESS_DIR, 'numerical_cols.pkl'))
        joblib.dump(categorical_columns, os.path.join(PREPROCESS_DIR, 'categorical_cols.pkl'))
        joblib.dump(all_generated_features, os.path.join(PREPROCESS_DIR, 'all_features.pkl'))
    
    end_time = datetime.now()
    print(f"Preprocessing completed in {end_time - start_time}.")
    print(f"Generated {len(all_generated_features)} features.")
    print(f"Final dataset shape: {df_aggregated.shape}")
    
    return df_aggregated, feature_metadata