"""
Sales prediction script for forecasting future quantities.

Usage: python predict.py future_data.csv [output_filename.csv]
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import sys
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

# Import processing functions
from processing import (
    clean_initial_dataset,
    normalize_names,
    normalize_daily_time_series,
    add_lag_features,
    add_rolling_window_features,
    add_days_since_last_sale,
    add_holiday_features,
    add_calendar_features,
    add_promo_feature,
    aggregate_biweekly
)

FEATURE_STORE_PATH = "feature_store.parquet"
MODEL_PATH = "xgboost_sales_model.json"
LOOKBACK_DAYS = 365  # Days of history to use for feature calculation
DATA_MAPPER_PATH = "data_mapper.csv"
REFERENCE_DATA_PATH = "data.csv"

def preprocess_future_data(future_data_path: str) -> str:
    """
    Preprocess future data (Excel or CSV) by filtering based on data_mapper.csv
    and return the path to the processed CSV file.
    
    Args:
        future_data_path: Path to the future data file (Excel or CSV)
    
    Returns:
        Path to the processed CSV file
    """
    print(f"Preprocessing future data from {future_data_path}")
    
    # Load the data_mapper.csv containing valid Item_No and Site_No pairs
    mapper = pd.read_csv(DATA_MAPPER_PATH)
    
    # Determine file format and load accordingly
    if future_data_path.lower().endswith('.xlsx'):
        future_data = pd.read_excel(future_data_path)
    else:
        future_data = pd.read_csv(future_data_path)
    
    # Merge to keep only rows where (Item_No, Site_No) pairs exist in data_mapper.csv
    filtered_data = future_data.merge(mapper, on=['Item_No', 'Site_No'], how='inner')
    
    # Create processed file path
    processed_file_path = 'filtered_future_data.csv'
    
    # Save the filtered rows to filtered_future_data.csv
    filtered_data.to_csv(processed_file_path, index=False)
    
    print(f"Filtered data saved to {processed_file_path}")
    print(f"Kept {len(filtered_data)}/{len(future_data)} rows after filtering ({len(filtered_data)/len(future_data)*100:.1f}%)")
    
    # Compare columns between data.csv and filtered future data
    if os.path.exists(REFERENCE_DATA_PATH):
        data = pd.read_csv(REFERENCE_DATA_PATH)
        
        # Get columns in data.csv but not in filtered data
        data_only_cols = sorted(set(data.columns) - set(filtered_data.columns))
        # Get columns in filtered data but not in data.csv
        future_only_cols = sorted(set(filtered_data.columns) - set(data.columns))
        # Get common columns
        common_cols = sorted(set(data.columns) & set(filtered_data.columns))
        
        print("\nColumns in reference data but NOT in future data:")
        for col in data_only_cols:
            print(f"  - {col}")
        
        print("\nColumns in future data but NOT in reference data:")
        for col in future_only_cols:
            print(f"  - {col}")
        
        print(f"\nCommon columns: {len(common_cols)}/{len(data.columns)} columns match")
        
        # Warn if there are significant differences in columns
        if len(data_only_cols) > 0:
            print(f"WARNING: {len(data_only_cols)} columns are missing from the future data!")
    
    return processed_file_path

def load_model_and_preprocessors():
    """Load the trained model, scaler and encoder."""
    try:
        model = xgb.Booster()
        model.load_model(MODEL_PATH)
        scaler = joblib.load('scaler.pkl')
        encoder = joblib.load('encoder.pkl')
        numerical_columns = joblib.load('numerical_cols.pkl')
        categorical_columns = joblib.load('categorical_cols.pkl')
        feature_columns = joblib.load('feature_cols.pkl')
        
        print("Model and preprocessors loaded successfully")
        return model, scaler, encoder, numerical_columns, categorical_columns, feature_columns
    except Exception as e:
        print(f"Error loading model or preprocessors: {e}")
        sys.exit(1)

def load_feature_store():
    """Load historical data for feature calculation"""
    try:
        feature_store = pd.read_parquet(FEATURE_STORE_PATH)
        # Convert date columns to datetime
        date_cols = ['Start_Date', 'End_Date']
        for col in date_cols:
            if col in feature_store.columns:
                feature_store[col] = pd.to_datetime(feature_store[col])
        return feature_store
    except Exception as e:
        print(f"Warning: Could not load feature store: {e}")
        return pd.DataFrame()

def process_future_data(future_data: pd.DataFrame, feature_store: pd.DataFrame):
    """Process future data for prediction, applying all preprocessing steps"""
    print("Processing future data for prediction...")
    
    # Combine with historical data for proper feature generation
    # This is crucial for time-series features like lag and rolling statistics
    combined_data = pd.concat([feature_store, future_data], ignore_index=True)
    
    # Apply preprocessing steps as in training
    combined_data = clean_initial_dataset(combined_data)
    combined_data = normalize_names(combined_data)
    combined_data = normalize_daily_time_series(combined_data)
    combined_data, _ = add_lag_features(combined_data, lag_periods=[1, 2, 3, 7, 14])
    combined_data, _ = add_rolling_window_features(combined_data, window_sizes=[3, 7, 14, 28], metrics=['mean', 'sum', 'std'])
    combined_data, _ = add_days_since_last_sale(combined_data)
    combined_data, _ = add_holiday_features(combined_data)
    combined_data, _ = add_calendar_features(combined_data)
    combined_data, _ = add_promo_feature(combined_data)
    
    # Aggregate to bi-weekly
    processed_data = aggregate_biweekly(combined_data)
    
    # Filter only future periods (those not in the feature store)
    future_dates = future_data['Date'].unique()
    min_date = min(future_dates)
    max_date = max(future_dates)
    
    # Get only periods containing future dates
    future_periods = processed_data[
        (processed_data['End_Date'] >= min_date) & 
        (processed_data['Start_Date'] <= max_date)
    ].copy()
    
    return future_periods

def prepare_features_for_prediction(df, scaler, numerical_columns, categorical_columns):
    """Prepare features for prediction by applying scaling and encoding"""
    df = df.copy()
    
    # Save original values before any encoding
    original_site_no = df['Site_No'].copy()
    original_item_no = df['Item_No'].copy()
    
    # 1. Scale numerical features
    if numerical_columns:
        X_numerical = df[numerical_columns].copy()
        try:
            X_numerical_scaled = scaler.transform(X_numerical)
            # Copy back scaled values
            for i, col in enumerate(numerical_columns):
                df[col] = X_numerical_scaled[:, i]
            print("Successfully scaled numerical features")
        except Exception as e:
            print(f"Error during scaling: {str(e)}")
            print("Using original values for numerical features")
    
    # 2. Convert categorical features to codes
    for col in categorical_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                print(f"Converting {col} to category codes")
                df[col] = df[col].astype('category').cat.codes
    
    # Also save row index for mapping back to original values
    df['_row_index'] = range(len(df))
    
    return df, original_site_no, original_item_no

def predict_future_sales(future_data_path: str, output_path: str = "predictions.csv"):
    """
    Main function to predict future sales quantities.
    
    Args:
        future_data_path: Path to CSV with future data points
        output_path: Output file path for predictions
    """
    # 0. Preprocess and filter the future data first
    processed_future_data_path = preprocess_future_data(future_data_path)
    
    # 1. Load model and preprocessors
    model, scaler, encoder, numerical_columns, categorical_columns, feature_columns = load_model_and_preprocessors()
    
    # 2. Load historical data
    feature_store = load_feature_store()
    if feature_store.empty:
        print("Warning: No historical data found. Time-series features may be inaccurate.")
    
    # 3. Load filtered future data
    print(f"Loading processed future data from {processed_future_data_path}")
    try:
        future_data = pd.read_csv(processed_future_data_path)
        future_data['Date'] = pd.to_datetime(future_data['Date'])
    except Exception as e:
        print(f"Error loading future data: {e}")
        sys.exit(1)
    
    # 4. Process future data
    future_periods = process_future_data(future_data, feature_store)
    
    # 5. Prepare features
    X_future, original_site_no, original_item_no = prepare_features_for_prediction(
        future_periods, scaler, numerical_columns, categorical_columns
    )
    
    # 6. Get features in exactly the same order as the model expects
    model_features = model.feature_names
    print(f"Preparing {len(model_features)} features for prediction")
    
    # Add any missing features
    for feat in model_features:
        if feat not in X_future.columns:
            print(f"Adding missing feature: {feat}")
            X_future[feat] = 0
            
    # Keep row index for mapping back
    row_indices = X_future['_row_index'].values
    
    # Create input with just the model features in correct order
    X_final = X_future[model_features]
    
    # 7. Generate predictions
    print("Making predictions...")
    dmatrix = xgb.DMatrix(X_final)
    predictions = model.predict(dmatrix)
    
    # 8. Create output with predictions mapped back to original data
    results_df = pd.DataFrame({
        '_row_index': row_indices,
        'Predicted_Quantity': predictions
    })
    
    # Create output with original identifiers
    output_df = pd.DataFrame({
        'Start_Date': future_periods['Start_Date'].values,
        'End_Date': future_periods['End_Date'].values,
        'Site_No': original_site_no.values,
        'Item_No': original_item_no.values,
        '_row_index': range(len(future_periods))
    })
    
    # Merge predictions with original data
    final_output = output_df.merge(results_df, on='_row_index')
    
    # Drop internal index column
    final_output = final_output.drop(columns=['_row_index'])
    
    # Round predictions to reasonable values
    final_output['Predicted_Quantity'] = final_output['Predicted_Quantity'].round(2)
    
    # 9. Save output
    final_output.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    print(f"Generated {len(final_output)} predictions")
    
    # 10. Summary statistics
    print("\nPrediction Summary:")
    print(f"Min quantity: {final_output['Predicted_Quantity'].min():.2f}")
    print(f"Max quantity: {final_output['Predicted_Quantity'].max():.2f}")
    print(f"Avg quantity: {final_output['Predicted_Quantity'].mean():.2f}")
    
    # Sample predictions
    print("\nSample predictions:")
    print(final_output.head().to_string())
    return final_output

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <future_data.csv> [output_file.csv]")
        sys.exit(1)
        
    future_data_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "predictions.csv"
    
    predict_future_sales(future_data_path, output_path)