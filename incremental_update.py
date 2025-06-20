"""
Incremental update script for sales prediction model

Allows new data to be processed and used to update the model without retraining from scratch.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import sys
from datetime import datetime, timedelta
from typing import Tuple, List, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import the same processing functions to ensure consistency
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

# CONFIGURABLE PARAMETERS
MODEL_PATH = "xgboost_sales_model.json"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "encoder.pkl"
FEATURE_STORE_PATH = "feature_store.parquet"
LOG_PATH = "incremental_update_log.csv"
LOOKBACK_DAYS = 365  # ~12 months
EXTRA_BOOST_ROUNDS = 10
DATA_MAPPER_PATH = "data_mapper.csv"
REFERENCE_DATA_PATH = "data.csv"

def preprocess_raw_data(raw_data_path: str) -> str:
    """
    Preprocess raw data (Excel or CSV) by filtering based on data_mapper.csv
    and return the path to the processed CSV file.
    
    Args:
        raw_data_path: Path to the raw data file (Excel or CSV)
    
    Returns:
        Path to the processed CSV file
    """
    print(f"Preprocessing raw data from {raw_data_path}")
    
    # Load the data_mapper.csv containing valid Item_No and Site_No pairs
    mapper = pd.read_csv(DATA_MAPPER_PATH)
    
    # Determine file format and load accordingly
    if raw_data_path.lower().endswith('.xlsx'):
        new_data = pd.read_excel(raw_data_path)
    else:
        new_data = pd.read_csv(raw_data_path)
    
    # Merge to keep only rows where (Item_No, Site_No) pairs exist in data_mapper.csv
    filtered_data = new_data.merge(mapper, on=['Item_No', 'Site_No'], how='inner')
    
    # Create processed file path
    processed_file_path = 'new_raw_data.csv'
    
    # Save the filtered rows to new_raw_data.csv
    filtered_data.to_csv(processed_file_path, index=False)
    
    print(f"Filtered data saved to {processed_file_path}")
    
    # Compare columns between data.csv and new_raw_data.csv
    if os.path.exists(REFERENCE_DATA_PATH):
        data = pd.read_csv(REFERENCE_DATA_PATH)
        
        # Get columns in data.csv but not in new_raw_data.csv
        data_only_cols = sorted(set(data.columns) - set(filtered_data.columns))
        # Get columns in new_raw_data.csv but not in data.csv
        new_raw_only_cols = sorted(set(filtered_data.columns) - set(data.columns))
        # Get common columns
        common_cols = sorted(set(data.columns) & set(filtered_data.columns))
        
        print("\nColumns in reference data but NOT in new raw data:")
        for col in data_only_cols:
            print(f"  - {col}")
        
        print("\nColumns in new raw data but NOT in reference data:")
        for col in new_raw_only_cols:
            print(f"  - {col}")
        
        print(f"\nCommon columns: {len(common_cols)}/{len(data.columns)} columns match")
        
        # Warn if there are significant differences in columns
        if len(data_only_cols) > 0:
            print(f"WARNING: {len(data_only_cols)} columns are missing from the new data!")
    
    return processed_file_path

def load_latest_transactions(csv_path: str) -> pd.DataFrame:
    """Load new transaction data from CSV file."""
    print(f"Loading new transactions from {csv_path}")
    return pd.read_csv(csv_path, parse_dates=["Date"])

def load_feature_store() -> pd.DataFrame:
    """Load the existing feature store with historical data."""
    if os.path.exists(FEATURE_STORE_PATH):
        print(f"Loading feature store from {FEATURE_STORE_PATH}")
        return pd.read_parquet(FEATURE_STORE_PATH)
    else:
        print("No existing feature store found. Creating new one.")
        return pd.DataFrame()

def save_feature_store(df: pd.DataFrame) -> None:
    """Save the updated feature store."""
    # Fix data type issues before saving
    df_clean = df.copy()
    
    # Find columns with mixed types and convert to string
    print("Fixing data types before saving feature store...")
    problematic_columns = ['Periodic_Disc_Amount_incl_PPN_IDR']
    
    # Check all object columns for potential issues
    for col in df_clean.select_dtypes(include=['object']).columns:
        # Sample some values to check for mixed types
        sample_values = df_clean[col].dropna().head(100)
        types = set(type(val) for val in sample_values)
        if len(types) > 1:
            print(f"Column {col} has mixed types: {types}")
            problematic_columns.append(col)
    
    # Convert problematic columns to strings
    for col in set(problematic_columns):
        if col in df_clean.columns:
            print(f"Converting {col} to string to ensure compatibility")
            df_clean[col] = df_clean[col].astype(str)
    
    # Save to Parquet with fixed data types
    df_clean.to_parquet(FEATURE_STORE_PATH, index=False)
    print(f"Updated feature store saved to {FEATURE_STORE_PATH}")

def load_model_and_preprocessors() -> Tuple[xgb.Booster, object, Dict]:
    """Load the saved model, scaler, and encoders."""
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    
    print(f"Loaded model from {MODEL_PATH}")
    print(f"Loaded scaler from {SCALER_PATH}")
    print(f"Loaded encoders from {ENCODER_PATH}")
    
    return model, scaler, encoder

def save_model(model: xgb.Booster) -> None:
    """Save the updated model."""
    model.save_model(MODEL_PATH)
    print(f"Updated model saved to {MODEL_PATH}")

def append_log(log_row: Dict) -> None:
    """Append a row to the update log."""
    log_exists = os.path.exists(LOG_PATH)
    log_df = pd.DataFrame([log_row])
    log_df.to_csv(LOG_PATH, mode='a', header=not log_exists, index=False)
    print(f"Log entry added to {LOG_PATH}")

def preprocess_new_data(new_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same preprocessing steps as in the original pipeline."""
    print("Preprocessing new data...")
    
    # Apply steps 1-4
    new_df = clean_initial_dataset(new_df)
    new_df = normalize_names(new_df)
    new_df = normalize_daily_time_series(new_df)
    new_df, _ = add_lag_features(new_df, lag_periods=[1, 2, 3, 7, 14])
    new_df, _ = add_rolling_window_features(new_df, window_sizes=[3, 7, 14, 28], metrics=['mean', 'sum', 'std'])
    new_df, _ = add_days_since_last_sale(new_df)
    new_df, _ = add_holiday_features(new_df)
    new_df, _ = add_calendar_features(new_df)
    new_df, _ = add_promo_feature(new_df)
    
    # Aggregate to bi-weekly
    new_agg = aggregate_biweekly(new_df)
    
    return new_agg

def category_to_codes(df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    """Convert categorical columns to codes to match training approach."""
    df = df.copy()
    for col in categorical_columns:
        if col in df.columns:  # No exclusion - encode all categorical columns
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                print(f"Converting {col} to category codes")
                df[col] = df[col].astype('category').cat.codes
    return df

def main(raw_data_path: str) -> None:
    """Main function for incremental update."""
    # Preprocess the raw data first
    processed_data_path = preprocess_raw_data(raw_data_path)
    
    # 1. Load new transactions and feature store
    new_df = load_latest_transactions(processed_data_path)
    feature_store = load_feature_store()
    
    # 2. Preprocess new data (Steps 1-4)
    new_agg = preprocess_new_data(new_df)
    
    # 3. Only keep windows after the most recent End_Date in feature store
    if not feature_store.empty:
        last_end = feature_store["End_Date"].max()
        print(f"Filtering data after {last_end.date()}...")
        new_agg = new_agg[new_agg["Start_Date"] > last_end]
        if new_agg.empty:
            print("No new data after the last update. Exiting.")
            return
    else:
        last_end = None
    
    print(f"Adding {len(new_agg)} new data points to feature store")
    
    # 4. Join with rolling feature store (last ~12 months)
    min_date = new_agg["Start_Date"].min() - timedelta(days=LOOKBACK_DAYS)
    rolling_store = feature_store[feature_store["Start_Date"] >= min_date] if not feature_store.empty else pd.DataFrame()
    combined = pd.concat([rolling_store, new_agg], ignore_index=True)
    
    # 5. Load model, scaler, encoder and feature column information
    model, scaler, encoder = load_model_and_preprocessors()
    numerical_columns = joblib.load('numerical_cols.pkl')
    categorical_columns = joblib.load('categorical_cols.pkl')
    feature_columns = joblib.load('feature_cols.pkl')
    
    # 6. Prepare features (match format with training data)
    # Remove target from numerical columns if present
    if 'Quantity' in numerical_columns:
        numerical_columns = [col for col in numerical_columns if col != 'Quantity']
    
    # Prepare X and y exactly like in training phase
    print("Preparing features...")
    X_new = new_agg.drop(columns=['Quantity', 'Start_Date', 'End_Date'], errors='ignore')
    y_new = new_agg["Quantity"].copy()
    
    # Add missing columns with default values
    for col in numerical_columns:
        if col not in X_new.columns:
            print(f"Adding missing numerical column: {col}")
            X_new[col] = 0
    
    for col in categorical_columns:
        if col not in X_new.columns:
            print(f"Adding missing categorical column: {col}")
            X_new[col] = "UNKNOWN"
    
    # Convert categorical columns to category codes - matching train.py approach
    X_new = category_to_codes(X_new, categorical_columns)
    
    # 7. Transform features (using existing scalers/encoders)
    print("Transforming features...")
    X_new_scaled = X_new.copy()
    
    # FIXED: Apply scaling to all numerical columns at once instead of column by column
    try:
        X_numerical = X_new[numerical_columns].copy()
        X_numerical_scaled = scaler.transform(X_numerical)
        
        # Copy scaled values back to the dataframe
        for i, col in enumerate(numerical_columns):
            X_new_scaled[col] = X_numerical_scaled[:, i]
        print("Successfully scaled all numerical features")
    except Exception as e:
        print(f"Error during batch scaling: {str(e)}")
        print("Falling back to original values for numerical features")
    
    # 8. Continue training model
    print(f"Training with {len(y_new)} samples")
    
    # FIXED: Create a dataframe with the EXACT same feature names and order as the model expects
    # This is critical for XGBoost to continue training correctly
    # Get the feature names directly from the model
    feature_names = model.feature_names
    X_final = X_new_scaled[feature_names]
    dtrain = xgb.DMatrix(X_final, label=y_new)
    
    try:
        # Continue training
        print(f"Continuing model training for {EXTRA_BOOST_ROUNDS} boost rounds")
        params = {'objective': 'reg:squarederror'}  # Use same as original
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            xgb_model=model,
            num_boost_round=EXTRA_BOOST_ROUNDS
        )
        print("Model training successful")
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        print("Training new model from scratch instead")
        # Train new model as fallback
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1
        }
        model = xgb.train(params, dtrain, num_boost_round=100)
    
    # 9. Save updated model and feature store
    save_model(model)
    save_feature_store(combined)
    
    # 10. Evaluate and log
    y_pred = model.predict(dtrain)
    # Fix for older scikit-learn versions
    rmse = np.sqrt(mean_squared_error(y_new, y_pred))  # Changed this line
    mae = mean_absolute_error(y_new, y_pred)
    
    print(f"New data evaluation - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # Load previous log to compare drift
    drift_flag = False
    if os.path.exists(LOG_PATH):
        prev_log = pd.read_csv(LOG_PATH)
        if not prev_log.empty:
            prev_rmse = prev_log["RMSE"].iloc[-1]
            drift_flag = (rmse - prev_rmse) / prev_rmse > 0.10
            if drift_flag:
                print(f"WARNING: Performance drift detected! RMSE increased from {prev_rmse:.4f} to {rmse:.4f}")
    
    # Add log entry
    log_row = {
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows_added": len(new_agg),
        "RMSE": rmse,
        "MAE": mae,
        "performance_drift": drift_flag
    }
    append_log(log_row)
    
    print("Incremental update complete!")
    
    # Check for potential data type issues
    print("Checking for data type consistency...")
    for col in combined.columns:
        if combined[col].dtype == 'object':
            types = set(type(val) for val in combined[col].dropna().head(100))
            if len(types) > 1:
                print(f"WARNING: Column {col} has mixed types and may cause issues: {types}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python incremental_update.py <raw_data.xlsx or raw_data.csv>")
    else:
        main(sys.argv[1])