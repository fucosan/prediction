#!/usr/bin/env python3
"""
Main script for sales prediction system using centralized config
"""

import os
import pandas as pd
from datetime import datetime

# Import modules from the new structure
from src.preprocess import process_sales_data, load_data
from src.train import (
    prepare_model_data, 
    split_time_series_data,
    train_xgboost_model,
    evaluate_model,
    plot_actual_vs_predicted,
    plot_feature_importance
)

# Import config settings
from config.config import (
    REFERENCE_DATA_PATH,
    PREPROCESS_DIR,
    FEATURE_STORE_PATH,
    MODEL_DIR,
    OUTPUT_DIR,
    TARGET_COLUMN
)

# Ensure output directories exist
for directory in [PREPROCESS_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

def main():
    """Main function for sales prediction pipeline"""
    start_time = datetime.now()
    print(f"=== Starting sales prediction pipeline at {start_time} ===")
    
    # 1. Load data
    print(f"\n1. Loading data from {REFERENCE_DATA_PATH}...")
    data = load_data(REFERENCE_DATA_PATH)
    print(f"Loaded data shape: {data.shape}")
    
    # 2. Preprocess
    print("\n2. Processing data...")
    processed_data, feature_metadata = process_sales_data(data)
    
    # Save processed data to configured location
    print(f"Saving processed data to {FEATURE_STORE_PATH}...")
    processed_data.to_parquet(FEATURE_STORE_PATH, index=False)
    
    # Save CSV version for easy inspection
    csv_path = os.path.join(PREPROCESS_DIR, "processed_data.csv")
    processed_data.to_csv(csv_path, index=False)
    
    # 3. Prepare data for modeling
    print("\n3. Preparing data for modeling...")
    numerical_cols = feature_metadata['numerical_columns']
    categorical_cols = feature_metadata['categorical_columns']
    
    X, y, scaler, encoder = prepare_model_data(
        processed_data,
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
        target_column=TARGET_COLUMN
    )
    
    # 4. Split data
    print("\n4. Creating train/validation/test splits...")
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_time_series_data(
        processed_data, X, y, use_all_for_train=True
    )
    
    # 5. Train model
    print("\n5. Training model...")
    model = train_xgboost_model(
        X_train, y_train, 
        X_valid=X_valid, 
        y_valid=y_valid
    )
    
    # 6. Evaluate model
    print("\n6. Evaluating model...")
    y_pred, metrics = evaluate_model(model, X_test, y_test)
    
    # 7. Create visualizations
    print("\n7. Creating visualizations...")
    plot_actual_vs_predicted(y_test, y_pred)
    plot_feature_importance(model, X.columns)
    
    print(f"\n=== Pipeline completed in {datetime.now() - start_time} ===")

if __name__ == "__main__":
    main()