#!/usr/bin/env python3
"""
Main script for sales prediction system using centralized config
"""

import os
import pandas as pd
from datetime import datetime
import pickle

# Import modules from the new structure
from src.preprocess import process_sales_data, load_data
from src.train import (
    prepare_model_data, 
    train_xgboost_model,
    plot_actual_vs_predicted,
    plot_feature_importance
)
from src.train.evaluation import time_series_cv_evaluation

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
    
    # 4. Create feature matrices
    print("\n4. Creating feature matrices...")
    X, y, scaler, encoder = prepare_model_data(
        processed_data,
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
        target_column=TARGET_COLUMN
    )
    
    # 5. Evaluate with time series CV and train final model
    print("\n5. Evaluating with time series cross-validation...")
    model = train_xgboost_model(X, y)  # Initial model for parameters
    
    # This gives meaningful evaluation metrics while still using all data
    cv_metrics, fold_metrics = time_series_cv_evaluation(model, X, y, n_splits=5, output_dir=OUTPUT_DIR)
    
    # Train final model on all data
    print("\n6. Training final model on all data...")
    final_model = train_xgboost_model(X, y)
    
    # 7. Create visualizations
    print("\n7. Creating visualizations...")
    plot_actual_vs_predicted(y, final_model.predict(X))
    plot_feature_importance(final_model, X.columns)
    
    print(f"\n=== Pipeline completed in {datetime.now() - start_time} ===")

if __name__ == "__main__":
    main()  # This will now use the bi-weekly first approach automatically