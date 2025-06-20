#!/usr/bin/env python3
"""
Incremental training pipeline for sales prediction system
"""

import os
import pandas as pd
import sys
from datetime import datetime
import argparse

# Import modules
from src.preprocess import load_data, preprocess_raw_data
from src.inc_train import (
    process_new_data, update_feature_store,
    update_model, load_existing_artifacts
)

# Import config settings
from config.config import (
    MODEL_FILE,
    PREPROCESS_DIR,
    FEATURE_STORE_PATH,
    INC_TRAIN_DIR,
    TARGET_COLUMN,
    EXTRA_BOOST_ROUNDS
)

# Ensure output directories exist
for directory in [PREPROCESS_DIR, INC_TRAIN_DIR]:
    os.makedirs(directory, exist_ok=True)

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Incremental training for sales prediction model')
    parser.add_argument('new_data_path', help='Path to new sales data (CSV or Excel)')
    parser.add_argument('--no-filter', action='store_true', 
                       help='Skip data filtering step (use if data is already filtered)')
    parser.add_argument('--rounds', type=int, default=EXTRA_BOOST_ROUNDS,
                       help=f'Number of boosting rounds to add (default: {EXTRA_BOOST_ROUNDS})')
    return parser.parse_args()

def main():
    """Main function for incremental training pipeline"""
    args = parse_arguments()
    
    start_time = datetime.now()
    print(f"=== Starting incremental training at {start_time} ===")
    
    # 1. Load existing artifacts
    print("\n1. Loading existing artifacts...")
    try:
        feature_metadata, feature_store = load_existing_artifacts()
        print(f"Loaded feature store with {len(feature_store)} records")
        numerical_cols = feature_metadata['numerical_columns']
        categorical_cols = feature_metadata['categorical_columns']
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        sys.exit(1)
    
    # 2. Process new data
    print(f"\n2. Processing new data from {args.new_data_path}...")
    try:
        processed_new_data, new_metadata = process_new_data(
            args.new_data_path, 
            prefilter=not args.no_filter
        )
        print(f"Processed {len(processed_new_data)} new records")
    except Exception as e:
        print(f"Error processing new data: {e}")
        sys.exit(1)
    
    # 3. Update feature store
    print("\n3. Updating feature store...")
    try:
        updated_feature_store = update_feature_store(processed_new_data)
        print(f"Feature store updated with {len(updated_feature_store)} total records")
    except Exception as e:
        print(f"Error updating feature store: {e}")
        sys.exit(1)
    
    # 4. Update model
    print("\n4. Updating model...")
    try:
        model, metrics = update_model(
            MODEL_FILE,
            updated_feature_store,
            numerical_cols,
            categorical_cols,
            target_column=TARGET_COLUMN,
            n_boost_rounds=args.rounds
        )
        
        print("\nUpdated model metrics:")
        for metric_name, value in metrics.items():
            print(f"  - {metric_name}: {value:.4f}")
    except Exception as e:
        print(f"Error updating model: {e}")
        sys.exit(1)
    
    end_time = datetime.now()
    print(f"\n=== Incremental training completed in {end_time - start_time} ===")

if __name__ == "__main__":
    main()