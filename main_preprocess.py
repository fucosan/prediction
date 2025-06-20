#!/usr/bin/env python3
"""
Main script for data preprocessing in the sales prediction pipeline

Usage: 
    python main_preprocess.py <raw_data_path> [--output <output_file>]

Example:
    python main_preprocess.py data/raw_sales.csv --output processed_sales.parquet
"""

import os
import sys
import argparse
from datetime import datetime
import pandas as pd

# Import preprocessing modules
from src.preprocess import process_sales_data, preprocess_raw_data
from src.preprocess.data_loader import load_data
from config.config import PREPROCESS_DIR, FEATURE_STORE_PATH

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Preprocess sales data for prediction model')
    parser.add_argument('raw_data_path', type=str, help='Path to raw data file (CSV or Excel)')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output file name (default: feature_store.parquet)')
    return parser.parse_args()

def main():
    """Main function for preprocessing pipeline"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Define output path
    output_path = os.path.join(PREPROCESS_DIR, 'feature_store.parquet') 
    if args.output:
        output_path = os.path.join(PREPROCESS_DIR, args.output)
    
    print(f"Starting preprocessing pipeline at {datetime.now()}")
    print(f"Raw data path: {args.raw_data_path}")
    print(f"Output will be saved to: {output_path}")
    
    try:
        # Step 1: Filter raw data based on data_mapper.csv
        filtered_data_path = preprocess_raw_data(args.raw_data_path)
        
        # Step 2: Load filtered data
        data = load_data(filtered_data_path, parse_dates=['Date'])
        print(f"Loaded {len(data)} rows from filtered data")
        
        # Step 3: Process data through the full pipeline
        processed_data, feature_metadata = process_sales_data(data, save_metadata=True)
        
        # Step 4: Save processed data
        processed_data.to_parquet(output_path, index=False)
        print(f"Saved processed data to {output_path}")
        
        # Also save a CSV version for inspection
        csv_path = output_path.replace('.parquet', '.csv')
        processed_data.to_csv(csv_path, index=False)
        print(f"Saved CSV version to {csv_path}")
        
        # Print summary
        print("\nPreprocessing Summary:")
        print(f"Input rows: {len(data)}")
        print(f"Output rows: {len(processed_data)}")
        print(f"Output columns: {len(processed_data.columns)}")
        print(f"Numerical features: {len(feature_metadata.get('numerical_columns', []))}")
        print(f"Categorical features: {len(feature_metadata.get('categorical_columns', []))}")
        print(f"Generated features: {sum(len(v) for k, v in feature_metadata.items() if k not in ['numerical_columns', 'categorical_columns'])}")
        print(f"Preprocessing completed successfully at {datetime.now()}")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())