#!/usr/bin/env python3
"""
Prediction pipeline for sales prediction system
"""

import os
import pandas as pd
import sys
import argparse
from datetime import datetime

# Import modules
from src.predict import (
    prepare_prediction_data,
    predict_sales,
    format_predictions,
    save_predictions,
)
from src.predict.output_formatter import create_prediction_summary
from src.predict.prediction import apply_business_rules
from src.inc_train.utils import load_existing_artifacts

# Import config settings
from config.config import (
    MODEL_FILE,
    PREDICT_DIR,
    SCALER_PATH,
    ENCODER_PATH
)

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Sales prediction system')
    parser.add_argument('input_file', help='Path to input data file')
    parser.add_argument('--output', help='Output file path base name')
    parser.add_argument('--format', choices=['csv', 'excel', 'parquet', 'all'], 
                        default='all', help='Output format')
    parser.add_argument('--confidence', action='store_true', 
                       help='Include confidence intervals in output')
    return parser.parse_args()

def main():
    """Main function for sales prediction pipeline"""
    args = parse_arguments()
    
    start_time = datetime.now()
    print(f"=== Starting prediction pipeline at {start_time} ===")
    
    # Ensure output directory exists
    os.makedirs(PREDICT_DIR, exist_ok=True)
    
    try:
        # 1. Load feature metadata from artifacts
        print("\n1. Loading existing artifacts...")
        feature_metadata, _ = load_existing_artifacts()
        numerical_cols = feature_metadata['numerical_columns']
        categorical_cols = feature_metadata['categorical_columns']
        
        # 2. Prepare data for prediction
        print(f"\n2. Preparing prediction data from {args.input_file}...")
        X, metadata, _ = prepare_prediction_data(
            args.input_file,
            numerical_cols,
            categorical_cols,
            scaler_path=SCALER_PATH,
            encoder_path=ENCODER_PATH
        )
        
        # 3. Make predictions
        print("\n3. Making sales predictions...")
        raw_predictions = predict_sales(X, model_path=MODEL_FILE)
        
        # 4. Apply business rules
        print("\n4. Applying business rules...")
        adjusted_predictions = apply_business_rules(raw_predictions, metadata)
        
        # 5. Format predictions
        print("\n5. Formatting prediction results...")
        predictions_df = format_predictions(
            adjusted_predictions, 
            metadata,
            include_confidence=args.confidence
        )
        
        # 6. Save predictions
        print("\n6. Saving prediction results...")
        output_base = args.output if args.output else os.path.join(
            PREDICT_DIR, f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        formats = ['csv', 'excel', 'parquet'] if args.format == 'all' else [args.format]
        output_files = save_predictions(predictions_df, output_base, formats)
        
        # 7. Create prediction summary
        print("\n7. Creating prediction summary...")
        summary_path = create_prediction_summary(
            predictions_df, 
            os.path.splitext(list(output_files.values())[0])[0] + "_summary.txt"
        )
        
        print(f"\n=== Prediction completed in {datetime.now() - start_time} ===")
        print(f"Prediction results saved to: {', '.join(output_files.values())}")
        print(f"Summary report: {summary_path}")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()