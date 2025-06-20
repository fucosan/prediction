#!/usr/bin/env python3
"""
Comparison tool for evaluating prediction performance against actual data
"""

import os
import sys
import argparse
from datetime import datetime

from src.compare import (
    load_prediction_data,
    load_actual_data,
    preprocess_data_for_comparison,
    compare_predictions,
    save_comparison_results,
    calculate_metrics,
    generate_metrics_report
)

# Import config settings
from config.config import (
    PREDICT_DIR,
    OUTPUT_DIR
)

COMPARE_DIR = os.path.join(OUTPUT_DIR, "compare")
os.makedirs(COMPARE_DIR, exist_ok=True)

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Compare predictions with actual data')
    
    parser.add_argument('prediction_file', 
                       help='Path to file containing predictions')
    parser.add_argument('actual_file',
                       help='Path to file containing actual data')
    
    parser.add_argument('--output', 
                       help='Base name for output files (without extension)')
    parser.add_argument('--format', choices=['csv', 'excel', 'parquet', 'all'], 
                       default='all', help='Output format for comparison file')
    
    parser.add_argument('--join-on', 
                       help='Comma-separated list of columns to join on (default: Site_No,Item_No,Date)',
                       default='Site_No,Item_No,Date')
    
    parser.add_argument('--group-by',
                       help='Comma-separated list of columns to group metrics by',
                       default='Site_No')
    
    parser.add_argument('--pred-col',
                       help='Column name for predictions (default: Predicted_Quantity)',
                       default='Predicted_Quantity')
    
    parser.add_argument('--actual-col',
                       help='Column name for actual values (default: Quantity)',
                       default='Quantity')
    
    parser.add_argument('--matched-only', action='store_true',
                       help='Include only records that match between predictions and actuals')
    
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip creating visualization plots')
    
    # Add new argument for handling prediction date format
    parser.add_argument('--pred-date-col', 
                       choices=['Date', 'Start_Date', 'End_Date'],
                       default='Date',
                       help='Column to use for date in predictions (default: Date)')
    
    return parser.parse_args()

def main():
    """Main function for comparison pipeline"""
    args = parse_arguments()
    
    start_time = datetime.now()
    print(f"=== Starting comparison at {start_time} ===")
    
    # Parse join columns and group-by columns
    join_columns = [col.strip() for col in args.join_on.split(',')]
    group_by_columns = [col.strip() for col in args.group_by.split(',')] if args.group_by else None
    
    # Detect if we need date mapping
    date_mapping_needed = ('Start_Date' in join_columns or 'End_Date' in join_columns) and args.pred_date_col != 'Date'
    date_column_in_pred = args.pred_date_col if date_mapping_needed else None
    
    try:
        # 1. Load data
        print("\n1. Loading data files...")
        
        # Load prediction data
        predictions = load_prediction_data(
            args.prediction_file, 
            required_columns=[args.pred_col] + join_columns
        )
        
        # Handle different date column names in actual data
        actual_join_cols = []
        for col in join_columns:
            if col == 'Start_Date' or col == 'End_Date':
                actual_join_cols.append('Date')
            else:
                actual_join_cols.append(col)
                
        print(f"Using join columns for predictions: {join_columns}")
        print(f"Using join columns for actuals: {actual_join_cols}")
        
        actuals = load_actual_data(
            args.actual_file,
            required_columns=[args.actual_col] + actual_join_cols
        )
        
        # 2. Preprocess data for comparison
        print("\n2. Preprocessing data...")
        
        # Handle date column mapping
        if date_mapping_needed:
            print(f"Mapping '{date_column_in_pred}' in predictions to 'Date' in actuals")
            # Create a copy of Date column with the name expected by comparison function
            if 'Date' in actuals.columns:
                for col in join_columns:
                    if col == 'Start_Date' or col == 'End_Date':
                        if col not in actuals.columns:
                            actuals[col] = actuals['Date']
        
        proc_predictions, proc_actuals = preprocess_data_for_comparison(predictions, actuals)
        
        # 3. Compare predictions with actuals
        print("\n3. Comparing predictions with actual data...")
        comparison = compare_predictions(
            proc_predictions,
            proc_actuals,
            join_columns=join_columns,
            prediction_column=args.pred_col,
            actual_column=args.actual_col,
            include_all=not args.matched_only
        )
        
        # 4. Save comparison results
        print("\n4. Saving comparison results...")
        output_base = args.output if args.output else os.path.join(
            COMPARE_DIR, f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        formats = ['csv', 'excel', 'parquet'] if args.format == 'all' else [args.format]
        output_files = save_comparison_results(comparison, output_base, COMPARE_DIR, formats)
        
        # 5. Calculate metrics
        print("\n5. Calculating performance metrics...")
        metrics = calculate_metrics(
            comparison,
            prediction_column=args.pred_col,
            actual_column=args.actual_col,
            group_by_columns=group_by_columns
        )
        
        # 6. Generate metrics report
        print("\n6. Generating metrics report...")
        metric_files = generate_metrics_report(
            metrics,
            comparison,
            output_dir=COMPARE_DIR,
            create_plots=not args.no_plots
        )
        
        print(f"\n=== Comparison completed in {datetime.now() - start_time} ===")
        print(f"Comparison results saved to: {', '.join(output_files.values())}")
        print(f"Metrics report: {metric_files['txt']}")
        if 'scatter_plot' in metric_files:
            print(f"Visualization plots saved to {os.path.dirname(metric_files['scatter_plot'])}")
    
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()