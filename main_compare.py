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

# Import the verification module
from src.compare.verification import verify_missing_actuals, check_specific_record

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
    
    # Update this line
    parser.add_argument('--join-on', 
                       help='Comma-separated list of columns to join on (default: Site_No,Item_No,Start_Date,End_Date)',
                       default='Site_No,Item_No,Start_Date,End_Date')
    
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
    
    # Update this line
    parser.add_argument('--pred-date-col', 
                       choices=['Date', 'Start_Date', 'End_Date'],
                       default='End_Date',
                       help='Column to use for date in predictions (default: End_Date)')
    
    # Add option to skip aggregation
    parser.add_argument('--no-aggregate', action='store_true',
                       help='Skip aggregating actual data to prediction date ranges')
    
    # Add new arguments for verification
    parser.add_argument('--verify-record', action='store_true',
                       help='Verify a specific record')
    parser.add_argument('--site-no', type=int,
                       help='Site number to check')
    parser.add_argument('--item-no', type=int,
                       help='Item number to check')
    parser.add_argument('--start-date', 
                       help='Start date to check (YYYY-MM-DD)')
    parser.add_argument('--end-date',
                       help='End date to check (YYYY-MM-DD)')
    
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
    
    # Extract key columns (non-date columns) for aggregation
    key_columns = [col for col in join_columns if col not in ['Date', 'Start_Date', 'End_Date']]
    
    print("\nOutput will have columns ordered as: Date columns first (Start_Date, End_Date, Date), then Site_No, Item_No, "
          f"followed by {args.actual_col} and {args.pred_col} side-by-side, then error metrics\n")
    
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
            # If we're joining on Start_Date/End_Date but actuals have Date
            if (col == 'Start_Date' or col == 'End_Date') and args.no_aggregate:
                actual_join_cols.append('Date')
            else:
                actual_join_cols.append(col)
                
        print(f"Using join columns for predictions: {join_columns}")
        print(f"Using join columns for actuals: {actual_join_cols}")
        
        actuals = load_actual_data(
            args.actual_file,
            required_columns=[args.actual_col] + [col for col in actual_join_cols if col != 'Start_Date' and col != 'End_Date'] + ['Date']
        )
        
        # 2. Preprocess data for comparison
        print("\n2. Preprocessing data...")
        
        if args.no_aggregate:
            # Handle date column mapping without aggregation
            if date_mapping_needed:
                print(f"Mapping '{date_column_in_pred}' in predictions to 'Date' in actuals")
                # Create a copy of Date column with the name expected by comparison function
                if 'Date' in actuals.columns:
                    for col in join_columns:
                        if col == 'Start_Date' or col == 'End_Date':
                            if col not in actuals.columns:
                                actuals[col] = actuals['Date']
                                
            proc_predictions, proc_actuals = predictions.copy(), actuals.copy()
        else:
            # Preprocess with aggregation to handle date ranges
            proc_predictions, proc_actuals = preprocess_data_for_comparison(
                predictions, 
                actuals,
                key_columns=key_columns
            )
        
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
        
        # 4. Verifying prediction-only records...
        verification_results = verify_missing_actuals(
            predictions=predictions,
            actuals=actuals,
            comparison_df=comparison,
            key_columns=key_columns,
            date_columns=['Start_Date', 'End_Date'],
            output_dir=COMPARE_DIR
        )
        
        # Example of checking a specific record (uncomment to use)
        # if 'Site_No' in actuals.columns and 'Item_No' in actuals.columns:
        #    check_specific_record(actuals, site_no=10003, item_no=200017407, 
        #                         start_date='2025-05-30', end_date='2025-06-12')
        
        # 5. Save comparison results
        print("\n5. Saving comparison results...")
        output_base = args.output if args.output else os.path.join(
            COMPARE_DIR, f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        formats = ['csv', 'excel', 'parquet'] if args.format == 'all' else [args.format]
        output_files = save_comparison_results(comparison, output_base, COMPARE_DIR, formats)
        
        # 6. Calculate metrics
        print("\n6. Calculating performance metrics...")
        metrics = calculate_metrics(
            comparison,
            prediction_column=args.pred_col,
            actual_column=args.actual_col,
            group_by_columns=group_by_columns
        )
        
        # 7. Generate metrics report
        print("\n7. Generating metrics report...")
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

    if args.verify_record and args.site_no is not None and args.item_no is not None:
        print("\nChecking specific record:")
        specific_records = check_specific_record(
            actuals=actuals,
            site_no=args.site_no,
            item_no=args.item_no,
            start_date=args.start_date,
            end_date=args.end_date
        )
        if len(specific_records) > 0:
            print("\nMatching records:")
            print(specific_records[['Date', 'Site_No', 'Item_No', 'Quantity']].to_string())
        else:
            print("No matching records found.")

if __name__ == "__main__":
    main()