#!/usr/bin/env python3
"""
Main script for generating synthetic sales data
"""

import os
import sys
import argparse
from datetime import datetime
import pandas as pd

# Add project root to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gen_data import extend_data_range

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Generate synthetic sales data')
    
    parser.add_argument('source_file',
                       help='Path to the source data file (.xlsx or .csv)')
    
    parser.add_argument('--end-date',
                       help='Target end date (YYYY-MM-DD)',
                       default='2025-06-26')
    
    parser.add_argument('--output-start-date',
                       help='Start date for the output file (YYYY-MM-DD)',
                       default=None)
    
    parser.add_argument('--output-dir',
                       help='Directory to save the generated data',
                       default='output/gen_data')
    
    parser.add_argument('--date-column',
                       help='Name of the date column in the source data',
                       default='Date')
    
    return parser.parse_args()

def generate_synthetic_records(
    site_item_pairs: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    historical_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate synthetic records based on historical patterns
    WITHOUT generating the Quantity field (which will be predicted by the model)
    
    Args:
        site_item_pairs: DataFrame with Site_No and Item_No combinations
        start_date: Start date for generation
        end_date: End date for generation
        historical_data: Historical data for reference
        
    Returns:
        DataFrame with generated records (without Quantity)
    """
    # Create a date range
    date_range = pd.date_range(start=start_date, end=end_date)
    
    all_records = []
    
    # For each site-item pair
    for _, row in site_item_pairs.iterrows():
        site_no = row['Site_No']
        item_no = row['Item_No']
        
        # Filter historical data for this site-item pair
        hist_site_item = historical_data[
            (historical_data['Site_No'] == site_no) & 
            (historical_data['Item_No'] == item_no)
        ]
        
        if len(hist_site_item) == 0:
            continue
            
        # Get the most recent record for this site-item pair
        last_record = hist_site_item.iloc[-1].copy()
        
        # For each date in the range
        for date in date_range:
            # Create a new record
            new_record = {
                'Date': date,
                'Site_No': site_no,
                'Item_No': item_no
            }
            
            # Copy all non-Quantity fields from the last record
            for col, value in last_record.items():
                if col not in ['Date', 'Quantity', 'Site_No', 'Item_No']:
                    new_record[col] = value
            
            # Do NOT include Quantity - this is what the model will predict
            
            all_records.append(new_record)
    
    # Create a DataFrame from all generated records
    if all_records:
        return pd.DataFrame(all_records)
    else:
        return pd.DataFrame()

def main():
    """Main function for data generation"""
    args = parse_arguments()
    
    start_time = datetime.now()
    print(f"=== Starting data generation at {start_time} ===")
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if source file exists
    if not os.path.exists(args.source_file):
        print(f"Error: Source file {args.source_file} does not exist")
        return
    
    try:
        # Extend the data range
        output_path = extend_data_range(
            args.source_file,
            args.end_date,
            args.output_dir,
            args.date_column,
            args.output_start_date  # Pass the new parameter
        )
        
        print(f"\nGenerated data saved to: {output_path}")
        print(f"Total execution time: {datetime.now() - start_time}")
        
    except Exception as e:
        print(f"Error generating data: {e}")

if __name__ == "__main__":
    main()