"""
Functions for generating synthetic data to extend existing datasets
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
import os
from pathlib import Path
import argparse

def extend_data_range(
    source_data_path: str,
    end_date: str,
    output_dir: str = "output/gen_data",
    date_column: str = "Date",
    output_start_date: str = None,  # New parameter for filtering output
) -> str:
    """
    Extend an existing dataset with generated data up to the specified end date,
    and optionally filter to a specific date range.
    
    Args:
        source_data_path: Path to the source data file (.xlsx or .csv)
        end_date: Target end date for generation (YYYY-MM-DD)
        output_dir: Directory to save the output file
        date_column: Name of the date column in the source data
        output_start_date: Optional start date for the output file (YYYY-MM-DD)
        
    Returns:
        Path to the generated file
    """
    print(f"Loading source data from {source_data_path}...")
    
    # Load the source data based on file extension
    try:
        if source_data_path.lower().endswith('.xlsx'):
            df = pd.read_excel(source_data_path, parse_dates=[date_column])
        elif source_data_path.lower().endswith('.csv'):
            df = pd.read_csv(source_data_path, parse_dates=[date_column])
        else:
            raise ValueError("Source file must be either .xlsx or .csv")
    except Exception as e:
        raise ValueError(f"Error loading source file: {e}")

    print(f"Loaded {len(df)} records from source file")
    
    # Print column information to help with debugging
    print(f"Source data columns: {df.columns.tolist()}")
    print(f"Numeric columns: {df.select_dtypes(include=['number']).columns.tolist()}")
    
    # Convert end_date to datetime
    target_end_date = pd.to_datetime(end_date)
    
    # Find the max date in the source data
    max_source_date = df[date_column].max()
    print(f"Source data date range: {df[date_column].min()} to {max_source_date}")
    
    # If the target end date is already covered, no need to generate new data
    if max_source_date >= target_end_date:
        print(f"Source data already extends to {max_source_date}, which is beyond target end date {target_end_date}")
        # We still continue to be able to filter by date range if requested
    else:
        # Calculate the date range to generate
        start_date = max_source_date + timedelta(days=1)
        days_to_generate = (target_end_date - start_date).days + 1
        print(f"Generating {days_to_generate} days of data from {start_date} to {target_end_date}")
        
        if days_to_generate <= 0:
            print("No new data to generate")
        else:
            # Get unique site-item combinations from the source data
            site_item_pairs = df[['Site_No', 'Item_No']].drop_duplicates()
            print(f"Found {len(site_item_pairs)} unique Site_No/Item_No combinations")
            
            # Generate new records
            try:
                new_records = generate_synthetic_records(
                    site_item_pairs, 
                    start_date, 
                    target_end_date, 
                    df
                )
                print(f"Generated {len(new_records)} new records")
                
                # Verify there's no Quantity in the generated data
                if 'Quantity' in new_records.columns:
                    print("WARNING: Removing Quantity column from generated data")
                    new_records = new_records.drop('Quantity', axis=1)
    
                if len(new_records) == 0:
                    print("Warning: No new records were generated, please check your data")
                else:
                    # Combine with original data
                    df = pd.concat([df, new_records], ignore_index=True)
                    print(f"Combined dataset has {len(df)} records")
            except Exception as e:
                import traceback
                print(f"Error generating records: {str(e)}")
                print(traceback.format_exc())
                raise
    
    # Filter the data to the requested output date range if specified
    if output_start_date:
        output_start = pd.to_datetime(output_start_date)
        print(f"Filtering output data to range: {output_start} to {target_end_date}")
        
        # Create a filtered dataset with the requested date range
        mask = (df[date_column] >= output_start) & (df[date_column] <= target_end_date)
        filtered_df = df[mask].copy()
        
        print(f"Filtered dataset has {len(filtered_df)} records from {len(df)} total records")
        
        # Use the filtered data for output
        output_df = filtered_df
    else:
        # Use all data for output
        output_df = df
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique filename based on the timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Include date range in filename if filtering was applied
    if output_start_date:
        start_str = pd.to_datetime(output_start_date).strftime("%Y%m%d")
        end_str = pd.to_datetime(end_date).strftime("%Y%m%d")
        filename = f"data_{start_str}_to_{end_str}_{timestamp}.xlsx"
    else:
        filename = f"extended_data_{timestamp}.xlsx"
    
    output_path = os.path.join(output_dir, filename)
    
    # Save the output data
    try:
        output_df.to_excel(output_path, index=False)
        print(f"Data saved to {output_path}")
    except Exception as e:
        print(f"Error saving output file: {e}")
        raise
    
    return output_path

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
    print("Generating records WITHOUT Quantity field")
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
            for col in historical_data.columns:
                if col not in ['Date', 'Quantity', 'Site_No', 'Item_No'] and col in last_record:
                    new_record[col] = last_record[col]
            
            # Do NOT include Quantity - this is what the model will predict
            
            all_records.append(new_record)
    
    # Create a DataFrame from all generated records
    if all_records:
        df = pd.DataFrame(all_records)
        print(f"Generated {len(df)} records WITHOUT Quantity field")
        
        # Verify Quantity is not present
        if 'Quantity' in df.columns:
            print("ERROR: Quantity is still in the generated data!")
            # Remove it forcefully
            df = df.drop('Quantity', axis=1)
            print("Removed Quantity column from generated data")
        else:
            print("Confirmed: No Quantity field in generated data")
            
        return df
    else:
        return pd.DataFrame()

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Generate and/or filter synthetic sales data')
    
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
        # Extend the data range and filter if needed
        output_path = extend_data_range(
            args.source_file,
            args.end_date,
            args.output_dir,
            args.date_column,
            args.output_start_date
        )
        
        print(f"\nData saved to: {output_path}")
        print(f"Total execution time: {datetime.now() - start_time}")
        
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    main()