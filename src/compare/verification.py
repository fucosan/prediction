"""
Functions for verifying data and diagnosing matching issues
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import os
from datetime import datetime

def verify_missing_actuals(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    comparison_df: pd.DataFrame,
    key_columns: List[str] = ['Site_No', 'Item_No'],
    date_columns: List[str] = ['Start_Date', 'End_Date'],
    output_dir: str = 'output/compare',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Verify why some predictions don't have matching actual values
    
    Args:
        predictions: Original predictions DataFrame
        actuals: Original actuals DataFrame
        comparison_df: Merged comparison DataFrame
        key_columns: Key columns like Site_No and Item_No
        date_columns: Date range columns in predictions
        output_dir: Directory to save verification reports
        verbose: Whether to print detailed info
    
    Returns:
        Dictionary with verification results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get prediction-only records from comparison
    prediction_only = comparison_df[comparison_df['Status'] == 'prediction_only'].copy()
    
    if len(prediction_only) == 0:
        print("No prediction-only records found.")
        return {"prediction_only_count": 0}
    
    print(f"\nAnalyzing {len(prediction_only)} prediction-only records:")
    
    # Check for missing key combinations in actuals
    key_missing_count = 0
    date_range_missing_count = 0
    verification_results = []
    
    # Create a set of key tuples from actuals for faster lookup
    actual_keys = set()
    for _, row in actuals.iterrows():
        key_tuple = tuple(row[col] for col in key_columns if col in row)
        actual_keys.add(key_tuple)
    
    # For each prediction-only record, check why it's missing
    for _, row in prediction_only.iterrows():
        # Extract key values
        key_values = tuple(row[col] for col in key_columns if col in row)
        
        # Check if key exists in actuals
        key_exists = key_values in actual_keys
        
        # If keys exist, check date ranges
        if key_exists:
            # Check for date range overlap
            if all(col in date_columns for col in ['Start_Date', 'End_Date']) and 'Date' in actuals.columns:
                # Get the prediction date range
                start_date = pd.to_datetime(row['Start_Date'])
                end_date = pd.to_datetime(row['End_Date'])
                
                # Filter actuals for this key and check date range
                key_filter = pd.Series(True, index=actuals.index)
                for i, col in enumerate(key_columns):
                    if col in actuals.columns:
                        key_filter &= (actuals[col] == key_values[i])
                
                matching_actuals = actuals[key_filter].copy()  # Create explicit copy
                
                # Check if any dates fall within range
                if 'Date' in matching_actuals.columns:
                    matching_actuals.loc[:, 'Date'] = pd.to_datetime(matching_actuals['Date'])
                    dates_in_range = ((matching_actuals['Date'] >= start_date) & 
                                     (matching_actuals['Date'] <= end_date))
                    
                    has_dates_in_range = dates_in_range.any()
                    
                    if not has_dates_in_range:
                        date_range_missing_count += 1
                        reason = "No actuals in date range"
                        
                        # Get closest dates for context
                        if len(matching_actuals) > 0:
                            earliest = matching_actuals['Date'].min()
                            latest = matching_actuals['Date'].max()
                            date_context = f"Actual dates: {earliest} to {latest}"
                        else:
                            date_context = "No dates found for this key"
                    else:
                        # This shouldn't happen - if dates exist in range, it should have matched
                        reason = "Date matching error"
                        date_context = f"Found {dates_in_range.sum()} dates in range"
                else:
                    reason = "Date column missing"
                    date_context = "No 'Date' column in actuals"
            else:
                reason = "Date columns missing"
                date_context = "Missing required date columns"
        else:
            key_missing_count += 1
            reason = "Key not in actuals"
            date_context = "N/A"
        
        # Add to verification results
        result = {
            'key_exists': key_exists,
            'reason': reason,
            'date_context': date_context
        }
        
        # Add key values for reference
        for i, col in enumerate(key_columns):
            if col in row:
                result[col] = row[col]
                
        # Add date range
        for col in date_columns:
            if col in row:
                result[col] = row[col]
                
        verification_results.append(result)
    
    # Summarize results
    print(f"  - {key_missing_count} records have keys not found in actual data")
    print(f"  - {date_range_missing_count} records have keys in actuals but no dates in the range")
    
    # For verbose mode, show some examples
    if verbose and verification_results:
        print("\nSample verification results:")
        for i, result in enumerate(verification_results[:5]):  # Show first 5 examples
            print(f"\nRecord {i+1}:")
            for k, v in result.items():
                print(f"  {k}: {v}")
        
        if len(verification_results) > 5:
            print(f"\n... and {len(verification_results) - 5} more records")
    
    # Save verification results to CSV and Excel
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_path = os.path.join(output_dir, f"verification_{timestamp}")
    csv_path = f"{base_path}.csv"
    excel_path = f"{base_path}.xlsx"
    
    # Create DataFrame from results
    verification_df = pd.DataFrame(verification_results)
    
    # Save to CSV
    verification_df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed verification report to {csv_path}")
    
    # Save to Excel with better formatting
    try:
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            verification_df.to_excel(writer, sheet_name='Missing Records', index=False)
            
            # Get the xlsxwriter workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Missing Records']
            
            # Add a format for the header
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'border': 1
            })
            
            # Write the column headers with the defined format
            for col_num, value in enumerate(verification_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                
            # Set column widths
            for i, col in enumerate(verification_df.columns):
                max_len = max(
                    verification_df[col].astype(str).map(len).max(),
                    len(str(col))
                ) + 2
                worksheet.set_column(i, i, min(max_len, 30))
                
            # Add summary sheet
            summary_data = {
                'Category': ['Total prediction-only records', 
                             'Keys not in actuals', 
                             'No dates in range'],
                'Count': [len(verification_results), 
                          key_missing_count, 
                          date_range_missing_count]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
        print(f"Saved formatted verification report to {excel_path}")
    except Exception as e:
        print(f"Could not save Excel report: {str(e)}")
    
    # Return summary
    return {
        "prediction_only_count": len(prediction_only),
        "key_missing_count": key_missing_count,
        "date_range_missing_count": date_range_missing_count,
        "verification_files": {
            "csv": csv_path,
            "excel": excel_path
        }
    }

def check_specific_record(
    actuals: pd.DataFrame,
    site_no: int,
    item_no: int,
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """
    Check specific record in actual data
    
    Args:
        actuals: DataFrame with actual data
        site_no: Site number to check
        item_no: Item number to check
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        
    Returns:
        DataFrame with filtered results
    """
    # Filter by Site_No and Item_No
    filtered = actuals[
        (actuals['Site_No'] == site_no) & 
        (actuals['Item_No'] == item_no)
    ].copy()
    
    print(f"Found {len(filtered)} records for Site_No={site_no}, Item_No={item_no}")
    
    # Filter by date range if provided
    if 'Date' in filtered.columns and start_date and end_date:
        filtered['Date'] = pd.to_datetime(filtered['Date'])
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        date_filtered = filtered[
            (filtered['Date'] >= start_date) & 
            (filtered['Date'] <= end_date)
        ]
        
        print(f"  Found {len(date_filtered)} records within date range {start_date} to {end_date}")
        return date_filtered
    
    return filtered