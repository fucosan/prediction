"""
Functions for loading and filtering data
"""

import pandas as pd
import os
from typing import Tuple, Optional
from config.config import DATA_MAPPER_PATH, REFERENCE_DATA_PATH

def preprocess_raw_data(raw_data_path: str, output_file: str = 'filtered_data.csv') -> str:
    """
    Preprocess raw data (Excel or CSV) by filtering based on data_mapper.csv
    and return the path to the processed CSV file.
    
    Args:
        raw_data_path: Path to the raw data file (Excel or CSV)
        output_file: Name for the output file
        
    Returns:
        Path to the processed CSV file
    """
    print(f"Preprocessing raw data from {raw_data_path}")
    
    # Load the data_mapper.csv containing valid Item_No and Site_No pairs
    mapper = pd.read_csv(DATA_MAPPER_PATH)
    
    # Determine file format and load accordingly
    if raw_data_path.lower().endswith('.xlsx'):
        raw_data = pd.read_excel(raw_data_path)
    else:
        raw_data = pd.read_csv(raw_data_path)
    
    # Merge to keep only rows where (Item_No, Site_No) pairs exist in data_mapper.csv
    filtered_data = raw_data.merge(mapper, on=['Item_No', 'Site_No'], how='inner')
    
    # Create processed file path in the output directory
    from config.config import PREPROCESS_DIR
    output_path = os.path.join(PREPROCESS_DIR, output_file)
    
    # Save the filtered rows to the output file
    filtered_data.to_csv(output_path, index=False)
    
    print(f"Filtered data saved to {output_path}")
    print(f"Kept {len(filtered_data)}/{len(raw_data)} rows after filtering ({len(filtered_data)/len(raw_data)*100:.1f}%)")
    
    # Compare columns between data.csv and filtered data
    if os.path.exists(REFERENCE_DATA_PATH):
        data = pd.read_csv(REFERENCE_DATA_PATH)
        
        # Get columns in data.csv but not in filtered data
        data_only_cols = sorted(set(data.columns) - set(filtered_data.columns))
        # Get columns in filtered data but not in data.csv
        filtered_only_cols = sorted(set(filtered_data.columns) - set(data.columns))
        # Get common columns
        common_cols = sorted(set(data.columns) & set(filtered_data.columns))
        
        print("\nColumns in reference data but NOT in filtered data:")
        for col in data_only_cols:
            print(f"  - {col}")
        
        print("\nColumns in filtered data but NOT in reference data:")
        for col in filtered_only_cols:
            print(f"  - {col}")
        
        print(f"\nCommon columns: {len(common_cols)}/{len(data.columns)} columns match")
        
        # Warn if there are significant differences in columns
        if len(data_only_cols) > 0:
            print(f"WARNING: {len(data_only_cols)} columns are missing from the filtered data!")
    
    return output_path


def load_data(file_path: str, parse_dates: Optional[list] = None) -> pd.DataFrame:
    """
    Load data from CSV or Excel file
    
    Args:
        file_path: Path to the data file
        parse_dates: List of columns to parse as dates
    
    Returns:
        DataFrame with loaded data
    """
    # Set default date columns if not provided
    if parse_dates is None:
        parse_dates = ['Date']
    
    # Determine file format and load accordingly
    if file_path.lower().endswith('.xlsx'):
        data = pd.read_excel(file_path, parse_dates=parse_dates)
    else:
        data = pd.read_csv(file_path, parse_dates=parse_dates)
    
    return data