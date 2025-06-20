#!/usr/bin/env python3
"""
Simple script to create a data mapper from an input file
"""

import pandas as pd
import os
import sys

from src.data_mapper import create_mapper
from config.config import DATA_MAPPER_PATH

def main():
    """Create a data mapper from data.csv"""
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'data.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found")
        sys.exit(1)
    
    output_file = DATA_MAPPER_PATH
    
    try:
        # Load the CSV file
        df = pd.read_csv(input_file)
        
        # Create mapper using our module
        create_mapper(
            df, 
            output_path=output_file,
            additional_cols=['Item_Name', 'Site_Name'] if 'Item_Name' in df.columns else None,
            overwrite=True
        )
        
        print(f"Data mapper created at {output_file}")
    
    except Exception as e:
        print(f"Error creating mapper: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()