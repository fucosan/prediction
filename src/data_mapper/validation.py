"""
Functions for validating and filtering data using the mapper
"""

import pandas as pd
from typing import Dict, Tuple, Optional

from config.config import DATA_MAPPER_PATH, MAPPER_REQUIRED_COLS
from .mapper import load_mapper

def filter_by_mapper(
    data: pd.DataFrame,
    mapper_path: str = DATA_MAPPER_PATH,
    active_only: bool = True,
    return_stats: bool = False
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict]:
    """
    Filter data to include only rows that match entries in the mapper
    
    Args:
        data: DataFrame to filter
        mapper_path: Path to mapper file
        active_only: Whether to use only active entries in mapper
        return_stats: Whether to return filtering statistics
    
    Returns:
        Filtered DataFrame or tuple of (Filtered DataFrame, Stats dictionary)
    """
    print(f"Filtering data using mapper at {mapper_path}")
    
    # Check if data has required columns
    for col in MAPPER_REQUIRED_COLS:
        if col not in data.columns:
            raise ValueError(f"Data must contain mapper key column: {col}")
    
    # Load mapper
    mapper = load_mapper(mapper_path, active_only=active_only)
    
    # Perform filtering
    before_count = len(data)
    filtered_data = data.merge(mapper[MAPPER_REQUIRED_COLS], on=MAPPER_REQUIRED_COLS, how='inner')
    after_count = len(filtered_data)
    
    # Calculate stats
    stats = {
        'before': before_count,
        'after': after_count,
        'removed': before_count - after_count,
        'removed_percent': round((before_count - after_count) / before_count * 100, 2) if before_count > 0 else 0
    }
    
    print(f"Filtering stats: {stats}")
    
    if return_stats:
        return filtered_data, stats
    return filtered_data

def validate_data_against_mapper(
    data: pd.DataFrame,
    mapper_path: str = DATA_MAPPER_PATH,
    active_only: bool = True
) -> Dict:
    """
    Validate data against a mapper without filtering
    
    Args:
        data: DataFrame to validate
        mapper_path: Path to mapper file
        active_only: Whether to use only active entries in mapper
    
    Returns:
        Dictionary with validation results
    """
    print(f"Validating data against mapper at {mapper_path}")
    
    # Load mapper
    mapper = load_mapper(mapper_path, active_only=active_only)
    
    # Create a set of valid combinations from mapper for faster lookup
    valid_pairs = set(
        tuple(row) for row in 
        mapper[MAPPER_REQUIRED_COLS].itertuples(index=False, name=None)
    )
    
    # Extract combinations from data
    data_pairs = set(
        tuple(row) for row in 
        data[MAPPER_REQUIRED_COLS].itertuples(index=False, name=None)
    )
    
    # Find valid and invalid entries
    valid_entries = data_pairs.intersection(valid_pairs)
    invalid_entries = data_pairs.difference(valid_pairs)
    
    # Calculate statistics
    total_data_rows = len(data)
    total_unique_pairs = len(data_pairs)
    
    # Count occurrences of each invalid pair
    invalid_counts = {}
    if invalid_entries:
        for pair in invalid_entries:
            pair_dict = dict(zip(MAPPER_REQUIRED_COLS, pair))
            mask = True
            for col, val in pair_dict.items():
                mask = mask & (data[col] == val)
            count = mask.sum()
            invalid_counts[pair] = count
    
    # Prepare result
    result = {
        'total_rows': total_data_rows,
        'unique_combinations': total_unique_pairs,
        'valid_combinations': len(valid_entries),
        'invalid_combinations': len(invalid_entries),
        'valid_percent': round(len(valid_entries) / total_unique_pairs * 100, 2) if total_unique_pairs > 0 else 100,
        'invalid_entries': invalid_counts
    }
    
    print(f"Validation results: {len(valid_entries)} valid combinations, {len(invalid_entries)} invalid combinations")
    return result