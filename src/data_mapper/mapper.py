"""
Core functions for creating and managing data mappers
"""

import pandas as pd
import os
from datetime import datetime
from typing import Optional, List, Dict, Tuple

from config.config import (
    DATA_MAPPER_PATH, MAPPER_REQUIRED_COLS, MAPPER_OPTIONAL_COLS
)
from .utils import backup_mapper

def create_mapper(
    source_data: pd.DataFrame,
    output_path: str = DATA_MAPPER_PATH,
    additional_cols: Optional[List[str]] = None,
    overwrite: bool = False,
    add_metadata: bool = True
) -> pd.DataFrame:
    """
    Create a data mapper from source data
    
    Args:
        source_data: DataFrame containing at least Item_No and Site_No columns
        output_path: Path to save the mapper
        additional_cols: Additional columns to include in the mapper
        overwrite: Whether to overwrite existing mapper
        add_metadata: Whether to add metadata columns (Active, Last_Updated)
    
    Returns:
        DataFrame containing the mapper
    """
    print(f"Creating data mapper from source with {len(source_data)} records")
    
    # Check if mapper already exists
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Mapper already exists at {output_path}. Use update_mapper() instead or set overwrite=True.")
    
    # Validate required columns
    for col in MAPPER_REQUIRED_COLS:
        if col not in source_data.columns:
            raise ValueError(f"Source data must contain column: {col}")
    
    # Determine columns to include
    include_cols = list(MAPPER_REQUIRED_COLS)
    
    if additional_cols:
        # Add user-specified additional columns if they exist in source data
        for col in additional_cols:
            if col in source_data.columns and col not in include_cols:
                include_cols.append(col)
    
    # Add optional columns that exist in source data
    for col in MAPPER_OPTIONAL_COLS:
        if col in source_data.columns and col not in include_cols:
            include_cols.append(col)
    
    # Extract unique combinations
    mapper_df = source_data[include_cols].drop_duplicates()
    
    # Add metadata if requested
    if add_metadata:
        if 'Active' not in mapper_df.columns:
            mapper_df['Active'] = True
        if 'Last_Updated' not in mapper_df.columns:
            mapper_df['Last_Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mapper_df.to_csv(output_path, index=False)
    
    print(f"Created mapper with {len(mapper_df)} unique Item-Site combinations")
    print(f"Saved mapper to {output_path}")
    
    return mapper_df

def update_mapper(
    new_data: pd.DataFrame,
    mapper_path: str = DATA_MAPPER_PATH,
    mode: str = 'merge',
    backup: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Update an existing data mapper with new data
    
    Args:
        new_data: DataFrame with new data
        mapper_path: Path to existing mapper
        mode: Update mode ('merge', 'replace', 'add_only', 'deactivate_missing')
        backup: Whether to create a backup of the existing mapper
    
    Returns:
        Tuple of (Updated mapper DataFrame, Stats dictionary)
    """
    print(f"Updating data mapper at {mapper_path}")
    
    # Check if mapper exists
    if not os.path.exists(mapper_path):
        print(f"Warning: Mapper not found at {mapper_path}. Creating new mapper.")
        return create_mapper(new_data, mapper_path), {'created': len(new_data)}
    
    # Load existing mapper
    existing_mapper = pd.read_csv(mapper_path)
    
    # Create backup if requested
    if backup:
        backup_path = backup_mapper(mapper_path)
        print(f"Backup created at {backup_path}")
    
    # Process based on mode
    stats = {'before': len(existing_mapper)}
    
    if mode == 'replace':
        # Replace entire mapper with new data
        updated_mapper = create_mapper(
            new_data, 
            output_path=mapper_path, 
            overwrite=True
        )
        stats['added'] = len(updated_mapper)
        stats['removed'] = len(existing_mapper)
        stats['after'] = len(updated_mapper)
    
    elif mode == 'add_only':
        # Only add new combinations without removing existing ones
        updated_mapper = pd.concat([
            existing_mapper,
            new_data[MAPPER_REQUIRED_COLS].drop_duplicates()
        ]).drop_duplicates(MAPPER_REQUIRED_COLS)
        
        stats['added'] = len(updated_mapper) - len(existing_mapper)
        stats['removed'] = 0
        stats['after'] = len(updated_mapper)
        
        # Add timestamp for new entries
        if 'Last_Updated' in updated_mapper.columns:
            # Mark only new entries with current timestamp
            merged = existing_mapper.merge(
                updated_mapper[MAPPER_REQUIRED_COLS],
                how='right',
                indicator=True
            )
            new_indices = merged[merged['_merge'] == 'right_only'].index
            updated_mapper.loc[new_indices, 'Last_Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        updated_mapper.to_csv(mapper_path, index=False)
    
    elif mode == 'deactivate_missing':
        # Mark items not in new data as inactive rather than removing them
        if 'Active' not in existing_mapper.columns:
            existing_mapper['Active'] = True
        
        # Find items in existing mapper that are not in new data
        missing_in_new = existing_mapper.merge(
            new_data[MAPPER_REQUIRED_COLS],
            how='left',
            indicator=True
        )
        
        missing_indices = missing_in_new[missing_in_new['_merge'] == 'left_only'].index
        existing_mapper.loc[missing_indices, 'Active'] = False
        
        # Update Last_Updated for changed rows
        if 'Last_Updated' in existing_mapper.columns:
            existing_mapper.loc[missing_indices, 'Last_Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Add new entries that don't exist yet
        new_only = new_data.merge(
            existing_mapper[MAPPER_REQUIRED_COLS],
            how='left',
            indicator=True
        )
        
        new_entries = new_only[new_only['_merge'] == 'left_only'].drop(columns=['_merge'])
        
        if not new_entries.empty:
            if 'Active' in new_entries.columns:
                new_entries['Active'] = True
            else:
                new_entries['Active'] = True
                
            if 'Last_Updated' in new_entries.columns:
                new_entries['Last_Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            elif 'Last_Updated' in existing_mapper.columns:
                new_entries['Last_Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            updated_mapper = pd.concat([existing_mapper, new_entries])
        else:
            updated_mapper = existing_mapper
            
        stats['deactivated'] = len(missing_indices)
        stats['added'] = len(new_entries) if 'new_entries' in locals() else 0
        stats['after'] = len(updated_mapper)
        
        updated_mapper.to_csv(mapper_path, index=False)
    
    else:  # Default: 'merge'
        # Merge existing with new, preferring new data for overlaps
        relevant_cols = [col for col in MAPPER_REQUIRED_COLS + MAPPER_OPTIONAL_COLS 
                        if col in existing_mapper.columns or col in new_data.columns]
        
        # Extract unique required columns from new data
        new_pairs = new_data[MAPPER_REQUIRED_COLS].drop_duplicates()
        
        # Find existing entries to keep (those not in new data)
        existing_to_keep = existing_mapper.merge(
            new_pairs, on=MAPPER_REQUIRED_COLS, how='left', indicator=True
        )
        existing_to_keep = existing_to_keep[existing_to_keep['_merge'] == 'left_only'].drop(columns=['_merge'])
        
        # Prepare new data with consistent columns
        new_processed = new_data[
            [col for col in relevant_cols if col in new_data.columns]
        ].drop_duplicates(MAPPER_REQUIRED_COLS)
        
        if 'Active' in relevant_cols and 'Active' not in new_processed.columns:
            new_processed['Active'] = True
            
        if 'Last_Updated' in relevant_cols and 'Last_Updated' not in new_processed.columns:
            new_processed['Last_Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Combine kept existing entries with new entries
        updated_mapper = pd.concat([existing_to_keep, new_processed])
        
        # Ensure we have all expected columns
        for col in MAPPER_REQUIRED_COLS:
            if col not in updated_mapper.columns:
                raise ValueError(f"Required column {col} missing after merge")
        
        stats['kept'] = len(existing_to_keep)
        stats['replaced'] = len(existing_mapper) - len(existing_to_keep)
        stats['added'] = len(new_processed) - stats['replaced']
        stats['after'] = len(updated_mapper)
        
        updated_mapper.to_csv(mapper_path, index=False)
    
    print(f"Mapper update summary: {stats}")
    return updated_mapper, stats

def load_mapper(
    mapper_path: str = DATA_MAPPER_PATH, 
    active_only: bool = True
) -> pd.DataFrame:
    """
    Load a data mapper from file
    
    Args:
        mapper_path: Path to mapper file
        active_only: Whether to return only active entries
    
    Returns:
        DataFrame containing the mapper
    """
    if not os.path.exists(mapper_path):
        raise FileNotFoundError(f"Mapper not found at {mapper_path}")
    
    mapper_df = pd.read_csv(mapper_path)
    
    if active_only and 'Active' in mapper_df.columns:
        active_mapper = mapper_df[mapper_df['Active'] == True]
        print(f"Loaded {len(active_mapper)} active entries from mapper (total: {len(mapper_df)})")
        return active_mapper
    
    print(f"Loaded mapper with {len(mapper_df)} entries")
    return mapper_df