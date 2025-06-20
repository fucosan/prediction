"""
Utility functions for data mapper management
"""

import pandas as pd
import os
import shutil
from datetime import datetime
import glob
from typing import List, Dict, Optional

from config.config import DATA_MAPPER_PATH, DATA_MAPPER_BACKUP_DIR

def backup_mapper(
    mapper_path: str = DATA_MAPPER_PATH,
    backup_dir: str = DATA_MAPPER_BACKUP_DIR
) -> str:
    """
    Create a backup of a mapper file
    
    Args:
        mapper_path: Path to mapper file
        backup_dir: Directory to store backups
    
    Returns:
        Path to the backup file
    """
    if not os.path.exists(mapper_path):
        raise FileNotFoundError(f"Mapper not found at {mapper_path}")
    
    # Create backup directory if it doesn't exist
    os.makedirs(backup_dir, exist_ok=True)
    
    # Generate backup filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.basename(mapper_path)
    name_parts = os.path.splitext(filename)
    backup_filename = f"{name_parts[0]}_{timestamp}{name_parts[1]}"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    # Copy the file
    shutil.copy2(mapper_path, backup_path)
    
    return backup_path

def restore_mapper(
    backup_path: str,
    target_path: str = DATA_MAPPER_PATH,
    make_backup: bool = True
) -> str:
    """
    Restore a mapper from backup
    
    Args:
        backup_path: Path to backup file
        target_path: Path to restore to
        make_backup: Whether to backup current mapper before restoring
    
    Returns:
        Path to the restored mapper
    """
    if not os.path.exists(backup_path):
        raise FileNotFoundError(f"Backup not found at {backup_path}")
    
    # Backup current mapper if it exists
    if os.path.exists(target_path) and make_backup:
        backup_mapper(target_path)
    
    # Restore from backup
    shutil.copy2(backup_path, target_path)
    
    return target_path

def list_backups(backup_dir: str = DATA_MAPPER_BACKUP_DIR) -> List[Dict]:
    """
    List all available mapper backups
    
    Args:
        backup_dir: Directory containing backups
    
    Returns:
        List of dictionaries with backup information
    """
    if not os.path.exists(backup_dir):
        return []
    
    # Find all backup files
    backup_files = glob.glob(os.path.join(backup_dir, "*.csv"))
    
    # Extract metadata from filenames
    backups = []
    for file_path in backup_files:
        filename = os.path.basename(file_path)
        try:
            # Extract timestamp from filename
            parts = os.path.splitext(filename)[0].split('_')
            date_part = parts[-2]
            time_part = parts[-1]
            
            # Parse timestamp (assuming format is YYYYMMDD_HHMMSS)
            timestamp_str = f"{date_part}_{time_part}"
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            
            # Get file stats
            file_stats = os.stat(file_path)
            file_size = file_stats.st_size
            
            # Get row count
            df = pd.read_csv(file_path)
            row_count = len(df)
            
            backups.append({
                'filename': filename,
                'path': file_path,
                'timestamp': timestamp,
                'size_bytes': file_size,
                'row_count': row_count
            })
        except:
            # Skip files that don't match expected format
            continue
    
    # Sort by timestamp (newest first)
    backups.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return backups

def get_mapper_stats(mapper_path: str = DATA_MAPPER_PATH) -> Dict:
    """
    Get statistics for a mapper file
    
    Args:
        mapper_path: Path to mapper file
    
    Returns:
        Dictionary with mapper statistics
    """
    if not os.path.exists(mapper_path):
        return {'exists': False}
    
    # Load mapper
    mapper = pd.read_csv(mapper_path)
    
    # Base stats
    stats = {
        'exists': True,
        'total_entries': len(mapper),
        'unique_sites': len(mapper['Site_No'].unique()) if 'Site_No' in mapper.columns else 0,
        'unique_items': len(mapper['Item_No'].unique()) if 'Item_No' in mapper.columns else 0,
    }
    
    # Add stats for active/inactive if available
    if 'Active' in mapper.columns:
        active_count = mapper['Active'].sum()
        stats.update({
            'active_entries': active_count,
            'inactive_entries': len(mapper) - active_count,
            'active_percent': round(active_count / len(mapper) * 100, 2) if len(mapper) > 0 else 0
        })
    
    # Last update info
    if 'Last_Updated' in mapper.columns:
        try:
            latest_update = pd.to_datetime(mapper['Last_Updated']).max()
            earliest_update = pd.to_datetime(mapper['Last_Updated']).min()
            stats.update({
                'latest_update': latest_update.strftime('%Y-%m-%d %H:%M:%S'),
                'earliest_update': earliest_update.strftime('%Y-%m-%d %H:%M:%S'),
            })
        except:
            pass
    
    # File stats
    file_stats = os.stat(mapper_path)
    stats.update({
        'file_size_bytes': file_stats.st_size,
        'file_created': datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
        'file_modified': datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
    })
    
    return stats