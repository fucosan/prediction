"""
Utility functions for incremental training
"""

import os
import joblib
import pandas as pd
import json
from typing import Dict, Tuple, Any, Optional

from config.config import (
    MODEL_FILE, SCALER_PATH, ENCODER_PATH, FEATURE_STORE_PATH,
    PREPROCESS_DIR
)

def load_existing_artifacts() -> Tuple[Dict[str, list], pd.DataFrame]:
    """
    Load existing artifacts needed for incremental training
    
    Returns:
        Tuple of (feature metadata, feature store)
    """
    # Load feature metadata
    metadata_path = os.path.join(PREPROCESS_DIR, "feature_metadata.json")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Feature metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        feature_metadata = json.load(f)
    
    # Convert string arrays back to lists
    for key, value in feature_metadata.items():
        if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            # Convert string representation of list back to actual list
            feature_metadata[key] = [x.strip() for x in value[1:-1].split(',')]
    
    # Load feature store
    if not os.path.exists(FEATURE_STORE_PATH):
        raise FileNotFoundError(f"Feature store not found: {FEATURE_STORE_PATH}")
    
    feature_store = pd.read_parquet(FEATURE_STORE_PATH)
    
    return feature_metadata, feature_store

def check_artifacts_exist() -> bool:
    """
    Check if all necessary artifacts exist for incremental training
    
    Returns:
        True if all artifacts exist, False otherwise
    """
    required_files = [MODEL_FILE, SCALER_PATH, ENCODER_PATH, FEATURE_STORE_PATH]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Missing required file: {file_path}")
            return False
    
    return True