"""
Model updating functions for incremental training
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from src.train import prepare_model_data, evaluate_model
from src.train.model import load_model
from config.config import MODEL_FILE, EXTRA_BOOST_ROUNDS, OUTPUT_DIR, INC_TRAIN_DIR

def update_model(
    model_path: str,
    feature_store: pd.DataFrame,
    numerical_columns: list,
    categorical_columns: list,
    target_column: str = 'Quantity',
    n_boost_rounds: int = EXTRA_BOOST_ROUNDS
) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
    """
    Update an existing model with new data
    
    Args:
        model_path: Path to the existing model
        feature_store: Updated feature store with new data
        numerical_columns: List of numerical columns
        categorical_columns: List of categorical columns
        target_column: Target column name
        n_boost_rounds: Number of additional boosting rounds
    
    Returns:
        Tuple of (Updated model, evaluation metrics)
    """
    print(f"Updating model at {model_path}")
    
    # Load existing model
    model = load_model(model_path)
    
    # Check if model exists
    if model is None:
        raise ValueError(f"Failed to load model from {model_path}")
    
    # Prepare new data for training
    X, y, _, _ = prepare_model_data(
        feature_store,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        target_column=target_column
    )
    
    print(f"Continuing training with {len(X)} samples and {n_boost_rounds} additional boosting rounds")
    
    # Continue training with new data
    model.fit(
        X, y,
        xgb_model=model_path,
        verbose=True
    )
    
    # Evaluate updated model
    y_pred, metrics = evaluate_model(model, X, y)
    
    # Save updated model
    model.save_model(model_path)
    print(f"Updated model saved to {model_path}")
    
    # Log the update details
    log_update(len(X), n_boost_rounds, metrics)
    
    return model, metrics

def log_update(n_samples: int, n_rounds: int, metrics: Dict[str, float]) -> None:
    """
    Log details of the incremental update
    
    Args:
        n_samples: Number of samples used for update
        n_rounds: Number of boosting rounds added
        metrics: Evaluation metrics after update
    """
    os.makedirs(INC_TRAIN_DIR, exist_ok=True)
    
    # Create a log entry
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': n_samples,
        'n_rounds': n_rounds,
        'metrics': metrics
    }
    
    # Log file path
    log_file = os.path.join(INC_TRAIN_DIR, 'update_log.json')
    
    # Load existing log if it exists
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                log_data = {'updates': []}
    else:
        log_data = {'updates': []}
    
    # Add new entry
    log_data['updates'].append(log_entry)
    
    # Write updated log
    with open(log_file, 'w') as f:
        # Convert any non-serializable objects to strings
        json.dump(log_data, f, indent=2, default=str)
    
    print(f"Update logged to {log_file}")