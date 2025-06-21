"""
Functions for making sales predictions
"""

import pickle
import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict

from config.config import MODEL_FILE
import xgboost as xgb

def predict_sales(
    X: pd.DataFrame,
    model_path: str = MODEL_FILE
) -> np.ndarray:
    """
    Make sales predictions using the trained model
    
    Args:
        X: Feature matrix for prediction
        model_path: Path to the saved model
    
    Returns:
        Array of predictions
    """
    print(f"Loading model from {model_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Use pickle to load the model instead of XGBoost's native method
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
    
    print("Making predictions...")
    predictions = model.predict(X)
    
    # Ensure predictions are non-negative
    predictions = np.maximum(predictions, 0)
    
    print(f"Generated {len(predictions)} predictions")
    print(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    return predictions

def apply_business_rules(
    predictions: np.ndarray, 
    metadata: pd.DataFrame,
    rules: Optional[Dict] = None
) -> np.ndarray:
    """
    Apply business rules to adjust predictions if needed
    
    Args:
        predictions: Raw model predictions
        metadata: Metadata DataFrame with Site_No, Item_No, etc.
        rules: Optional dictionary of business rules
    
    Returns:
        Adjusted predictions
    """
    adjusted_predictions = predictions.copy()
    
    if rules is None:
        # Default rules
        rules = {
            "min_prediction": 0,  # Minimum prediction value
            "round_to_integer": True,  # Round predictions to integers
            "special_sites": {}  # Special rules for specific sites
        }
    
    print("Applying business rules to predictions...")
    
    # Apply minimum prediction value
    adjusted_predictions = np.maximum(adjusted_predictions, rules["min_prediction"])
    
    # Apply site-specific adjustments if any
    if "special_sites" in rules and rules["special_sites"]:
        for site_no, adjustment in rules["special_sites"].items():
            site_mask = metadata['Site_No'] == site_no
            if adjustment['type'] == 'multiply':
                adjusted_predictions[site_mask] *= adjustment['value']
            elif adjustment['type'] == 'add':
                adjusted_predictions[site_mask] += adjustment['value']
    
    # Round to integers if specified
    if rules.get("round_to_integer", True):
        adjusted_predictions = np.round(adjusted_predictions).astype(int)
    
    return adjusted_predictions