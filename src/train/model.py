"""
Model training functions for sales prediction
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import pickle
from typing import Dict, Any, Optional

from config.config import XGBOOST_PARAMS, MODEL_FILE

def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: Optional[pd.DataFrame] = None,
    y_valid: Optional[pd.Series] = None,
    params: Dict[str, Any] = XGBOOST_PARAMS,
    save_model: bool = True
) -> xgb.XGBRegressor:
    """
    Train an XGBoost regression model.
    
    Args:
        X_train: Training feature matrix
        y_train: Training target vector
        X_valid: Validation feature matrix (optional)
        y_valid: Validation target vector (optional)
        params: XGBoost parameters dictionary
        save_model: Whether to save the model to disk
    
    Returns:
        Trained XGBoost model
    """
    print("Training XGBoost model...")
    
    # Initialize model with parameters
    model = xgb.XGBRegressor(**params)
    
    # Train with early stopping if validation data provided
    if X_valid is not None and y_valid is not None:
        print("Using early stopping with validation data")
        try:
            # Different XGBoost versions have different parameter names
            # First try callbacks approach (newer versions)
            try:
                from xgboost.callback import EarlyStopping
                callbacks = [EarlyStopping(rounds=10)]
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    callbacks=callbacks,
                    verbose=True
                )
            except (ImportError, TypeError):
                # Try older API with eval_metric
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric='rmse',
                    verbose=True
                )
        except TypeError as e:
            print(f"Warning: Could not use early stopping: {e}")
            print("Training without early stopping")
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    
    print("Model training completed")
    
    # Save model if requested
    if save_model:
        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        # Use pickle instead of save_model
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {MODEL_FILE}")
    
    return model

def load_model(model_path: str = MODEL_FILE) -> xgb.XGBRegressor:
    """
    Load a trained XGBoost model.
    
    Args:
        model_path: Path to the saved model file
    
    Returns:
        Loaded XGBoost model
    """
    print(f"Loading model from {model_path}...")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Use pickle instead of load_model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def save_model(model, filepath, encoder=None, scaler=None):
    """Save the model and preprocessing objects to disk"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")
    
    # Keep the rest of the function that saves encoder/scaler