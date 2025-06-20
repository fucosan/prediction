"""
Model evaluation functions for sales prediction
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from typing import Dict, List, Tuple

from config.config import METRICS_FILE, EVALUATION_SUMMARY

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Evaluate model using RMSE and MAE.
    
    Args:
        model: Trained model
        X_test: Test feature matrix
        y_test: Test target vector
    
    Returns:
        Tuple of (predictions, metrics dictionary)
    """
    print("Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test.replace(0, np.nan))) * 100
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
    
    print(f"Model Evaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"  - {metric_name}: {value:.4f}")
    
    # Save metrics to file
    save_metrics(metrics, y_test, y_pred)
    
    return y_pred, metrics

def save_metrics(metrics: Dict[str, float], y_test: pd.Series, y_pred: np.ndarray) -> None:
    """
    Save metrics to file.
    
    Args:
        metrics: Dictionary of metric names and values
        y_test: Test target vector
        y_pred: Predicted values
    """
    # Save metrics as CSV
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    metrics_df.to_csv(METRICS_FILE, index=False)
    print(f"Metrics saved to {METRICS_FILE}")
    
    # Save detailed evaluation summary
    os.makedirs(os.path.dirname(EVALUATION_SUMMARY), exist_ok=True)
    with open(EVALUATION_SUMMARY, 'w') as f:
        f.write("Sales Prediction Model - Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write("Performance Metrics:\n")
        for metric_name, value in metrics.items():
            f.write(f"  - {metric_name}: {value:.4f}\n")
        f.write("\nData Statistics:\n")
        f.write(f"  - Test samples: {len(y_test)}\n")
        f.write(f"  - Actual values range: [{y_test.min():.4f}, {y_test.max():.4f}]\n")
        f.write(f"  - Predicted values range: [{y_pred.min():.4f}, {y_pred.max():.4f}]\n")
        f.write(f"  - Actual mean: {y_test.mean():.4f}\n")
        f.write(f"  - Predicted mean: {y_pred.mean():.4f}\n")
    
    print(f"Evaluation summary saved to {EVALUATION_SUMMARY}")