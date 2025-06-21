"""
Model evaluation functions for sales prediction
"""

from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

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
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Handle division by zero by excluding zeros in y_test
    mask = y_test != 0
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    
    # Calculate total error and percentage
    total_error = np.sum(y_pred - y_test)
    total_error_pct = (total_error / np.sum(y_test)) * 100 if np.sum(y_test) > 0 else 0
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Total_Error': total_error,
        'Total_Error_Pct': total_error_pct
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

def time_series_cv_evaluation(model, X, y, n_splits=5, gap=0, plot=True, output_dir=None):
    """
    Evaluate time series model using cross-validation while still using all data for final model
    
    Args:
        model: Model instance with fit/predict methods
        X: Feature data
        y: Target data
        n_splits: Number of cross-validation splits
        gap: Optional gap between train and test sets
        plot: Whether to create a visualization of the CV splits
        output_dir: Directory to save the plot (optional)
    
    Returns:
        Dictionary with cross-validation metrics
    """
    print("Performing time series cross-validation...")
    
    # Initialize time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    
    # Store metrics from each fold
    cv_metrics = {
        'MAE': [], 
        'RMSE': [], 
        'MAPE': []
    }
    
    # Store indices for visualization
    train_indices = []
    test_indices = []
    
    # Perform CV
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # Store indices for plotting
        train_indices.append(train_idx)
        test_indices.append(test_idx)
        
        # Create train/test split for this fold
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]
        
        # Train a copy of the model
        model_copy = model.__class__(**model.get_params())
        model_copy.fit(X_train_fold, y_train_fold)
        
        # Evaluate on test fold
        y_pred_fold = model_copy.predict(X_test_fold)
        fold_metrics = calculate_regression_metrics(y_test_fold, y_pred_fold)
        
        # Store metrics
        for key in cv_metrics.keys():
            cv_metrics[key].append(fold_metrics[key])
            
        # Print fold results
        print(f"  Fold {i+1}: MAE={fold_metrics['MAE']:.2f}, "
              f"RMSE={fold_metrics['RMSE']:.2f}, MAPE={fold_metrics['MAPE']:.2f}%")
            
    # Calculate average metrics
    avg_metrics = {key: np.mean(values) for key, values in cv_metrics.items()}
    
    # Print average results
    print(f"\nAverage CV metrics: MAE={avg_metrics['MAE']:.2f}, "
          f"RMSE={avg_metrics['RMSE']:.2f}, MAPE={avg_metrics['MAPE']:.2f}%")
    
    # Create visualization of CV splits
    if plot:
        plt.figure(figsize=(10, n_splits * 0.5))
        for i, (train, test) in enumerate(zip(train_indices, test_indices)):
            plt.barh(i, len(train), left=0, height=0.4, color='blue', alpha=0.6, label='Train' if i==0 else "")
            plt.barh(i, len(test), left=max(train)+gap, height=0.4, color='red', alpha=0.6, label='Test' if i==0 else "")
            
        plt.yticks(range(n_splits), [f"Fold {i+1}" for i in range(n_splits)])
        plt.xlabel('Sample index')
        plt.title('Time Series Cross-Validation Splits')
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        # Only save if output_dir is provided
        if output_dir:
            plot_path = os.path.join(output_dir, 'time_series_cv_splits.png')
            plt.savefig(plot_path)
            print(f"Cross-validation plot saved to {plot_path}")
        else:
            plt.show()
        
    return avg_metrics, cv_metrics

def calculate_regression_metrics(y_true, y_pred):
    """Calculate standard regression metrics"""
    # Ensure numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    # MAPE with handling for zeros
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }