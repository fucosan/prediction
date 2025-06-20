"""
Visualization functions for model results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Optional

from config.config import FEATURE_IMPORTANCE_PLOT, PREDICTION_PLOT

def plot_actual_vs_predicted(y_test: pd.Series, y_pred: np.ndarray, max_samples: int = 100) -> None:
    """
    Plot actual vs predicted values.
    
    Args:
        y_test: Test target vector
        y_pred: Predicted values
        max_samples: Maximum number of samples to plot
    """
    print("Creating actual vs predicted visualization...")
    plt.figure(figsize=(12, 6))
    
    # Get first N samples (or all if less than max_samples)
    samples = min(max_samples, len(y_test))
    
    # Create the plot
    plt.plot(range(samples), y_test.iloc[:samples], 'b-', label='Actual')
    plt.plot(range(samples), y_pred[:samples], 'r-', label='Predicted')
    
    # Set labels and title
    plt.title('Actual vs Predicted Quantity')
    plt.xlabel('Time Index')
    plt.ylabel('Quantity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(PREDICTION_PLOT), exist_ok=True)
    
    # Save the plot
    plt.savefig(PREDICTION_PLOT)
    print(f"Plot saved to {PREDICTION_PLOT}")
    plt.close()

def plot_feature_importance(model, feature_names: List[str], top_n: int = 10) -> None:
    """
    Visualize top N most important features using a bar chart.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        top_n: Number of top features to display (default: 10)
    """
    print(f"Creating feature importance visualization...")
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create a DataFrame for better visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False).head(top_n).reset_index(drop=True)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create bar chart
    bars = plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='#3498db')
    
    # Add values on the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{width:.4f}', ha='left', va='center')
    
    # Set labels and title
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Important Features')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(FEATURE_IMPORTANCE_PLOT), exist_ok=True)
    
    # Save the plot
    plt.savefig(FEATURE_IMPORTANCE_PLOT)
    print(f"Feature importance plot saved to {FEATURE_IMPORTANCE_PLOT}")
    plt.close()