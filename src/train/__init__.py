"""
Training module for sales prediction system
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from .model import train_xgboost_model
from .evaluation import evaluate_model
from .data_preparation import prepare_model_data, split_time_series_data
# Import the visualization functions - don't redefine them here
from .visualization import plot_actual_vs_predicted, plot_feature_importance
from config.config import OUTPUT_DIR

__all__ = [
    'train_xgboost_model', 
    'evaluate_model',
    'prepare_model_data',
    'split_time_series_data',
    'plot_actual_vs_predicted',
    'plot_feature_importance'
]