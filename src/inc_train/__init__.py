"""
Incremental training module for sales prediction system
"""

from .data_processing import process_new_data, update_feature_store
from .model_updating import update_model
from .utils import load_existing_artifacts

__all__ = [
    'process_new_data',
    'update_feature_store',
    'update_model',
    'load_existing_artifacts'
]