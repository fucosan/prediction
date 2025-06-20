"""
Functions for formatting and saving prediction results
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Optional, Dict, Any

from config.config import PREDICT_DIR, OUTPUT_DIR

def format_predictions(
    predictions: np.ndarray,
    metadata: pd.DataFrame,
    include_confidence: bool = False
) -> pd.DataFrame:
    """
    Format predictions into a DataFrame with metadata
    
    Args:
        predictions: Array of predictions
        metadata: DataFrame with metadata (Site_No, Item_No, etc.)
        include_confidence: Whether to include confidence intervals
    
    Returns:
        DataFrame with predictions and metadata
    """
    print("Formatting prediction results...")
    
    # Create a copy of metadata
    result = metadata.copy()
    
    # Add predictions
    result['Predicted_Quantity'] = predictions
    
    # Add confidence intervals if requested
    if include_confidence:
        # This is a simplified approach - in a real system you'd use proper
        # confidence interval calculations based on prediction variance
        confidence_level = 0.2  # 20% confidence band
        result['Lower_Bound'] = result['Predicted_Quantity'] * (1 - confidence_level)
        result['Upper_Bound'] = result['Predicted_Quantity'] * (1 + confidence_level)
    
    return result

def save_predictions(
    predictions_df: pd.DataFrame,
    output_path: Optional[str] = None,
    formats: list = ['csv', 'excel']
) -> Dict[str, str]:
    """
    Save predictions to various output formats
    
    Args:
        predictions_df: DataFrame with predictions
        output_path: Base output path (without extension)
        formats: List of output formats ('csv', 'excel', 'parquet')
    
    Returns:
        Dictionary of output paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(PREDICT_DIR, exist_ok=True)
    
    # Generate default output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(PREDICT_DIR, f'predictions_{timestamp}')
    
    output_files = {}
    
    # Save in each format
    for fmt in formats:
        if fmt.lower() == 'csv':
            file_path = f"{output_path}.csv"
            predictions_df.to_csv(file_path, index=False)
            output_files['csv'] = file_path
            print(f"Predictions saved to {file_path}")
        
        elif fmt.lower() == 'excel':
            file_path = f"{output_path}.xlsx"
            predictions_df.to_excel(file_path, index=False, sheet_name='Predictions')
            output_files['excel'] = file_path
            print(f"Predictions saved to {file_path}")
        
        elif fmt.lower() == 'parquet':
            file_path = f"{output_path}.parquet"
            predictions_df.to_parquet(file_path, index=False)
            output_files['parquet'] = file_path
            print(f"Predictions saved to {file_path}")
    
    return output_files

def create_prediction_summary(
    predictions_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> str:
    """
    Create a summary report of the predictions
    
    Args:
        predictions_df: DataFrame with predictions
        output_path: Output path for the summary
    
    Returns:
        Path to the saved summary
    """
    # Create output directory if it doesn't exist
    os.makedirs(PREDICT_DIR, exist_ok=True)
    
    # Generate default output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(PREDICT_DIR, f'summary_{timestamp}.txt')
    
    # Calculate summary statistics
    total_quantity = predictions_df['Predicted_Quantity'].sum()
    avg_quantity = predictions_df['Predicted_Quantity'].mean()
    site_summary = predictions_df.groupby('Site_No')['Predicted_Quantity'].agg(['sum', 'mean', 'count'])
    
    if 'Item_No' in predictions_df.columns:
        item_summary = predictions_df.groupby('Item_No')['Predicted_Quantity'].agg(['sum', 'mean', 'count'])
        top_items = item_summary.sort_values('sum', ascending=False).head(10)
    
    # Write summary
    with open(output_path, 'w') as f:
        f.write("Sales Prediction Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Prediction date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total records: {len(predictions_df)}\n")
        f.write(f"Total predicted quantity: {total_quantity:,.2f}\n")
        f.write(f"Average predicted quantity: {avg_quantity:,.2f}\n\n")
        
        f.write("Site Summary (Top 10 by Total Quantity)\n")
        f.write("-" * 50 + "\n")
        top_sites = site_summary.sort_values('sum', ascending=False).head(10)
        for site, stats in top_sites.iterrows():
            f.write(f"Site {site}: Total={stats['sum']:,.2f}, Avg={stats['mean']:,.2f}, Count={stats['count']}\n")
        
        if 'Item_No' in predictions_df.columns:
            f.write("\nItem Summary (Top 10 by Total Quantity)\n")
            f.write("-" * 50 + "\n")
            for item, stats in top_items.iterrows():
                f.write(f"Item {item}: Total={stats['sum']:,.2f}, Avg={stats['mean']:,.2f}, Count={stats['count']}\n")
    
    print(f"Prediction summary saved to {output_path}")
    return output_path