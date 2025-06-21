"""
Functions for calculating error metrics and generating reports
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
import seaborn as sns
from datetime import datetime

def calculate_metrics(
    comparison_df: pd.DataFrame,
    prediction_column: str = 'Predicted_Quantity',
    actual_column: str = 'Quantity',
    group_by_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate error metrics from comparison data
    
    Args:
        comparison_df: DataFrame with predictions and actual values
        prediction_column: Column name for predictions
        actual_column: Column name for actual values
        group_by_columns: Optional list of columns to group by for segmented metrics
    
    Returns:
        Dictionary with calculated metrics
    """
    print("Calculating error metrics")
    
    # Filter to only matched records with both prediction and actual data
    matched_data = comparison_df[
        (comparison_df['Status'] == 'matched') & 
        (~comparison_df[actual_column].isna()) &
        (~comparison_df[prediction_column].isna())
    ].copy()
    
    if len(matched_data) == 0:
        print("Warning: No matched records with actual data for metric calculation")
        return {
            'error': 'No matched records for comparison',
            'record_count': 0
        }
    
    # Calculate overall metrics
    metrics = {
        'record_count': len(matched_data),
        'mae': matched_data['Abs_Error'].mean(),
        'mse': (matched_data['Error'] ** 2).mean(),
        'rmse': np.sqrt((matched_data['Error'] ** 2).mean()),
        'mape': matched_data['Abs_Pct_Error'].replace([np.inf, -np.inf], np.nan).mean(),
        'total_actual': matched_data[actual_column].sum(),
        'total_predicted': matched_data[prediction_column].sum(),
        'total_error': matched_data['Error'].sum(),
        'total_error_percent': matched_data['Error'].sum() / matched_data[actual_column].sum() * 100 if matched_data[actual_column].sum() != 0 else np.nan
    }
    
    # Calculate median metrics (often more robust)
    metrics.update({
        'median_error': matched_data['Error'].median(),
        'median_abs_error': matched_data['Abs_Error'].median(),
        'median_pct_error': matched_data['Pct_Error'].median(),
        'median_abs_pct_error': matched_data['Abs_Pct_Error'].median()
    })
    
    # Add error distribution metrics
    metrics.update({
        'error_std': matched_data['Error'].std(),
        'error_min': matched_data['Error'].min(),
        'error_25pct': matched_data['Error'].quantile(0.25),
        'error_75pct': matched_data['Error'].quantile(0.75),
        'error_max': matched_data['Error'].max()
    })
    
    # Calculate grouped metrics if requested
    if group_by_columns:
        valid_group_cols = [col for col in group_by_columns if col in matched_data.columns]
        
        if valid_group_cols:
            print(f"Calculating metrics grouped by {valid_group_cols}")
            grouped_metrics = {}
            
            # Create groups
            grouped = matched_data.groupby(valid_group_cols)
            
            # Calculate metrics for each group
            for name, group in grouped:
                # Convert group name to string for dict key
                if isinstance(name, tuple):
                    group_name = '_'.join(str(x) for x in name)
                else:
                    group_name = str(name)
                
                grouped_metrics[group_name] = {
                    'record_count': len(group),
                    'mae': group['Abs_Error'].mean(),
                    'rmse': np.sqrt((group['Error'] ** 2).mean()),
                    'mape': group['Abs_Pct_Error'].replace([np.inf, -np.inf], np.nan).mean(),
                    'total_actual': group[actual_column].sum(),
                    'total_predicted': group[prediction_column].sum(),
                    'total_error': group['Error'].sum(),
                    'total_error_percent': group['Error'].sum() / group[actual_column].sum() * 100 if group[actual_column].sum() != 0 else np.nan
                }
            
            metrics['grouped_metrics'] = grouped_metrics
    
    # Add metadata
    metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    metrics['metric_version'] = '1.0'
    
    print(f"Calculated metrics: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%")
    
    return metrics

def generate_metrics_report(
    metrics: Dict[str, Any],
    comparison_df: pd.DataFrame,
    output_dir: str = 'output/compare',
    output_prefix: str = 'metrics',
    create_plots: bool = True
) -> Dict[str, str]:
    """
    Generate report files based on metrics
    
    Args:
        metrics: Dictionary of calculated metrics
        comparison_df: DataFrame with predictions and actual values
        output_dir: Directory for output files
        output_prefix: Prefix for output filenames
        create_plots: Whether to create visualization plots
    
    Returns:
        Dictionary of output file paths
    """
    print("Generating metrics report")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_path = os.path.join(output_dir, f"{output_prefix}_{timestamp}")
    
    output_files = {}
    
    # Convert NumPy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Apply conversion to metrics dictionary
    metrics_json_safe = convert_numpy_types(metrics)
    
    # Save metrics as JSON
    json_path = f"{base_path}.json"
    with open(json_path, 'w') as f:
        json.dump(metrics_json_safe, f, indent=2)
    output_files['json'] = json_path
    print(f"Saved metrics to {json_path}")
    
    # Create text report
    txt_path = f"{base_path}.txt"
    with open(txt_path, 'w') as f:
        f.write("PREDICTION EVALUATION METRICS\n")
        f.write("============================\n\n")
        f.write(f"Generated: {metrics.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("--------------\n")
        f.write(f"Records Compared: {metrics.get('record_count', 'N/A')}\n")
        f.write(f"Mean Absolute Error (MAE): {metrics.get('mae', 'N/A'):.2f}\n")
        f.write(f"Root Mean Square Error (RMSE): {metrics.get('rmse', 'N/A'):.2f}\n")
        f.write(f"Mean Absolute Percentage Error (MAPE): {metrics.get('mape', 'N/A'):.2f}%\n")
        f.write(f"Total Actual: {metrics.get('total_actual', 'N/A'):.0f}\n")
        f.write(f"Total Predicted: {metrics.get('total_predicted', 'N/A'):.0f}\n")
        f.write(f"Total Error: {metrics.get('total_error', 'N/A'):.0f} ({metrics.get('total_error_percent', 'N/A'):.2f}%)\n\n")
        
        f.write("ERROR DISTRIBUTION\n")
        f.write("-----------------\n")
        f.write(f"Minimum Error: {metrics.get('error_min', 'N/A'):.2f}\n")
        f.write(f"25th Percentile: {metrics.get('error_25pct', 'N/A'):.2f}\n")
        f.write(f"Median Error: {metrics.get('median_error', 'N/A'):.2f}\n")
        f.write(f"75th Percentile: {metrics.get('error_75pct', 'N/A'):.2f}\n")
        f.write(f"Maximum Error: {metrics.get('error_max', 'N/A'):.2f}\n\n")
        
        if 'grouped_metrics' in metrics and metrics['grouped_metrics']:
            f.write("SEGMENTED METRICS\n")
            f.write("----------------\n")
            for group_name, group_metrics in metrics['grouped_metrics'].items():
                f.write(f"\nGroup: {group_name}\n")
                f.write(f"  Records: {group_metrics.get('record_count', 'N/A')}\n")
                f.write(f"  MAE: {group_metrics.get('mae', 'N/A'):.2f}\n")
                f.write(f"  RMSE: {group_metrics.get('rmse', 'N/A'):.2f}\n")
                f.write(f"  MAPE: {group_metrics.get('mape', 'N/A'):.2f}%\n")
                f.write(f"  Total Error: {group_metrics.get('total_error', 'N/A'):.0f} ({group_metrics.get('total_error_percent', 'N/A'):.2f}%)\n")
    
    output_files['txt'] = txt_path
    print(f"Saved text report to {txt_path}")
    
    # Create visualizations if requested
    if create_plots:
        # Filter valid data for plotting
        plot_data = comparison_df[
            (comparison_df['Status'] == 'matched') & 
            (~comparison_df['Predicted_Quantity'].isna()) &
            (~comparison_df['Quantity'].isna())
        ].copy()
        
        if len(plot_data) > 0:
            # 1. Actual vs. Predicted Scatter Plot
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='Quantity', y='Predicted_Quantity', data=plot_data, alpha=0.5)
            
            # Add diagonal line (perfect predictions)
            max_val = max(plot_data['Quantity'].max(), plot_data['Predicted_Quantity'].max())
            plt.plot([0, max_val], [0, max_val], 'r--')
            
            plt.title('Actual vs Predicted Values')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.grid(True, alpha=0.3)
            
            scatter_path = f"{base_path}_scatter.png"
            plt.tight_layout()
            plt.savefig(scatter_path)
            plt.close()
            output_files['scatter_plot'] = scatter_path
            print(f"Saved scatter plot to {scatter_path}")
            
            # 2. Error Distribution Histogram
            plt.figure(figsize=(10, 6))
            sns.histplot(plot_data['Error'], kde=True, bins=30)
            plt.title('Error Distribution')
            plt.xlabel('Error (Predicted - Actual)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Add vertical line at zero
            plt.axvline(x=0, color='r', linestyle='--')
            
            hist_path = f"{base_path}_error_hist.png"
            plt.tight_layout()
            plt.savefig(hist_path)
            plt.close()
            output_files['hist_plot'] = hist_path
            print(f"Saved error histogram to {hist_path}")
            
            # 3. Error by Quantity Magnitude
            plt.figure(figsize=(10, 6))
            plt.scatter(plot_data['Quantity'], plot_data['Error'], alpha=0.5)
            plt.title('Error by Actual Quantity')
            plt.xlabel('Actual Quantity')
            plt.ylabel('Error (Predicted - Actual)')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='r', linestyle='--')
            
            error_by_qty_path = f"{base_path}_error_by_qty.png"
            plt.tight_layout()
            plt.savefig(error_by_qty_path)
            plt.close()
            output_files['error_by_qty_plot'] = error_by_qty_path
            print(f"Saved error by quantity plot to {error_by_qty_path}")
    
    return output_files