"""
Recursive prediction and evaluation for sales forecasting.

This module implements sections 8 and 9 of the requirements, handling:
- Recursive prediction on test set using previously predicted values
- Evaluation of recursive predictions
- Comparison with one-step predictions
- Export of final results to CSV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import sys


def load_data_and_model():
    """
    Load the preprocessed test data and trained model.
    
    Returns:
        Tuple of (processed_df, X_test, y_test, model)
    """
    print("Loading test data and model...")
    
    # Check if required files exist
    required_files = ['processed_sales_data.csv', 'X_test.csv', 'y_test.csv', 'output/xgboost_sales_model.json']
    for file in required_files:
        if not os.path.exists(file):
            sys.exit(f"Error: Required file '{file}' not found. Run processing.py and train.py first.")
    
    # Load processed data with all features - Parse dates explicitly
    processed_df = pd.read_csv('processed_sales_data.csv', parse_dates=['Start_Date', 'End_Date'])
    
    # Load test data
    X_test = pd.read_csv('X_test.csv')
    
    # Remove any date columns that shouldn't be used for prediction
    date_cols = ['Date', 'Start_Date', 'End_Date']
    for col in date_cols:
        if col in X_test.columns:
            print(f"Removing {col} column from test data")
            X_test = X_test.drop(columns=[col])
    
    # Convert any remaining object columns
    object_cols = X_test.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"Converting object columns to numeric: {list(object_cols)}")
        for col in object_cols:
            # Try to convert to numeric, if fails convert to category codes
            try:
                X_test[col] = pd.to_numeric(X_test[col])
            except:
                X_test[col] = X_test[col].astype('category').cat.codes
                print(f"  - Converted {col} to category codes")
    
    y_test = pd.read_csv('y_test.csv').iloc[:, 0]  # Convert to Series
    
    # Load trained model
    model = xgb.XGBRegressor()
    model.load_model('output/xgboost_sales_model.json')
    
    print("Data and model loaded successfully.")
    return processed_df, X_test, y_test, model


def get_feature_mappings(feature_names):
    """
    Identify feature indexes for lag, rolling, and days_since features.
    
    Args:
        feature_names: List of feature names in the X matrix
    
    Returns:
        Dictionary mapping feature types to their column indexes
    """
    feature_mappings = {
        'lag_features': [],
        'rolling_mean_features': [],
        'rolling_sum_features': [],
        'rolling_std_features': [],
        'days_since_features': []
    }
    
    # Map feature names to their indexes
    for i, feature in enumerate(feature_names):
        if feature.startswith('lag_'):
            feature_mappings['lag_features'].append((i, int(feature.split('_')[1])))
        elif feature.startswith('rolling_mean_'):
            feature_mappings['rolling_mean_features'].append((i, int(feature.split('_')[2])))
        elif feature.startswith('rolling_sum_'):
            feature_mappings['rolling_sum_features'].append((i, int(feature.split('_')[2])))
        elif feature.startswith('rolling_std_'):
            feature_mappings['rolling_std_features'].append((i, int(feature.split('_')[2])))
        elif feature.startswith('days_since_last_sale'):
            feature_mappings['days_since_features'].append(i)
    
    return feature_mappings


def recursive_predict(processed_df, X_test, y_test, model):
    """
    Perform recursive predictions on test data.
    
    For each (Site_No, Item_No) group:
    1. Get the last known quantities from training data
    2. Process test data in chronological order
    3. For each step:
       - Update lag/rolling features using previously predicted values
       - Make prediction
       - Store prediction and update history
    
    Args:
        processed_df: Full processed dataframe with all features
        X_test: Test features
        y_test: Test target values
        model: Trained XGBoost model
    
    Returns:
        Tuple of (pandas Series of recursive predictions, original index from y_test)
    """
    print("Starting recursive prediction...")
    
    # Get test records from processed_df to access Site_No, Item_No, and End_Date
    test_indices = y_test.index
    test_records = processed_df.loc[test_indices].copy()
    
    # Make a copy of X_test for recursive prediction
    X_recursive = X_test.copy()
    
    # Get feature mappings
    feature_mappings = get_feature_mappings(X_test.columns)
    
    # Prepare history data (needs to include training data for initial predictions)
    # For this we need to find which records are in the training set
    all_indices = set(processed_df.index)
    train_indices = list(all_indices - set(test_indices))
    train_records = processed_df.loc[train_indices].copy()
    
    # Store predictions here
    y_pred_recursive = np.zeros(len(y_test))
    
    # Get unique (Site_No, Item_No) pairs in test set
    site_item_pairs = test_records[['Site_No', 'Item_No']].drop_duplicates()
    
    # Sort test_records by End_Date to ensure chronological processing
    test_records = test_records.sort_values('End_Date')
    
    print(f"Processing {len(site_item_pairs)} site-item pairs...")
    
    # Process each site-item pair separately
    for idx, (_, row) in enumerate(site_item_pairs.iterrows()):
        site_no = row['Site_No']
        item_no = row['Item_No']
        
        if idx > 0 and idx % 10 == 0:
            print(f"Processed {idx}/{len(site_item_pairs)} site-item pairs...")
        
        # Get historical quantities for this site-item pair
        site_item_history = train_records[(train_records['Site_No'] == site_no) & 
                                        (train_records['Item_No'] == item_no)].copy()
        site_item_history = site_item_history.sort_values('End_Date')
        
        # Get test records for this site-item pair in chronological order
        site_item_test = test_records[(test_records['Site_No'] == site_no) & 
                                     (test_records['Item_No'] == item_no)].copy()
        site_item_test = site_item_test.sort_values('End_Date')
        
        if len(site_item_test) == 0:
            continue
            
        # Extract historical quantities (include only most recent ones needed for features)
        max_lag = 0
        max_window = 0
        
        for _, lag in feature_mappings['lag_features']:
            max_lag = max(max_lag, lag)
            
        for feature_list in ['rolling_mean_features', 'rolling_sum_features', 'rolling_std_features']:
            for _, window in feature_mappings[feature_list]:
                max_window = max(max_window, window)
        
        history_size = max(max_lag, max_window)
        quantities = list(site_item_history['Quantity'].values[-history_size:]) if len(site_item_history) > 0 else []
        
        # If no history available, initialize with zeros
        if len(quantities) < history_size:
            quantities = [0] * (history_size - len(quantities)) + quantities
        
        # Process test records chronologically
        days_since_last_sale = site_item_history['days_since_last_sale'].values[-1] if len(site_item_history) > 0 else 0
        
        for i, test_idx in enumerate(site_item_test.index):
            # Get the corresponding row in X_recursive
            x_idx = X_recursive.index.get_loc(test_idx)
            
            # Update lag features
            for feat_idx, lag in feature_mappings['lag_features']:
                if lag <= len(quantities):
                    X_recursive.iloc[x_idx, feat_idx] = quantities[-lag]
                else:
                    X_recursive.iloc[x_idx, feat_idx] = 0
            
            # Update rolling mean features
            for feat_idx, window in feature_mappings['rolling_mean_features']:
                if window <= len(quantities):
                    X_recursive.iloc[x_idx, feat_idx] = np.mean(quantities[-window:])
                else:
                    X_recursive.iloc[x_idx, feat_idx] = np.mean(quantities)
            
            # Update rolling sum features
            for feat_idx, window in feature_mappings['rolling_sum_features']:
                if window <= len(quantities):
                    X_recursive.iloc[x_idx, feat_idx] = np.sum(quantities[-window:])
                else:
                    X_recursive.iloc[x_idx, feat_idx] = np.sum(quantities)
            
            # Update rolling std features
            for feat_idx, window in feature_mappings['rolling_std_features']:
                if window <= len(quantities):
                    X_recursive.iloc[x_idx, feat_idx] = np.std(quantities[-window:]) if len(quantities[-window:]) > 1 else 0
                else:
                    X_recursive.iloc[x_idx, feat_idx] = np.std(quantities) if len(quantities) > 1 else 0
            
            # Update days_since_last_sale
            for feat_idx in feature_mappings['days_since_features']:
                X_recursive.iloc[x_idx, feat_idx] = days_since_last_sale
            
            # Make the prediction
            pred = model.predict(X_recursive.iloc[[x_idx]])[0]
            
            # Ensure prediction is non-negative
            pred = max(0, pred)
            
            # Store the prediction
            y_pred_recursive[y_test.index.get_loc(test_idx)] = pred
            
            # Update history
            quantities.append(pred)
            quantities = quantities[1:]  # Remove oldest
            
            # Update days_since_last_sale
            if pred > 0:
                days_since_last_sale = 0
            else:
                days_since_last_sale += 1
    
    print("Recursive prediction complete.")
    return pd.Series(y_pred_recursive, index=y_test.index), test_records


def evaluate_and_compare(y_test, y_pred_standard, y_pred_recursive):
    """
    Evaluate and compare standard and recursive predictions.
    
    Args:
        y_test: True values
        y_pred_standard: One-step predictions
        y_pred_recursive: Recursive predictions
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("Evaluating predictions...")
    
    # Calculate metrics for standard predictions
    rmse_standard = np.sqrt(mean_squared_error(y_test, y_pred_standard))
    mae_standard = mean_absolute_error(y_test, y_pred_standard)
    
    # Calculate metrics for recursive predictions
    rmse_recursive = np.sqrt(mean_squared_error(y_test, y_pred_recursive))
    mae_recursive = mean_absolute_error(y_test, y_pred_recursive)
    
    print("Standard Prediction Metrics:")
    print(f"  - RMSE: {rmse_standard:.4f}")
    print(f"  - MAE: {mae_standard:.4f}")
    
    print("Recursive Prediction Metrics:")
    print(f"  - RMSE: {rmse_recursive:.4f}")
    print(f"  - MAE: {mae_recursive:.4f}")
    
    return {
        'rmse_standard': rmse_standard,
        'mae_standard': mae_standard,
        'rmse_recursive': rmse_recursive,
        'mae_recursive': mae_recursive
    }


def plot_comparison(y_test, y_pred_standard, y_pred_recursive):
    """
    Create comparison plots of actual vs standard predictions vs recursive predictions.
    
    Args:
        y_test: True values
        y_pred_standard: One-step predictions
        y_pred_recursive: Recursive predictions
    """
    print("Creating visualization...")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot first 100 samples (or all if less than 100)
    samples = min(100, len(y_test))
    
    # Create a plot comparing both prediction methods
    plt.figure(figsize=(16, 8))
    
    # Plot actual, standard prediction, and recursive prediction
    plt.plot(range(samples), y_test.iloc[:samples], 'b-', label='Actual', linewidth=2)
    plt.plot(range(samples), y_pred_standard[:samples], 'g-', label='Standard Prediction', linewidth=1, alpha=0.7)
    plt.plot(range(samples), y_pred_recursive[:samples], 'r-', label='Recursive Prediction', linewidth=1.5)
    
    # Set labels and title
    plt.title('Comparison of Actual, Standard, and Recursive Predictions (first 100 points)')
    plt.xlabel('Time Index')
    plt.ylabel('Quantity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'prediction_comparison.png')
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")
    plt.close()


def export_results(processed_df, y_test, y_pred_recursive):
    """
    Prepare and export final results dataframe.
    
    Args:
        processed_df: Full processed dataframe with all features
        y_test: True values 
        y_pred_recursive: Recursive predictions
        
    Returns:
        Path to the exported CSV file
    """
    print("Exporting results...")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get test records
    test_indices = y_test.index
    test_records = processed_df.loc[test_indices].copy()
    
    # Create final results dataframe
    results_df = pd.DataFrame()
    
    # Add required columns
    results_df['Site_No'] = test_records['Site_No']
    results_df['Item_No'] = test_records['Item_No']
    
    # Add Site_Name and Item_Name if available
    if 'Site_Name' in test_records.columns:
        results_df['Site_Name'] = test_records['Site_Name']
    else:
        results_df['Site_Name'] = 'Unknown'
        
    if 'Item_Name' in test_records.columns:
        results_df['Item_Name'] = test_records['Item_Name']
    else:
        results_df['Item_Name'] = 'Unknown'
    
    # Add date information - ensure they're properly formatted
    if 'End_Date' in test_records.columns:
        results_df['End_Date'] = pd.to_datetime(test_records['End_Date']).dt.strftime('%Y-%m-%d')
    if 'Start_Date' in test_records.columns:
        results_df['Start_Date'] = pd.to_datetime(test_records['Start_Date']).dt.strftime('%Y-%m-%d')
    
    # Add actual and predicted quantities
    results_df['actual_quantity'] = y_test.values
    results_df['predicted_quantity'] = y_pred_recursive.values
    
    # Sort by Site_No, Item_No, and date
    sort_cols = ['Site_No', 'Item_No']
    if 'End_Date' in results_df.columns:
        sort_cols.append('End_Date')
    
    results_df = results_df.sort_values(sort_cols)
    
    # Export to CSV with date format specified
    csv_path = os.path.join(output_dir, 'xgboost_recursive_predictions.csv')
    results_df.to_csv(csv_path, index=False, date_format='%Y-%m-%d')
    
    print(f"Results exported to {csv_path}")
    return csv_path


def run_test_pipeline():
    """
    Run the complete test pipeline:
    1. Load data and model
    2. Perform recursive predictions
    3. Evaluate and compare with standard predictions
    4. Visualize results
    5. Export final results
    """
    # 1. Load data and model
    processed_df, X_test, y_test, model = load_data_and_model()
    
    # 2. Make standard predictions (one-step)
    print("Making standard (one-step) predictions...")
    y_pred_standard = model.predict(X_test)
    
    # 3. Perform recursive predictions
    y_pred_recursive, test_records = recursive_predict(processed_df, X_test, y_test, model)
    
    # 4. Evaluate and compare predictions
    metrics = evaluate_and_compare(y_test, y_pred_standard, y_pred_recursive)
    
    # 5. Visualize results
    plot_comparison(y_test, y_pred_standard, y_pred_recursive)
    
    # 6. Export final results
    csv_path = export_results(processed_df, y_test, y_pred_recursive)
    
    print("\nTest pipeline completed successfully!")
    return metrics, csv_path


if __name__ == "__main__":
    run_test_pipeline()