"""
Normal (one-step) prediction and evaluation for sales forecasting.

This module implements section 9 of the requirements, handling:
- Normal prediction on test set using actual historical data
- Evaluation of predictions using RMSE and MAE
- Visualization of actual vs predicted values
- Export of results to CSV
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
    
    # Load test labels
    y_test = pd.read_csv('y_test.csv').iloc[:, 0]  # Convert to Series
    
    # Load trained model
    model = xgb.XGBRegressor()
    model.load_model('output/xgboost_sales_model.json')
    
    print("Data and model loaded successfully.")
    return processed_df, X_test, y_test, model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model using RMSE and MAE.
    
    Args:
        model: Trained XGBoost model
        X_test: Test feature matrix
        y_test: Test target values
        
    Returns:
        Tuple of (predictions, RMSE, MAE)
    """
    print("Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print("Normal Prediction Metrics:")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - MAE: {mae:.4f}")
    
    return y_pred, rmse, mae


def plot_actual_vs_predicted(y_test, y_pred):
    """
    Plot actual vs predicted values.
    
    Args:
        y_test: True values
        y_pred: Predicted values
    """
    print("Creating visualization...")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot first 100 samples (or all if less than 100)
    samples = min(100, len(y_test))
    
    plt.figure(figsize=(12, 6))
    
    # Plot actual and predicted values
    plt.plot(range(samples), y_test.iloc[:samples], 'b-', label='Actual')
    plt.plot(range(samples), y_pred[:samples], 'r-', label='Predicted (Normal)')
    
    # Set labels and title
    plt.title('Actual vs Predicted Quantity - Normal Prediction')
    plt.xlabel('Time Index')
    plt.ylabel('Quantity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'normal_prediction.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()


def export_results(processed_df, y_test, y_pred):
    """
    Export results to CSV file with correct date alignment.
    """
    print("Exporting results...")
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the ORIGINAL test data split that contains the metadata
    # This is needed for proper mapping
    test_data_full = pd.read_csv('X_test_metadata.csv', parse_dates=['Start_Date', 'End_Date'])
    
    if 'X_test_metadata.csv' not in os.listdir():
        print("Creating a direct mapping using available data...")
        # Get test data info
        X_test_info = pd.read_csv('X_test.csv')
        
        # Create results dataframe
        results_df = pd.DataFrame()
        results_df['actual_quantity'] = y_test.values
        results_df['predicted_quantity_normal'] = y_pred
        
        # Try extracting Site_No and Item_No from X_test directly
        if 'Site_No' in X_test_info.columns:
            results_df['Site_No'] = X_test_info['Site_No']
        else:
            print("Warning: Site_No not found in X_test.csv")
            results_df['Site_No'] = "Unknown"
        
        if 'Item_No' in X_test_info.columns:
            results_df['Item_No'] = X_test_info['Item_No']
        else:
            print("Warning: Item_No not found in X_test.csv")
            results_df['Item_No'] = "Unknown"
        
        # Match with processed data (convert all identifiers to string to ensure matching works)
        results_df['Site_No'] = results_df['Site_No'].astype(str)
        results_df['Item_No'] = results_df['Item_No'].astype(str)
        
        # Convert processed_df identifiers to strings too
        processed_df['Site_No'] = processed_df['Site_No'].astype(str)
        processed_df['Item_No'] = processed_df['Item_No'].astype(str)
        
        # Create lookup dictionaries
        site_names = dict(zip(processed_df['Site_No'], processed_df['Site_Name'])) if 'Site_Name' in processed_df.columns else {}
        item_names = dict(zip(processed_df['Item_No'], processed_df['Item_Name'])) if 'Item_Name' in processed_df.columns else {}
        
        # Apply lookups
        results_df['Site_Name'] = results_df['Site_No'].map(site_names).fillna('Unknown')
        results_df['Item_Name'] = results_df['Item_No'].map(item_names).fillna('Unknown')
        
        # Get test dates by position - this is simplistic but should work if order is preserved
        if 'End_Date' in processed_df.columns:
            # Get unique Site_No, Item_No pairs in results
            unique_pairs = results_df[['Site_No', 'Item_No']].drop_duplicates()
            
            # Get the most recent dates for each Site_No, Item_No pair
            date_data = []
            for _, row in unique_pairs.iterrows():
                site = row['Site_No']
                item = row['Item_No']
                matching_rows = processed_df[(processed_df['Site_No'] == site) & 
                                          (processed_df['Item_No'] == item)]
                
                if not matching_rows.empty:
                    # Get the latest dates
                    latest = matching_rows.sort_values('End_Date', ascending=False).iloc[0]
                    date_data.append({
                        'Site_No': site,
                        'Item_No': item,
                        'Start_Date': latest['Start_Date'],
                        'End_Date': latest['End_Date']
                    })
            
            # Create a dates dataframe
            if date_data:
                dates_df = pd.DataFrame(date_data)
                
                # Merge dates into results
                results_df = results_df.merge(dates_df, on=['Site_No', 'Item_No'], how='left')
    else:
        print("Using X_test_metadata.csv for direct mapping...")
        # Use the metadata file directly
        results_df = pd.DataFrame()
        results_df['Site_No'] = test_data_full['Site_No']
        results_df['Item_No'] = test_data_full['Item_No']
        results_df['Site_Name'] = test_data_full['Site_Name'] if 'Site_Name' in test_data_full else 'Unknown'
        results_df['Item_Name'] = test_data_full['Item_Name'] if 'Item_Name' in test_data_full else 'Unknown'
        results_df['Start_Date'] = test_data_full['Start_Date']
        results_df['End_Date'] = test_data_full['End_Date']
        results_df['actual_quantity'] = y_test.values
        results_df['predicted_quantity_normal'] = y_pred
    
    # Sort by Site_No, Item_No, and date if available
    sort_cols = ['Site_No', 'Item_No']
    if 'End_Date' in results_df.columns:
        sort_cols.append('End_Date')
    
    results_df = results_df.sort_values(sort_cols)
    
    # Export to CSV
    csv_path = os.path.join(output_dir, 'xgboost_normal_predictions.csv')
    results_df.to_csv(csv_path, index=False)
    
    print(f"Results exported to {csv_path}")
    return csv_path


def run_test_pipeline():
    """
    Run the complete normal prediction test pipeline:
    1. Load data and model
    2. Make normal predictions
    3. Evaluate performance
    4. Visualize results
    5. Export results to CSV
    """
    # 1. Load data and model
    processed_df, X_test, y_test, model = load_data_and_model()
    
    # 2. Make normal predictions (one-step)
    y_pred, rmse, mae = evaluate_model(model, X_test, y_test)
    
    # 3. Visualize results
    plot_actual_vs_predicted(y_test, y_pred)
    
    # 4. Export results to CSV
    csv_path = export_results(processed_df, y_test, y_pred)
    
    print("\nNormal prediction test completed successfully!")
    return rmse, mae, csv_path


if __name__ == "__main__":
    run_test_pipeline()