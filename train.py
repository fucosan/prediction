"""
XGBoost-based sales prediction model training and evaluation.

This module implements section 7 of the requirements, handling:
- Model training using XGBoost regression
- Model evaluation using RMSE and MAE
- Visualization of actual vs predicted values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import os
import sys


def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train an XGBoost regression model.
    """
    # Default XGBoost parameters
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'objective': 'reg:squarederror',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    # Initialize and train the model
    print("Training XGBoost model...")
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model using RMSE and MAE.
    """
    # Make predictions
    print("Evaluating model performance...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Model Evaluation Metrics:")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - MAE: {mae:.4f}")
    
    # Save metrics to file
    output_dir = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    metrics_path = os.path.join(output_dir, 'model_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("XGBoost Model Evaluation Metrics\n")
        f.write("=" * 35 + "\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        f.write("\n")
        f.write(f"Test samples: {len(y_test)}\n")
        f.write(f"Predictions range: [{y_pred.min():.4f}, {y_pred.max():.4f}]\n")
        f.write(f"Actual values range: [{y_test.min():.4f}, {y_test.max():.4f}]\n")
    
    print(f"Metrics saved to {metrics_path}")
    
    # Also save as CSV for easy data analysis
    metrics_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE'],
        'Value': [rmse, mae]
    })
    metrics_csv_path = os.path.join(output_dir, 'model_metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics CSV saved to {metrics_csv_path}")
    
    return y_pred, rmse, mae


def plot_actual_vs_predicted(y_test, y_pred):
    """
    Plot actual vs predicted values (first 100 points).
    """
    print("Creating visualization...")
    plt.figure(figsize=(12, 6))
    
    # Get first 100 samples (or all if less than 100)
    samples = min(100, len(y_test))
    
    # Create the plot
    plt.plot(range(samples), y_test.iloc[:samples], 'b-', label='Actual')
    plt.plot(range(samples), y_pred[:samples], 'r-', label='Predicted')
    
    # Set labels and title
    plt.title('Actual vs Combined Predicted Quantity (first 100 points)')
    plt.xlabel('Time Index')
    plt.ylabel('Quantity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'actual_vs_predicted.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()


def plot_feature_importance(model, feature_names, top_n=10):
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
    plt.title(f'Top {top_n} Important Features (XGBoost)')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plot_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(plot_path)
    print(f"Feature importance plot saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    # TRAINING PHASE - Only using training data
    # =========================================
    
    # Check if training files exist
    if not os.path.exists('X_train.csv') or not os.path.exists('y_train.csv'):
        sys.exit("Error: Required training files 'X_train.csv' and 'y_train.csv' not found. "
                 "Run the processing script first to generate these files.")
    
    # Load training data
    print("=== TRAINING PHASE ===")
    print("Loading training data...")
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv').iloc[:, 0]  # Convert to Series
    
    # Clean training data
    date_cols = ['Date', 'Start_Date', 'End_Date']
    for col in date_cols:
        if col in X_train.columns:
            print(f"Removing {col} column from training data")
            X_train = X_train.drop(columns=[col])
    
    # Process object columns in training data
    object_cols = X_train.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"Converting object columns to numeric: {list(object_cols)}")
        for col in object_cols:
            try:
                X_train[col] = pd.to_numeric(X_train[col])
            except:
                X_train[col] = X_train[col].astype('category').cat.codes
                print(f"  - Converted {col} to category codes")
    
    # Train the model (using ONLY training data)
    model = train_xgboost_model(X_train, y_train)
    
    # Save the model after training is complete
    output_dir = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_path = os.path.join(output_dir, 'xgboost_sales_model.json')
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Display feature importance from training data
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    
    print("\nTop 10 important features:")
    print(feature_importance.head(10))
    
    # Visualize feature importance
    plot_feature_importance(model, X_train.columns, top_n=10)
    
    print("\nTraining phase complete!")
    
    # EVALUATION PHASE - Only using test data for evaluation (NOT training)
    # ====================================================================
    
    # Check if test files exist
    has_test_data = os.path.exists('X_test.csv') and os.path.exists('y_test.csv')
    
    if has_test_data:
        print("\n=== EVALUATION PHASE ===")
        print("Loading test data...")
        X_test = pd.read_csv('X_test.csv')
        y_test = pd.read_csv('y_test.csv').iloc[:, 0]  # Convert to Series
        
        # Clean test data - using same preprocessing as training data
        for col in date_cols:
            if col in X_test.columns:
                print(f"Removing {col} column from test data")
                X_test = X_test.drop(columns=[col])
        
        object_cols_test = X_test.select_dtypes(include=['object']).columns
        if len(object_cols_test) > 0:
            print(f"Converting object columns in test data: {list(object_cols_test)}")
            for col in object_cols_test:
                try:
                    X_test[col] = pd.to_numeric(X_test[col])
                except:
                    X_test[col] = X_test[col].astype('category').cat.codes
                    print(f"  - Converted {col} to category codes")
        
        # Evaluate model on test data (NO retraining or model updates here)
        y_pred, rmse, mae = evaluate_model(model, X_test, y_test)
        
        # Save detailed results summary
        output_dir = os.path.join(os.getcwd(), 'output')
        summary_path = os.path.join(output_dir, 'evaluation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("XGBoost Sales Prediction Model - Evaluation Summary\n")
            f.write("=" * 55 + "\n\n")
            f.write("Training Configuration:\n")
            f.write(f"  - Training samples: {len(X_train)}\n")
            f.write(f"  - Features: {len(X_train.columns)}\n")
            f.write(f"  - Test samples: {len(X_test)}\n\n")
            f.write("Performance Metrics:\n")
            f.write(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}\n")
            f.write(f"  - Mean Absolute Error (MAE): {mae:.4f}\n\n")
            f.write("Data Statistics:\n")
            f.write(f"  - Actual values range: [{y_test.min():.4f}, {y_test.max():.4f}]\n")
            f.write(f"  - Predicted values range: [{y_pred.min():.4f}, {y_pred.max():.4f}]\n")
            f.write(f"  - Actual mean: {y_test.mean():.4f}\n")
            f.write(f"  - Predicted mean: {y_pred.mean():.4f}\n")
        
        print(f"Evaluation summary saved to {summary_path}")
        
        # Visualize results (only using test data)
        plot_actual_vs_predicted(y_test, y_pred)
    else:
        print("\nWarning: Test files 'X_test.csv' and 'y_test.csv' not found.")
        print("Skipping evaluation phase. Only model training was performed.")
    
    print("\nProcess complete!")