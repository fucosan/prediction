# Sales Prediction System

A comprehensive sales prediction system for forecasting product sales with modular architecture supporting preprocessing, model training, incremental updates, and predictions.

## System Architecture

The system is organized into several key modules:

- **Preprocessing**: Data cleaning, feature engineering and preparation
- **Training**: Model training and evaluation
- **Incremental Training**: Update existing models with new data
- **Prediction**: Generate forecasts using trained models
- **Data Mapper**: Maintain valid item and site combinations

## Quick Start

### Initial Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Operations Guide

This guide provides step-by-step instructions for running each component of the sales prediction system.

### 1. Data Mapper Management

Create a data mapper to define valid item-site combinations:

```bash
# Create mapper from data file
python create_mapper.py data/sales_data.csv

# Use advanced options with CLI tool
python data_mapper_cli.py create data/sales_data.csv --add-cols Item_Name Site_Name

# Update existing mapper with new data
python data_mapper_cli.py update data/new_products.csv --mode merge

# View mapper statistics
python data_mapper_cli.py stats
```

### 2. Model Training

Train a new prediction model:

```bash
# Run full training pipeline
python main_training.py
```

The trained model and evaluation metrics will be saved to `output/train`.

### 3. Incremental Training

Update an existing model with new data:

```bash
# Basic incremental update
python main_incremental.py data/new_sales_data.csv

# Skip filtering for pre-filtered data
python main_incremental.py data/new_sales_data.csv --no-filter

# Specify additional boosting rounds
python main_incremental.py data/new_sales_data.csv --rounds 20
```

### 4. Making Predictions

Generate sales forecasts:

```bash
# Basic prediction
python main_predict.py data/data_to_predict.csv

# Save with specific name and format
python main_predict.py data/data_to_predict.csv --output forecasts/june_forecast --format excel

# Include confidence intervals
python main_predict.py data/data_to_predict.csv --confidence
```

### 5. Compare Predictions with Actual Sales

```bash
# Basic comparison
python main_compare.py output/predict/predictions.csv data/actual_sales.csv
```

Output files will be saved to `output/predict` by default.

### 6. Common Workflows

#### Weekly Forecast Update

```bash
# 1. Update mapper with latest product list
python data_mapper_cli.py update data/data.csv --mode deactivate_missing

# 2. Train model from scratch
python main_training.py

# 3. Generate weekly forecasts
## data could be csv or excel
python main_predict.py data/new_data.xlsx

# 4. Incremental update with new sales data
python main_incremental.py data/new_data.csv

# 5. Compare predictions with actual sales
python3 main_compare.py output/predict/predictions_20250621_211011.csv data/new_data.xlsx

```

### 7. Troubleshooting

If you encounter errors:

1. Check the data format matches expected schema
2. Ensure the data mapper is up-to-date with valid product-site combinations
3. Make sure all directories in config.py exist
4. For prediction errors, verify model file exists at the configured path

For more detailed information, refer to the logs output during execution.
