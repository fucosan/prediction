"""
Configuration settings for the sales prediction system
"""

import os
from typing import Dict, List

# Path configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "train")
PREPROCESS_DIR = os.path.join(OUTPUT_DIR, "preprocess")
INC_TRAIN_DIR = os.path.join(OUTPUT_DIR, "inc_train")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")
PREDICT_DIR = os.path.join(OUTPUT_DIR, "predict")
COMPARE_DIR = os.path.join(OUTPUT_DIR, "compare")
GEN_DATA_DIR = os.path.join(OUTPUT_DIR, "gen_data")

# Create output directories if they don't exist
for directory in [OUTPUT_DIR, MODEL_DIR, PREPROCESS_DIR, INC_TRAIN_DIR, TEST_DIR, PREDICT_DIR, COMPARE_DIR, GEN_DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data files
DATA_MAPPER_PATH = os.path.join(PREPROCESS_DIR, "data_mapper.csv")
REFERENCE_DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")

# Model files
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_sales_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
FEATURE_STORE_PATH = os.path.join(PREPROCESS_DIR, "feature_store.parquet")
LOG_PATH = os.path.join(INC_TRAIN_DIR, "incremental_update_log.csv")

# Feature engineering configurations
LOOKBACK_DAYS = 365  # ~12 months
EXTRA_BOOST_ROUNDS = 10
LAG_PERIODS = [1, 2, 3, 7, 14]
WINDOW_SIZES = [3, 7, 14, 28]
ROLLING_METRICS = ['mean', 'sum', 'std']

# Bi-weekly specific feature parameters
# These represent number of periods, not days
BI_WEEKLY_LAG_PERIODS = [1, 2, 3, 4, 6, 8, 12]  # Previous periods (2 weeks to 24 weeks)
BI_WEEKLY_WINDOW_SIZES = [2, 4, 6, 12, 26]  # Rolling windows (1 month to 12 months)

# Column groups
KEEP_COLS = [
    'Add_Periodic_Disc_Amount_IDR',
    'Amount_Discount_Tambahan_incl_PPN_IDR',
    'Category_Name',
    'Date',
    'Discount_Tambahan_Percentage',
    'Item_No',
    'Line_Disc_Amount_Approval_IDR',
    'Periodic_Disc_Amount_incl_PPN_IDR',
    'Periodic_Disc_CC_Percentage',
    'Periodic_Disc_Percentage',
    'Quantity',
    'Selling_Price_Incl_PPN_IDR',
    'Site_No',
    'Sub_Category_Name',
    # 'Total_Sales_Incl_PPN_IDR',
    'Unit_of_Measure',
    'Vendor_No'
]

PCT_COLUMNS = [
    'Periodic_Disc_Percentage',
    'Periodic_Disc_CC_Percentage',
    'Discount_Tambahan_Percentage'
]

COLUMN_PAIRS = [
    ('Item_No', 'Item_Name'),
    ('Site_No', 'Site_Name'),
    ('Vendor_No', 'Vendor_Name')
]

DATE_FIELDS = ['Date', 'Start_Date', 'End_Date']
TIME_CATEGORICALS = ['Month', 'Week', 'Year', 'isPromo', 'IsHoliday', 'IsPaydayWindow']
ID_COLUMNS = ['Site_No', 'Item_No', 'Vendor_No']

# Training configurations
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'objective': 'reg:squarederror',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

TRAIN_TEST_SPLIT_PARAMS = {
    'use_all_for_train': True,
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42
}

# File paths for model artifacts
MODEL_FILE = os.path.join(MODEL_DIR, "xgboost_sales_model.pkl")
METRICS_FILE = os.path.join(OUTPUT_DIR, "model_metrics.csv")
FEATURE_IMPORTANCE_PLOT = os.path.join(OUTPUT_DIR, "feature_importance.png")
PREDICTION_PLOT = os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
EVALUATION_SUMMARY = os.path.join(OUTPUT_DIR, "evaluation_summary.txt")

# Column naming and types
TARGET_COLUMN = 'Quantity'
DATE_COLUMNS = ['Date', 'Start_Date', 'End_Date']

# Data Mapper settings
DATA_MAPPER_PATH = os.path.join(PREPROCESS_DIR, "data_mapper.csv")
DATA_MAPPER_BACKUP_DIR = os.path.join(PREPROCESS_DIR, "mapper_backups")
MAPPER_REQUIRED_COLS = ['Item_No', 'Site_No']
MAPPER_OPTIONAL_COLS = ['Item_Name', 'Site_Name', 'Active', 'Last_Updated']