import pandas as pd
from processing import process_sales_data, split_time_series_data

# 1. Load your data
df = pd.read_csv('data.csv')

# 2. Process the data with default parameters (without model preparation)
processed_df, feature_cols, numerical_cols, categorical_cols = process_sales_data(df)

# 3. View information about the generated features
print(f"Generated {len(feature_cols)} features:")
print(feature_cols)

# 4. View the column classifications
print(f"\nNumerical columns ({len(numerical_cols)}):")
print(numerical_cols)
print(f"\nCategorical columns ({len(categorical_cols)}):")
print(categorical_cols)

# 5. Check the processed data
print("\nProcessed data shape:", processed_df.shape)
print("\nFirst few rows of processed data:")
print(processed_df.head())

# Convert date columns to datetime before saving to ensure proper format
if 'Start_Date' in processed_df.columns:
    processed_df['Start_Date'] = pd.to_datetime(processed_df['Start_Date'])
if 'End_Date' in processed_df.columns:
    processed_df['End_Date'] = pd.to_datetime(processed_df['End_Date'])

# 6. Save the processed data
processed_df.to_csv('processed_sales_data.csv', index=False, date_format='%Y-%m-%d')
print("\nProcessed data saved to 'processed_sales_data.csv'")

# 7. Optional: Prepare data for modeling and create a train/test split
prepare_model = True  # Set to True to prepare data for modeling
if prepare_model:
    # Process with model preparation
    processed_df, feature_cols, numerical_cols, categorical_cols, X, y = process_sales_data(
        df, 
        prepare_model_data=True,
        model_type='tree',  # Options: 'tree' or 'linear'
        encoding_type='label'  # Options: 'label' or 'onehot'
    )
    
    # Create time-based train/test split using fixed month windows
    # The last 2 months will be used as test set as specified in the requirements
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_time_series_data(
        processed_df, X, y, 
        test_months=2,  # Use last 2 months as test set
        valid_months=1  # Optional: Use 1 month before test set as validation
    )
    
    # Save test set metadata to CSV - use the indices from y_test to get metadata
    test_indices = y_test.index
    X_test_metadata = pd.DataFrame()
    X_test_metadata['Site_No'] = processed_df.loc[test_indices, 'Site_No'].values
    X_test_metadata['Item_No'] = processed_df.loc[test_indices, 'Item_No'].values
    if 'Site_Name' in processed_df.columns:
        X_test_metadata['Site_Name'] = processed_df.loc[test_indices, 'Site_Name'].values
    if 'Item_Name' in processed_df.columns:
        X_test_metadata['Item_Name'] = processed_df.loc[test_indices, 'Item_Name'].values
    X_test_metadata['Start_Date'] = processed_df.loc[test_indices, 'Start_Date'].values
    X_test_metadata['End_Date'] = processed_df.loc[test_indices, 'End_Date'].values
    X_test_metadata.to_csv('X_test_metadata.csv', index=False)
    
    print("\nModel data preparation complete.")
    print(f"X shape: {X.shape}, contains features: {X.columns.tolist()[:5]}...")
    
    # Example of saving train/test data for modeling
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)