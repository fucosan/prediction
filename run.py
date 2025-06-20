import pandas as pd
from processing import process_sales_data, split_time_series_data
import joblib

# 1. Load your data
df = pd.read_csv('data.csv')

# 2. Process the data ONCE with ALL needed parameters
# Remove the first duplicate call and just use this one
processed_df, feature_cols, numerical_cols, categorical_cols, X, y, scaler, encoder = process_sales_data(
    df,
    prepare_model_data=True,
    model_type='tree',
    encoding_type='label',
    return_scaler_encoder=True
)

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

# 7. Prepare data for modeling and create a train/test split
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')
print("Scaler and encoder saved for incremental updates.")

# Single year as test set
print("\n=== Single Year Test Split ===")
X_train, X_valid, X_test, y_train, y_valid, y_test = split_time_series_data(
    processed_df, X, y, use_all_for_train=True
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

# Example of saving train/test data for modeling
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Save the feature store as parquet for incremental updates
processed_df.to_parquet('feature_store.parquet', index=False)
print("Feature store saved to 'feature_store.parquet'.")

joblib.dump(numerical_cols, 'numerical_cols.pkl')
joblib.dump(categorical_cols, 'categorical_cols.pkl')
joblib.dump(feature_cols, 'feature_cols.pkl')