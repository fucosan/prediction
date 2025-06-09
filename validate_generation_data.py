"""
Validation script to ensure proper data separation between training and test sets.
This script checks for:
1. Direct duplicates between X_train and X_test
2. Proper time-based separation
3. Data leakage through identifiers
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

def load_data():
    """Load training and test data files."""
    print("Loading data files...")
    
    required_files = ['X_train.csv', 'X_test.csv', 'processed_sales_data.csv']
    for file in required_files:
        if not os.path.exists(file):
            sys.exit(f"Error: Required file '{file}' not found.")
    
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    
    # Also load metadata if available
    if os.path.exists('X_test_metadata.csv'):
        test_metadata = pd.read_csv('X_test_metadata.csv', parse_dates=['Start_Date', 'End_Date'])
    else:
        test_metadata = None
        print("Warning: X_test_metadata.csv not found. Some validations will be skipped.")
    
    # Load processed data to access dates and identifiers
    processed_df = pd.read_csv('processed_sales_data.csv', parse_dates=['Start_Date', 'End_Date'])
    
    return X_train, X_test, processed_df, test_metadata

def check_direct_duplicates(X_train, X_test):
    """Check if any rows in X_test are exact duplicates of rows in X_train."""
    print("\n1. Checking for direct duplicates between training and test sets...")
    
    # Convert to string to ensure consistent comparison
    train_strings = X_train.astype(str).apply(lambda row: ','.join(row), axis=1)
    test_strings = X_test.astype(str).apply(lambda row: ','.join(row), axis=1)
    
    # Find duplicates
    duplicates = test_strings[test_strings.isin(train_strings)]
    
    if len(duplicates) > 0:
        print(f"❌ FAILED: Found {len(duplicates)} exact duplicate rows between training and test sets.")
        print("Example duplicate rows (indices in test set):", duplicates.index[:5].tolist())
        return False
    else:
        print("✓ PASSED: No exact duplicates found between training and test sets.")
        return True

def check_time_separation(processed_df, test_metadata):
    """Verify that test data comes after training data in time sequence."""
    print("\n2. Checking for proper time-based separation...")
    
    if test_metadata is None:
        print("⚠️ SKIPPED: Test metadata not available, cannot verify time separation.")
        return None
    
    # Extract indices from test metadata
    test_indices = test_metadata.index
    
    # Get all dates from processed data
    if 'Start_Date' in processed_df.columns and 'End_Date' in processed_df.columns:
        # Find the earliest test date
        earliest_test_date = test_metadata['Start_Date'].min()
        
        # Check if any training records end after the earliest test date
        training_indices = [i for i in processed_df.index if i not in test_indices]
        training_data = processed_df.loc[training_indices]
        
        # Find training records that overlap with test period
        overlapping = training_data[training_data['End_Date'] >= earliest_test_date]
        
        if len(overlapping) > 0:
            print(f"❌ FAILED: Found {len(overlapping)} training records that overlap with the test period.")
            print(f"Earliest test date: {earliest_test_date.date()}")
            print(f"Latest training date found: {training_data['End_Date'].max().date()}")
            return False
        else:
            print("✓ PASSED: Training data properly precedes test data in time sequence.")
            return True
    else:
        print("⚠️ SKIPPED: Date columns not found in processed data.")
        return None

def check_id_separation(processed_df, test_metadata):
    """Check if the same identifiers (Site_No, Item_No) with different dates appear in both sets."""
    print("\n3. Checking for potential data leakage through identifiers...")
    
    if test_metadata is None:
        print("⚠️ SKIPPED: Test metadata not available, cannot verify identifier separation.")
        return None
    
    # Check if we have the necessary identifier columns
    has_identifiers = 'Site_No' in processed_df.columns and 'Item_No' in processed_df.columns
    
    if not has_identifiers:
        print("⚠️ SKIPPED: Identifier columns not found in processed data.")
        return None
    
    # Extract test identifiers
    test_indices = test_metadata.index
    test_identifiers = processed_df.loc[test_indices, ['Site_No', 'Item_No']].copy()
    test_identifiers['is_test'] = True
    
    # Extract train identifiers
    train_indices = [i for i in processed_df.index if i not in test_indices]
    train_identifiers = processed_df.loc[train_indices, ['Site_No', 'Item_No']].copy()
    train_identifiers['is_test'] = False
    
    # Combine and check for duplicates
    all_identifiers = pd.concat([test_identifiers, train_identifiers])
    
    # Find identifiers that appear in both sets
    duplicated_ids = all_identifiers[all_identifiers.duplicated(subset=['Site_No', 'Item_No'], keep=False)]
    
    # Count unique pairs that appear in both sets
    shared_pairs = duplicated_ids.groupby(['Site_No', 'Item_No']).filter(
        lambda x: len(x['is_test'].unique()) > 1
    )
    unique_shared_pairs = shared_pairs[['Site_No', 'Item_No']].drop_duplicates()
    
    if len(unique_shared_pairs) > 0:
        # This is expected for time series data - we're predicting future values for same Sites/Items
        print(f"ℹ️ NOTE: Found {len(unique_shared_pairs)} Site_No/Item_No pairs that appear in both training and test sets.")
        print("This is EXPECTED for time series forecasting (training on past data to predict future values).")
        
        # Check that the dates are properly separated
        if 'End_Date' in processed_df.columns:
            print("Verifying time separation for shared identifiers...")
            shared_records = processed_df[
                processed_df.apply(
                    lambda row: (row['Site_No'], row['Item_No']) in 
                    set(zip(unique_shared_pairs['Site_No'], unique_shared_pairs['Item_No'])), 
                    axis=1
                )
            ]
            
            # Group by identifiers and check date ranges
            id_groups = shared_records.groupby(['Site_No', 'Item_No'])
            time_violations = 0
            
            for name, group in id_groups:
                test_group = group.loc[group.index.isin(test_indices)]
                train_group = group.loc[group.index.isin(train_indices)]
                
                if len(test_group) > 0 and len(train_group) > 0:
                    min_test_date = test_group['End_Date'].min()
                    max_train_date = train_group['End_Date'].max()
                    
                    if max_train_date >= min_test_date:
                        time_violations += 1
            
            if time_violations > 0:
                print(f"❌ FAILED: Found {time_violations} identifier pairs with time sequence violations.")
                return False
            else:
                print("✓ PASSED: All shared identifiers have proper time separation between train and test.")
                return True
        else:
            print("⚠️ SKIPPED: Date columns not found, cannot verify time separation for shared identifiers.")
            return None
    else:
        print("✓ PASSED: No shared identifiers found between training and test sets.")
        return True

def check_feature_distributions(X_train, X_test):
    """Compare feature distributions between train and test to identify potential issues."""
    print("\n4. Checking feature distributions...")
    
    # Get common columns
    common_cols = list(set(X_train.columns) & set(X_test.columns))
    if len(common_cols) == 0:
        print("⚠️ SKIPPED: No common columns found between training and test sets.")
        return None
    
    # Compare distributions
    distribution_issues = []
    for col in common_cols:
        try:
            train_mean = X_train[col].mean()
            test_mean = X_test[col].mean()
            
            # Check if means differ by more than 50%
            if abs(train_mean - test_mean) > 0.5 * max(abs(train_mean), abs(test_mean)):
                distribution_issues.append({
                    'column': col,
                    'train_mean': train_mean,
                    'test_mean': test_mean,
                    'difference_pct': abs(train_mean - test_mean) / max(abs(train_mean), abs(test_mean)) * 100
                })
        except:
            # Skip columns where mean calculation fails
            pass
    
    if len(distribution_issues) > 0:
        print(f"⚠️ WARNING: Found {len(distribution_issues)} features with substantially different distributions.")
        print("This is not necessarily an error, but could indicate concept drift or improper splitting.")
        
        # Display top differences
        sorted_issues = sorted(distribution_issues, key=lambda x: x['difference_pct'], reverse=True)
        for issue in sorted_issues[:5]:
            print(f"  - {issue['column']}: Train mean = {issue['train_mean']:.4f}, Test mean = {issue['test_mean']:.4f}, Diff = {issue['difference_pct']:.2f}%")
    else:
        print("✓ PASSED: Feature distributions are reasonably consistent between train and test.")
    
    return len(distribution_issues) == 0

def check_exact_record_overlap(processed_df, test_metadata):
    """
    Check if any records with identical (Site_No, Item_No, Start_Date, End_Date) appear in both train and test.
    This is a critical check to ensure the exact same observation isn't used in both sets.
    """
    print("\n1. Checking for exact record overlaps (same Site_No, Item_No, Start_Date, End_Date)...")
    
    if test_metadata is None:
        print("⚠️ SKIPPED: Test metadata not available, cannot verify exact record overlap.")
        return None
    
    # Extract test indices from metadata
    test_indices = test_metadata.index
    
    # Check if all required columns exist
    required_cols = ['Site_No', 'Item_No', 'Start_Date', 'End_Date']
    if not all(col in processed_df.columns for col in required_cols):
        print("⚠️ SKIPPED: One or more required columns not found in processed data.")
        return None
    
    # Create a unique key for each record combining all identifiers
    processed_df = processed_df.copy()
    processed_df['record_key'] = processed_df.apply(
        lambda row: f"{row['Site_No']}_{row['Item_No']}_{row['Start_Date'].strftime('%Y-%m-%d')}_{row['End_Date'].strftime('%Y-%m-%d')}", 
        axis=1
    )
    
    # Split into test and train sets
    test_records = processed_df.loc[test_indices].copy()
    train_indices = [i for i in processed_df.index if i not in test_indices]
    train_records = processed_df.loc[train_indices].copy()
    
    # Find records that appear in both sets
    test_keys = set(test_records['record_key'])
    train_keys = set(train_records['record_key'])
    overlapping_keys = test_keys.intersection(train_keys)
    
    if len(overlapping_keys) > 0:
        print(f"❌ FAILED: Found {len(overlapping_keys)} records that appear in both train and test sets.")
        print("Example overlapping records:")
        
        # Get some examples of the overlapping records
        for i, key in enumerate(list(overlapping_keys)[:3]):
            test_example = test_records[test_records['record_key'] == key].iloc[0]
            print(f"  {i+1}. Site_No: {test_example['Site_No']}, "
                  f"Item_No: {test_example['Item_No']}, "
                  f"Start_Date: {test_example['Start_Date'].strftime('%Y-%m-%d')}, "
                  f"End_Date: {test_example['End_Date'].strftime('%Y-%m-%d')}")
        
        return False
    else:
        print("✓ PASSED: No records with identical identifiers found in both train and test sets.")
        return True

def run_validation():
    """Run all validation checks."""
    print("=== DATA SPLIT VALIDATION ===")
    print(f"Starting validation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    X_train, X_test, processed_df, test_metadata = load_data()
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Run checks
    # First, check for exact record overlaps (most critical)
    exact_overlap_check = check_exact_record_overlap(processed_df, test_metadata)
    
    # Additional checks
    duplicate_check = check_direct_duplicates(X_train, X_test)
    time_check = check_time_separation(processed_df, test_metadata)
    id_check = check_id_separation(processed_df, test_metadata)
    dist_check = check_feature_distributions(X_train, X_test)
    
    # Summarize results
    print("\n=== VALIDATION SUMMARY ===")
    print(f"Exact record overlap check: {'PASS' if exact_overlap_check else 'FAIL' if exact_overlap_check is False else 'SKIPPED'}")
    print(f"Direct duplicate check: {'PASS' if duplicate_check else 'FAIL'}")
    print(f"Time separation check: {'PASS' if time_check else 'FAIL' if time_check is False else 'SKIPPED'}")
    print(f"Identifier separation check: {'PASS' if id_check else 'FAIL' if id_check is False else 'SKIPPED'}")
    print(f"Feature distribution check: {'PASS' if dist_check else 'WARNING'}")
    
    # Overall assessment - exact overlap check is most critical
    if exact_overlap_check is False:
        print("\n❌ CRITICAL FAILURE: Exact same records found in both train and test sets.")
        print("This invalidates your model evaluation as the model has seen these exact data points during training.")
    elif all(c for c in [exact_overlap_check, duplicate_check, time_check, id_check] if c is not None):
        print("\n✅ OVERALL: Validation PASSED. Training and test sets appear to be properly separated.")
    else:
        print("\n❌ OVERALL: Validation FAILED. Issues detected in the training/test split.")
        
    print(f"Validation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    run_validation()