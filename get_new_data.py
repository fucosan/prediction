import pandas as pd

# Load the data_mapper.csv containing valid Item_No and Site_No pairs
mapper = pd.read_csv('data_mapper.csv')

# Load the new_data.xlsx file
new_data = pd.read_excel('new_data.xlsx')

# Merge to keep only rows where (Item_No, Site_No) pairs exist in data_mapper.csv
filtered_data = new_data.merge(mapper, on=['Item_No', 'Site_No'], how='inner')

# Save the filtered rows to new_raw_data.csv
filtered_data.to_csv('new_raw_data.csv', index=False)

print("Filtered data saved to new_raw_data.csv")

# --- Compare columns between data.csv and new_raw_data.csv ---

# Load data.csv and new_raw_data.csv
data = pd.read_csv('data.csv')
new_raw_data = pd.read_csv('new_raw_data.csv')

# Get columns in data.csv but not in new_raw_data.csv
data_only_cols = sorted(set(data.columns) - set(new_raw_data.columns))
# Get columns in new_raw_data.csv but not in data.csv
new_raw_only_cols = sorted(set(new_raw_data.columns) - set(data.columns))
# Get common columns
common_cols = sorted(set(data.columns) & set(new_raw_data.columns))

print("\nColumns in data.csv but NOT in new_raw_data.csv:")
for col in data_only_cols:
    print(f"  - {col}")

print("\nColumns in new_raw_data.csv but NOT in data.csv:")
for col in new_raw_only_cols:
    print(f"  - {col}")

print("\nCommon columns:")
for col in common_cols:
    print(f"  - {col}")