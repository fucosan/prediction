import pandas as pd

# Read the CSV file
df = pd.read_csv('data.csv')

# Extract unique pairs of Item_No and Site_No
unique_pairs = df[['Item_No', 'Site_No']].drop_duplicates()

# Save to data_mapper.csv
unique_pairs.to_csv('data_mapper.csv', index=False)

print("Unique pairs saved to data_mapper.csv")