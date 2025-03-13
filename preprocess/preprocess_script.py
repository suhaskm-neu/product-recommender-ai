import pandas as pd

# Load the CSV from the /data folder
data_path = 'data/user_item_interactions.csv'
df = pd.read_csv(data_path)

# Inspect the column names to see if there are any extra spaces
print("Columns before cleaning:", df.columns.tolist())

# Strip any leading/trailing whitespace from column names
df.columns = df.columns.str.strip()
print("Columns after cleaning:", df.columns.tolist())

# Convert 'timestamp' to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create a target column as an example (binary: 1 if click_rate > median, else 0)
df['target'] = (df['click_rate'] > df['click_rate'].median()).astype(int)

# Save the preprocessed CSV for future use
processed_path = 'data/processed_user_item_interactions.csv'
df.to_csv(processed_path, index=False)
print(f"Processed dataset saved to {processed_path}")
