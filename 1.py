import pandas as pd

# Read the two CSV files
df_phishing = pd.read_csv("Phishing URLs.csv")
df_mixed = pd.read_csv("URL dataset.csv")

# 1) Rename columns to a consistent 'type'
df_phishing.rename(columns={"Type": "type"}, inplace=True)

# The second file might already have 'type' in lowercase, but let's ensure consistency
if "Type" in df_mixed.columns:
    df_mixed.rename(columns={"Type": "type"}, inplace=True)

# 2) Concatenate both DataFrames into one
df_unified = pd.concat([df_phishing, df_mixed], ignore_index=True)

# 3) Check for duplicates based on the 'url' column
initial_length = len(df_unified)
df_unified.drop_duplicates(subset="url", inplace=True)
num_duplicates = initial_length - len(df_unified)
print(f"Number of duplicate URLs removed: {num_duplicates}")

# 4) Encode labels: phishing -> 1, legitimate -> 0
df_unified["type"] = df_unified["type"].str.lower().map({
    "phishing": 1,
    "legitimate": 0
})

# (Optional) Check if there are any null values after mapping
print("Number of rows with unmapped labels:",
      df_unified["type"].isna().sum())

# 5) Print some basic info
print("Final dataset shape:", df_unified.shape)
print("Class distribution:\n", df_unified["type"].value_counts())

# (Optional) Save the unified dataset to a new CSV
df_unified.to_csv("unified_dataset.csv", index=False)
