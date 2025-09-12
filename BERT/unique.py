import pandas as pd

input_file = "darkweb_data.csv"
output_file = "new_darkweb_data.csv"

# Read CSV with flexible parsing
df = pd.read_csv(input_file, on_bad_lines="skip", engine="python")  # Skips problematic rows

# Drop duplicate rows
df_cleaned = df.drop_duplicates()

# Save back
df_cleaned.to_csv(output_file, index=False)

print(f"✅ Cleaned file saved as '{output_file}' with {len(df_cleaned)} unique rows.")
