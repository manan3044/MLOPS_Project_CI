import pandas as pd
import yaml
import os

params = yaml.safe_load(open("params.yaml"))["data_preprocessing"]

input_path = params["input_path"]
output_path = params["output_path"]

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load data
df = pd.read_csv(input_path)

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Drop duplicates
df.drop_duplicates(inplace=True)

df.to_csv(output_path, index=False)
print(f"✅ Preprocessed data saved to {output_path}")
