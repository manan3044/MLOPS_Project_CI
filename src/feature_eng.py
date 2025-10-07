import pandas as pd
import yaml
import os

params = yaml.safe_load(open("params.yaml"))["feature_engineering"]

input_path = params["input_path"]
output_path = params["output_path"]

os.makedirs(os.path.dirname(output_path), exist_ok=True)

df = pd.read_csv(input_path)

# Encode categorical variables
df['day_of_week'] = df['day_of_week'].astype('category').cat.codes
df['city'] = df['city'].astype('category').cat.codes

df.to_csv(output_path, index=False)
print(f"✅ Feature engineered data saved to {output_path}")
