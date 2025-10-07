import pandas as pd
import os
import yaml

params = yaml.safe_load(open("params.yaml"))["data_ingestion"]

input_path = params["input_path"]
output_path = params["output_path"]

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Just copy the dataset (synthetic already generated)
data = pd.read_csv(input_path)
data.to_csv(output_path, index=False)

print(f"✅ Data ingested and saved to {output_path}")
