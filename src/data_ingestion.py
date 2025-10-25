import pandas as pd
import os
import logging
import yaml
from sklearn.model_selection import train_test_split

# Logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(log_dir, "data_ingestion.log"))
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    data_path = params["data_ingestion"]["input_path"]
    output_dir = params["data_ingestion"]["output_dir"]
    test_size = params["data_ingestion"]["test_size"]

    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=params["base"]["random_state"])

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    logger.info("Data ingestion completed")

if __name__ == "__main__":
    main()
