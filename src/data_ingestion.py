import pandas as pd
import os
import logging
import yaml
from sklearn.model_selection import train_test_split

# Logging setup
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

    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")

    # Split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=params["base"]["random_state"]
    )

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    train_out = os.path.join(output_dir, "train.csv")
    test_out = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    logger.info(f"Train saved to {train_out}, Test saved to {test_out}")
    logger.info("Data ingestion completed successfully!")

if __name__ == "__main__":
    main()
