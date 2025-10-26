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
    train_out = params["data_ingestion"]["output_train_path"]
    test_out = params["data_ingestion"]["output_test_path"]
    test_size = params["data_ingestion"]["test_size"]
    random_state = params["base"]["random_state"]

    # Load data
    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Ensure directory exists
    os.makedirs(os.path.dirname(train_out), exist_ok=True)

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    logger.info("Data ingestion completed successfully.")
    logger.debug(f"Train saved to {train_out}, Test saved to {test_out}")


if __name__ == "__main__":
    main()
