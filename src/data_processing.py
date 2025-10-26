import pandas as pd
import os
import logging
import yaml
from sklearn.preprocessing import LabelEncoder

# Logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(log_dir, "data_preprocessing.log"))
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def preprocess(df):
    df.fillna(method="ffill", inplace=True)
    if "day_of_week" in df.columns:
        df["day_of_week"] = LabelEncoder().fit_transform(df["day_of_week"])
    return df

def main():
    params = load_params()
    train_in = params["data_preprocessing"]["input_train_path"]
    test_in = params["data_preprocessing"]["input_test_path"]
    train_out = params["data_preprocessing"]["output_train_path"]
    test_out = params["data_preprocessing"]["output_test_path"]

    # Read data
    train_df = pd.read_csv(train_in)
    test_df = pd.read_csv(test_in)
    logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")

    # Preprocess
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    # Save processed files
    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    logger.info(f"Saved preprocessed train to {train_out}, test to {test_out}")
    logger.info("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()
