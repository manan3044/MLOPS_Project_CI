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
    input_dir = params["data_preprocessing"]["input_dir"]

    train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(input_dir, "test.csv"))

    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    output_dir = params["data_preprocessing"]["output_dir"]
    train_df.to_csv(os.path.join(output_dir, "train_preprocessed.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_preprocessed.csv"), index=False)
    logger.info("Data preprocessing completed")

if __name__ == "__main__":
    main()
