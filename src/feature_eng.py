import pandas as pd
import os
import logging
import yaml

# Logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(log_dir, "feature_engineering.log"))
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def add_features(df):
    if "sales" in df.columns:
        df["sales_7day_avg"] = df["sales"].rolling(7, min_periods=1).mean()
    if "day_of_week" in df.columns:
        df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    if "temperature" in df.columns and "local_event" in df.columns:
        df["temp_event_interaction"] = df["temperature"] * df["local_event"]
    return df

def main():
    params = load_params()
    input_dir = params["feature_engineering"]["input_dir"]

    train_df = pd.read_csv(os.path.join(input_dir, "train_preprocessed.csv"))
    test_df = pd.read_csv(os.path.join(input_dir, "test_preprocessed.csv"))

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    output_dir = params["feature_engineering"]["output_dir"]
    train_df.to_csv(os.path.join(output_dir, "train_fe.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_fe.csv"), index=False)
    logger.info("Feature engineering completed")

if __name__ == "__main__":
    main()
