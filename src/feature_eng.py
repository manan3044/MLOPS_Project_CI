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

def main():
    params = load_params()
    train_in = params["feature_eng"]["input_train_path"]
    test_in = params["feature_eng"]["input_test_path"]
    train_out = params["feature_eng"]["output_train_path"]
    test_out = params["feature_eng"]["output_test_path"]

    train_df = pd.read_csv(train_in)
    test_df = pd.read_csv(test_in)

    # Example: add no feature engineering for now (just pass-through)
    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    logger.info(f"Feature engineered train saved to {train_out}, test saved to {test_out}")

if __name__ == "__main__":
    main()
