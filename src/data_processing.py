# File: src/data_processing.py

import pandas as pd
import os
import logging
import yaml
from sklearn.preprocessing import LabelEncoder

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s -')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from YAML file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters loaded from %s", params_path)
        return params
    except Exception as e:
        logger.error("Error loading params.yaml: %s", e)
        raise


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Handle missing values
        df.ffill(inplace=True)
        # Encode categorical variables
        if 'day_of_week' in df.columns:
            le = LabelEncoder()
            df['day_of_week'] = le.fit_transform(df['day_of_week'])
        logger.debug("Preprocessing complete")
        return df
    except Exception as e:
        logger.error("Error during preprocessing: %s", e)
        raise


def main():
    try:
        params = load_params('params.yaml')
        input_path = params['data_preprocessing']['input_path']
        output_path = params['data_preprocessing']['output_path']

        # Read the input CSV (ingestion output)
        df = pd.read_csv(input_path)
        df = preprocess(df)

        # Save processed data to output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Data preprocessing completed successfully: %s", output_path)

    except Exception as e:
        logger.error("Data preprocessing failed: %s", e)
        raise


if __name__ == "__main__":
    main()
