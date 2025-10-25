# File: src/feature_eng.py

import pandas as pd
import os
import logging
import yaml

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
log_file_path = os.path.join(log_dir, 'feature_engineering.log')
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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['sales_7day_avg'] = df['sales'].rolling(window=7, min_periods=1).mean()
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        if 'temperature' in df.columns and 'local_event' in df.columns:
            df['temp_event_interaction'] = df['temperature'] * df['local_event']
        logger.debug('Feature engineering completed')
        return df
    except Exception as e:
        logger.error('Feature engineering error: %s', e)
        raise


def main():
    try:
        params = load_params('params.yaml')
        input_path = params['feature_engineering']['input_path']
        output_path = params['feature_engineering']['output_path']

        df = pd.read_csv(input_path)
        df = add_features(df)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info('Feature engineering completed successfully: %s', output_path)

    except Exception as e:
        logger.error('Feature engineering failed: %s', e)
        raise


if __name__ == "__main__":
    main()
