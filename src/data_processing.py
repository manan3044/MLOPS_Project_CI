# File: data_preprocessing.py

import pandas as pd
import os
import logging
import yaml
from sklearn.preprocessing import LabelEncoder

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s -')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    with open(params_path, 'r') as file:
        return yaml.safe_load(file)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Handle missing values
        df.fillna(method='ffill', inplace=True)
        # Encode categorical variables
        le = LabelEncoder()
        df['day_of_week'] = le.fit_transform(df['day_of_week'])
        logger.debug('Preprocessing complete')
        return df
    except Exception as e:
        logger.error('Error during preprocessing: %s', e)
        raise


def main():
    try:
        train_df = pd.read_csv('./data/processed/train.csv')
        test_df = pd.read_csv('./data/processed/test.csv')
        train_df = preprocess(train_df)
        test_df = preprocess(test_df)
        os.makedirs('./data/processed', exist_ok=True)
        train_df.to_csv('./data/processed/train_preprocessed.csv', index=False)
        test_df.to_csv('./data/processed/test_preprocessed.csv', index=False)
        logger.info('Data preprocessing completed successfully.')
    except Exception as e:
        logger.error('Data preprocessing failed: %s', e)
        raise


if __name__ == "__main__":
    main()
