# File: feature_engineering.py

import pandas as pd
import os
import logging

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s -')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['sales_7day_avg'] = df['sales'].rolling(window=7, min_periods=1).mean()
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['temp_event_interaction'] = df['temperature'] * df['local_event']
        logger.debug('Feature engineering completed')
        return df
    except Exception as e:
        logger.error('Feature engineering error: %s', e)
        raise


def main():
    try:
        train_df = pd.read_csv('./data/processed/train_preprocessed.csv')
        test_df = pd.read_csv('./data/processed/test_preprocessed.csv')
        train_df = add_features(train_df)
        test_df = add_features(test_df)
        train_df.to_csv('./data/processed/train_fe.csv', index=False)
        test_df.to_csv('./data/processed/test_fe.csv', index=False)
        logger.info('Feature engineering completed successfully.')
    except Exception as e:
        logger.error('Feature engineering failed: %s', e)
        raise


if __name__ == "__main__":
    main()
