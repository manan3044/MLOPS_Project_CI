import pandas as pd
import os
import logging
import yaml
from sklearn.model_selection import train_test_split

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s -')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading params.yaml: %s', e)
        raise


def load_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.debug('Data loaded from %s', data_path)
        return df
    except Exception as e:
        logger.error('Error loading CSV file: %s', e)
        raise


def save_split_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str):
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        logger.debug('Train and test data saved to %s', output_dir)
    except Exception as e:
        logger.error('Error saving train/test data: %s', e)
        raise


def main():
    try:
        params = load_params('params.yaml')
        data_path = params['data_ingestion']['input_path']
        test_size = params['data_ingestion']['test_size']
        random_state = params['base']['random_state']

        df = load_data(data_path)
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        save_split_data(train_df, test_df, './data/processed')
        logger.info('Data ingestion completed successfully.')
    except Exception as e:
        logger.error('Data ingestion failed: %s', e)
        raise


if __name__ == "__main__":
    main()
