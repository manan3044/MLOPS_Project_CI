# File: model_training.py

import pandas as pd
import joblib
import os
import logging
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
log_file_path = os.path.join(log_dir, 'model_training.log')
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


def train_and_save_models(X_train, y_train, models_config):
    """Train multiple models based on params.yaml configuration"""
    trained_models = {}
    for model_name, config in models_config.items():
        try:
            logger.info(f"Training model: {model_name}")

            if model_name == "LinearRegression":
                model = LinearRegression()

            elif model_name == "RandomForest":
                model = RandomForestRegressor(
                    n_estimators=config.get('n_estimators', 100),
                    max_depth=config.get('max_depth', None),
                    random_state=config.get('random_state', 42)
                )

            elif model_name == "XGBRegressor":
                model = XGBRegressor(
                    n_estimators=config.get('n_estimators', 100),
                    learning_rate=config.get('learning_rate', 0.1),
                    max_depth=config.get('max_depth', 6),
                    random_state=config.get('random_state', 42),
                    objective="reg:squarederror",
                    verbosity=0
                )
            else:
                logger.warning(f"Unknown model: {model_name}, skipping...")
                continue

            model.fit(X_train, y_train)
            trained_models[model_name] = model

            os.makedirs('./models', exist_ok=True)
            joblib.dump(model, f'./models/{model_name}.joblib')
            logger.info(f"Model {model_name} saved successfully.")

        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            raise
    return trained_models


def main():
    try:
        params = load_params('params.yaml')
        train_df = pd.read_csv('./data/processed/train_fe.csv')
        X_train = train_df.drop(columns=['food_waste'])
        y_train = train_df['food_waste']

        trained_models = train_and_save_models(X_train, y_train, params['models'])
        logger.info("All models trained successfully.")
    except Exception as e:
        logger.error("Model training failed: %s", e)
        raise


if __name__ == "__main__":
    main()
