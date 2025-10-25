# File: src/model_train.py

import pandas as pd
import joblib
import os
import logging
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('model_training')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
log_file_path = os.path.join(log_dir, 'model_training.log')
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


def train_and_save_models(X_train, y_train, models_config, output_dir):
    """Train multiple models based on params.yaml configuration"""
    trained_models = {}
    os.makedirs(output_dir, exist_ok=True)

    for model_name, config in models_config.items():
        try:
            logger.info(f"Training model: {model_name}")

            if model_name.lower() == "linearregression":
                model = LinearRegression()

            elif model_name.lower() == "randomforest":
                model = RandomForestRegressor(
                    n_estimators=config.get('n_estimators', 100),
                    max_depth=config.get('max_depth', None),
                    random_state=config.get('random_state', 42)
                )

            elif model_name.lower() == "xgbregressor":
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

            # Save the model
            model_file = os.path.join(output_dir, f"{model_name}.joblib")
            joblib.dump(model, model_file)
            logger.info(f"Model {model_name} saved to {model_file}")

        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            raise

    return trained_models


def main():
    try:
        params = load_params("params.yaml")
        input_path = params['model_training']['input_path']
        output_dir = params['model_training']['model_output_dir']
        models_config = params['models']

        df = pd.read_csv(input_path)
        X_train = df.drop(columns=['food_waste'])
        y_train = df['food_waste']

        train_and_save_models(X_train, y_train, models_config, output_dir)
        logger.info("All models trained successfully.")

    except Exception as e:
        logger.error("Model training failed: %s", e)
        raise


if __name__ == "__main__":
    main()
