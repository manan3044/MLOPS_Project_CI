# File: src/model_eval.py

import pandas as pd
import joblib
import logging
import os
import json
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
log_file_path = os.path.join(log_dir, 'model_evaluation.log')
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


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def main():
    try:
        params = load_params('params.yaml')
        data_path = params['model_evaluation']['data_path']
        models_dir = params['model_evaluation']['model_path']
        metrics_output = params['model_evaluation']['metrics_output']

        df = pd.read_csv(data_path)
        X_test = df.drop(columns=['food_waste'])
        y_test = df['food_waste']

        results = {}
        for model_file in os.listdir(models_dir):
            if model_file.endswith('.pkl') or model_file.endswith('.joblib'):
                model_name = model_file.rsplit('.', 1)[0]
                model = joblib.load(os.path.join(models_dir, model_file))
                logger.info(f"Evaluating {model_name}...")
                metrics = evaluate_model(model, X_test, y_test)
                results[model_name] = metrics

        os.makedirs(os.path.dirname(metrics_output), exist_ok=True)
        with open(metrics_output, 'w') as f:
            json.dump(results, f, indent=4)

        logger.info("All model evaluations completed successfully.")
        logger.info(f"Metrics saved to {metrics_output}")
        logger.debug(results)

    except Exception as e:
        logger.error("Model evaluation failed: %s", e)
        raise


if __name__ == "__main__":
    main()
