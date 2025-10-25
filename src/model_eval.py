# File: model_evaluation.py

import pandas as pd
import joblib
import logging
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s -')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def main():
    try:
        test_df = pd.read_csv('./data/processed/test_fe.csv')
        X_test = test_df.drop(columns=['food_waste'])
        y_test = test_df['food_waste']

        results = {}
        for model_file in os.listdir('./models'):
            if model_file.endswith('.joblib'):
                model_name = model_file.replace('.joblib', '')
                model = joblib.load(os.path.join('./models', model_file))
                logger.info(f"Evaluating {model_name}...")
                metrics = evaluate_model(model, X_test, y_test)
                results[model_name] = metrics

        with open('metrics.json', 'w') as f:
            json.dump(results, f, indent=4)

        logger.info("All model evaluations completed.")
        logger.info(f"Metrics saved to metrics.json")
        logger.debug(results)

    except Exception as e:
        logger.error("Model evaluation failed: %s", e)
        raise


if __name__ == "__main__":
    main()
