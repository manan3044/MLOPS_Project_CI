import pandas as pd
import os
import joblib
import logging
import yaml
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(log_dir, "model_evaluation.log"))
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    return {
        "MAE": mean_absolute_error(y, y_pred),
        "RMSE": mean_squared_error(y, y_pred, squared=False),
        "R2": r2_score(y, y_pred)
    }

def main():
    params = load_params()
    input_dir = params["model_evaluation"]["input_dir"]
    model_dir = params["model_evaluation"]["model_dir"]
    metrics_file = params["model_evaluation"]["metrics_file"]

    test_df = pd.read_csv(os.path.join(input_dir, "test_fe.csv"))
    X_test = test_df.drop(columns=["food_waste"])
    y_test = test_df["food_waste"]

    results = {}
    for model_file in os.listdir(model_dir):
        if model_file.endswith(".joblib"):
            model_name = model_file.replace(".joblib","")
            model = joblib.load(os.path.join(model_dir, model_file))
            metrics = evaluate_model(model, X_test, y_test)
            results[model_name] = metrics
            logger.info(f"Evaluated {model_name}: {metrics}")

    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved metrics to {metrics_file}")

if __name__ == "__main__":
    main()
