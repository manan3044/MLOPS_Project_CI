import pandas as pd
import os
import joblib
import logging
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("model_train")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(log_dir, "model_train.log"))
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    input_dir = params["model_train"]["input_dir"]
    output_dir = params["model_train"]["output_dir"]
    models_config = params["model_train"]["models"]
    random_state = params["base"]["random_state"]

    train_df = pd.read_csv(os.path.join(input_dir, "train_fe.csv"))
    X = train_df.drop(columns=["food_waste"])
    y = train_df["food_waste"]

    os.makedirs(output_dir, exist_ok=True)

    for model_name, cfg in models_config.items():
        logger.info(f"Training {model_name}")
        if model_name == "linear_regression":
            model = LinearRegression()
        elif model_name == "random_forest":
            model = RandomForestRegressor(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                random_state=random_state
            )
        elif model_name == "xgboost":
            model = XGBRegressor(
                n_estimators=cfg["n_estimators"],
                learning_rate=cfg["learning_rate"],
                max_depth=cfg["max_depth"],
                random_state=random_state,
                objective="reg:squarederror",
                verbosity=0
            )
        else:
            continue
        model.fit(X, y)
        joblib.dump(model, cfg["file"])
        logger.info(f"Saved {model_name} to {cfg['file']}")

if __name__ == "__main__":
    main()
