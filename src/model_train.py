import pandas as pd
import yaml
import joblib
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

params = yaml.safe_load(open("params.yaml"))["model_training"]

input_path = params["input_path"]
model_output = params["model_output"]
os.makedirs(os.path.dirname(model_output), exist_ok=True)

# Load processed data
data = pd.read_csv(input_path)

X = data.drop(columns=["waste", "date"])
y = data["waste"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"]
)

model = XGBRegressor(
    n_estimators=params["n_estimators"],
    learning_rate=params["learning_rate"],
    max_depth=params["max_depth"],
    random_state=params["random_state"]
)

model.fit(X_train, y_train)

joblib.dump(model, model_output)
print(f"✅ Model trained and saved to {model_output}")
