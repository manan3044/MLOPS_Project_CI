import pandas as pd
import joblib
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

params = yaml.safe_load(open("params.yaml"))["model_evaluation"]

model_path = params["model_path"]
data_path = params["data_path"]
metrics_output = params["metrics_output"]

data = pd.read_csv(data_path)
model = joblib.load(model_path)

X = data.drop(columns=["waste", "date"])
y = data["waste"]

# Split same as before
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

with open(metrics_output, "w") as f:
    f.write(f"MAE: {mae}\nRMSE: {rmse}\nR2: {r2}")

print(f"✅ Evaluation complete.\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
