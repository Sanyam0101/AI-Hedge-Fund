from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import csv
from datetime import datetime
import pandas as pd

app = FastAPI()

# ---------------------
# ML MODEL (simple demo)
# ---------------------
class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = float(np.mean(X))  # demo model

    # ---------------------
    # LOG PREDICTIONS
    # ---------------------
    with open("predictions_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), data.features, prediction])

    return {"predicted_return": prediction}


@app.get("/simulate_trading")
def simulate_trading():
    try:
        df = pd.read_csv("predictions_log.csv", header=None, names=["Timestamp", "Features", "Predicted Return"])
    except FileNotFoundError:
        return {"message": "No data to simulate."}

    balance = 1000  # starting capital
    for _, row in df.iterrows():
        ret = row["Predicted Return"]
        if ret > 1.5:  # Buy condition
            balance *= 1.02
        else:
            balance *= 0.998  # small loss

    return {"final_balance": round(balance, 2), "trades": len(df)}