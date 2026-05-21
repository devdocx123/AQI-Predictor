from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title="AQI Prediction API")

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = "models/aqi_model.pkl"

model = joblib.load(MODEL_PATH)

# -----------------------------
# INPUT SCHEMA (FIXED)
# -----------------------------
class AQIInput(BaseModel):
    pm10: float
    pm2_5: float
    carbon_monoxide: float
    nitrogen_dioxide: float
    sulphur_dioxide: float
    ozone: float
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float


# -----------------------------
# LABEL MAPPING
# -----------------------------
AQI_LABELS = {
    0: "Good",
    1: "Satisfactory",
    2: "Moderate",
    3: "Poor",
    4: "Very Poor",
    5: "Severe"
}


# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def home():
    return {"message": "AQI API running"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: AQIInput):

    features = np.array([[
        data.pm10,
        data.pm2_5,
        data.carbon_monoxide,
        data.nitrogen_dioxide,
        data.sulphur_dioxide,
        data.ozone,
        data.temperature,
        data.humidity,
        data.pressure,
        data.wind_speed
    ]])

    pred_code = int(model.predict(features)[0])

    return {
        "prediction_code": pred_code,
        "prediction_class": AQI_LABELS.get(pred_code, "Unknown")
    }