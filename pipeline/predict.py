import joblib
import numpy as np

# Load model + encoder
model = joblib.load("../models/aqi_model.pkl")
le = joblib.load("../models/label_encoder.pkl")

def predict_aqi(sample):
    """
    sample = [pm10, pm2_5, co, no2, so2, o3, temp, humidity, pressure, wind_speed]
    """

    sample = np.array(sample).reshape(1, -1)

    pred = model.predict(sample)[0]
    label = le.inverse_transform([pred])[0]

    return label

# Example test
if __name__ == "__main__":
    test_sample = [30, 25, 500, 20, 5, 60, 25, 70, 1012, 3]

    result = predict_aqi(test_sample)
    print("Predicted AQI Category:", result)