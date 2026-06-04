import requests
import pandas as pd
from datetime import datetime
import os

# ----------------------------
# 1. CONFIGURATION
# ----------------------------

API_TOKEN = os.getenv("WAQI_API_TOKEN", "93df09c9ac87115018acb73eccba893684d285d5")
CITY = "Islamabad"

BASE_URL = f"https://api.waqi.info/feed/{CITY}/?token={API_TOKEN}"

# ----------------------------
# 2. FETCH DATA FROM API
# ----------------------------

def fetch_aqi_data():
    try:
        response = requests.get(BASE_URL, timeout=10)
        data = response.json()

        if data["status"] != "ok":
            print("Invalid response from API")
            return None

        return data

    except requests.exceptions.Timeout:
        print("Request timed out!")
        return None

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# ----------------------------
# 3. PARSE DATA
# ----------------------------

def parse_data(data):
    iaqi = data["data"]["iaqi"]

    record = {
        "datetime": datetime.now(),
        "city": CITY,
        "aqi": data["data"].get("aqi"),
        "pm25": iaqi.get("pm25", {}).get("v", 0),
        "pm10": iaqi.get("pm10", {}).get("v", 0),
        "temperature": iaqi.get("t", {}).get("v", 0),
        "humidity": iaqi.get("h", {}).get("v", 0),
    }

    return record

# ----------------------------
# 4. SAVE TO CSV (raw buffer)
#    Raw CSV is kept as a local staging layer.
#    features.py reads this and pushes engineered
#    features to the Hopsworks Feature Store.
# ----------------------------

def save_data(record):
    df = pd.DataFrame([record])

    file_path = "data/raw/raw_aqi_data.csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    file_exists = os.path.exists(file_path)

    df.to_csv(
        file_path,
        mode='a',
        header=not file_exists,
        index=False
    )

    print("Raw data saved to CSV staging buffer!")

# ----------------------------
# 5. MAIN FUNCTION
# ----------------------------

def main():
    print("Fetching AQI data...")

    data = fetch_aqi_data()

    if data:
        record = parse_data(data)
        save_data(record)
        print("Done:", record)

if __name__ == "__main__":
    main()
