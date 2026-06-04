import pandas as pd
import numpy as np
import hopsworks
import os
import tempfile
from dotenv import load_dotenv

# override=False ensures GitHub Actions secrets are not overwritten by .env file
load_dotenv(override=False)

os.environ["TMPDIR"] = tempfile.gettempdir()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

if not HOPSWORKS_API_KEY:
    raise ValueError("HOPSWORKS_API_KEY environment variable is not set!")
if not HOPSWORKS_PROJECT:
    raise ValueError("HOPSWORKS_PROJECT environment variable is not set!")

print(f"PROJECT: '{HOPSWORKS_PROJECT}'")
print(f"KEY SET: {bool(HOPSWORKS_API_KEY)}")

# -----------------------------
# 1. LOAD RAW DATA
# -----------------------------
file_path = "data/raw/raw_aqi_data.csv"
df = pd.read_csv(file_path)
print("Raw data loaded successfully!")
print(df.head())

# -----------------------------
# 2. CONVERT DATETIME
# -----------------------------
df["datetime"] = pd.to_datetime(df["datetime"])

# -----------------------------
# 3. HANDLE MISSING VALUES
# -----------------------------
df = df.ffill()

# -----------------------------
# 4. CREATE TIME FEATURES
# -----------------------------
df["hour"]    = df["datetime"].dt.hour
df["day"]     = df["datetime"].dt.day
df["month"]   = df["datetime"].dt.month
df["weekday"] = df["datetime"].dt.weekday

# -----------------------------
# 5. CREATE LAG FEATURES
# -----------------------------
df["aqi_lag_1"] = df["aqi"].shift(1)
df["aqi_lag_2"] = df["aqi"].shift(2)
df["aqi_lag_3"] = df["aqi"].shift(3)

# -----------------------------
# 6. ROLLING AVERAGE FEATURES
# -----------------------------
df["aqi_rolling_mean_3"] = df["aqi"].rolling(window=3).mean()

# -----------------------------
# 7. AQI CHANGE RATE
# -----------------------------
df["aqi_change"] = df["aqi"].diff()

# -----------------------------
# 8. REMOVE EMPTY ROWS
# -----------------------------
df = df.dropna()

# -----------------------------
# 9. PREPARE FOR HOPSWORKS
# -----------------------------
df = df.rename(columns={"datetime": "event_time"})

df["aqi"]                = df["aqi"].astype(float)
df["pm25"]               = df["pm25"].astype(float)
df["pm10"]               = df["pm10"].astype(float)
df["temperature"]        = df["temperature"].astype(float)
df["humidity"]           = df["humidity"].astype(float)
df["hour"]               = df["hour"].astype(int)
df["day"]                = df["day"].astype(int)
df["month"]              = df["month"].astype(int)
df["weekday"]            = df["weekday"].astype(int)
df["aqi_lag_1"]          = df["aqi_lag_1"].astype(float)
df["aqi_lag_2"]          = df["aqi_lag_2"].astype(float)
df["aqi_lag_3"]          = df["aqi_lag_3"].astype(float)
df["aqi_rolling_mean_3"] = df["aqi_rolling_mean_3"].astype(float)
df["aqi_change"]         = df["aqi_change"].astype(float)

print("\nFeature engineering completed!")
print(df.dtypes)
print(df.head())

# -----------------------------
# 10. CONNECT TO HOPSWORKS
# -----------------------------
print("\nConnecting to Hopsworks...")

project = hopsworks.login(
    api_key_value=HOPSWORKS_API_KEY,
    project=HOPSWORKS_PROJECT,
)

fs = project.get_feature_store()
print("Connected to Hopsworks Feature Store!")

# -----------------------------
# 11. UPLOAD VIA DATASET API (no Kafka needed)
# -----------------------------
import io

print("\nUploading features via dataset API...")

file_name = f"aqi_features_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet"
tmp_path  = os.path.join(tempfile.gettempdir(), file_name)
df.to_parquet(tmp_path, index=False)

dataset_api = project.get_dataset_api()

try:
    dataset_api.mkdir("Resources/aqi_features")
except Exception:
    pass  # directory already exists

dataset_api.upload(tmp_path, "Resources/aqi_features", overwrite=True)

print(f"\nFeatures successfully uploaded to Hopsworks!")
print(f"Path: Resources/aqi_features/{file_name}")
print(f"Rows uploaded: {len(df)}")