import pandas as pd
import numpy as np
import hopsworks
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

os.environ["TMPDIR"] = tempfile.gettempdir()

print(f"PROJECT: '{os.getenv('HOPSWORKS_PROJECT')}'")
print(f"KEY SET: {bool(os.getenv('HOPSWORKS_API_KEY'))}")

# -----------------------------
# 1. LOAD RAW DATA
# -----------------------------
file_path = "data/raw/raw_aqi_data.csv"
df = pd.read_csv(file_path)
print("Raw data loaded successfully!")
print(df.head())

# -----------------------------
# 2-8. FEATURE ENGINEERING
# -----------------------------
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.ffill()

df["hour"]    = df["datetime"].dt.hour
df["day"]     = df["datetime"].dt.day
df["month"]   = df["datetime"].dt.month
df["weekday"] = df["datetime"].dt.weekday

df["aqi_lag_1"] = df["aqi"].shift(1)
df["aqi_lag_2"] = df["aqi"].shift(2)
df["aqi_lag_3"] = df["aqi"].shift(3)

df["aqi_rolling_mean_3"] = df["aqi"].rolling(window=3).mean()
df["aqi_change"] = df["aqi"].diff()
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
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

print("\nConnecting to Hopsworks...")

project = hopsworks.login(
    api_key_value=HOPSWORKS_API_KEY,
    project=HOPSWORKS_PROJECT,
)

fs = project.get_feature_store()
print("Connected to Hopsworks Feature Store!")

# -----------------------------
# 11. SAVE AS PARQUET TO HOPSWORKS FILE SYSTEM
#     Bypasses Kafka entirely — writes directly via REST upload
# -----------------------------

import uuid
import io

print("\nUploading features via dataset API (no Kafka)...")

# Save df to parquet in memory
parquet_buffer = io.BytesIO()
df.to_parquet(parquet_buffer, index=False)
parquet_buffer.seek(0)

# Upload to Hopsworks project dataset
dataset_api = project.get_dataset_api()

file_name = f"aqi_features_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet"
upload_path = f"Resources/aqi_features/{file_name}"

# Ensure directory exists
try:
    dataset_api.mkdir("Resources/aqi_features")
except Exception:
    pass  # Directory already exists

# Write parquet to temp file then upload
tmp_path = os.path.join(tempfile.gettempdir(), file_name)
df.to_parquet(tmp_path, index=False)

dataset_api.upload(tmp_path, "Resources/aqi_features", overwrite=True)

print(f"\nFeatures successfully uploaded to Hopsworks!")
print(f"Path: Resources/aqi_features/{file_name}")
print(f"Rows uploaded: {len(df)}")
print(f"\nView your files at: https://eu-west.cloud.hopsworks.ai/p/33071/datasets")