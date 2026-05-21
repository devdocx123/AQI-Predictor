import pandas as pd
import numpy as np

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

df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
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
# 9. SAVE PROCESSED DATA
# -----------------------------
output_path = "data/processed/processed_aqi_data.csv"
df.to_csv(output_path, index=False)

print("\nFeature engineering completed!")
print(df)