import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------
# 1. LOAD DATA
# -----------------------
import os


os.makedirs("../models", exist_ok=True)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset", "pakistan_air_quality_final_clean.csv")

df = pd.read_csv(DATA_PATH)

# -----------------------
# 2. SELECT FEATURES
# -----------------------
features = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "temperature",
    "humidity", "pressure", "wind_speed"
]

df = df[features + ["aqi_category"]].dropna()

# -----------------------
# 3. ENCODE TARGET
# -----------------------
le = LabelEncoder()
df["aqi_category"] = le.fit_transform(df["aqi_category"])

X = df[features]
y = df["aqi_category"]

# -----------------------
# 4. SPLIT DATA
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# 5. TRAIN MODEL
# -----------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------
# 6. SAVE MODEL + ENCODER
# -----------------------
joblib.dump(model, "../models/aqi_model.pkl")
joblib.dump(le, "../models/label_encoder.pkl")

print("Model training completed and saved!")