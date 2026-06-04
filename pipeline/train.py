import os
import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# override=False ensures GitHub Actions secrets are not overwritten by .env file
load_dotenv(override=False)

# -----------------------
# 1. CONNECT TO HOPSWORKS
# -----------------------
import hopsworks

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

if not HOPSWORKS_API_KEY:
    raise ValueError("HOPSWORKS_API_KEY environment variable is not set!")
if not HOPSWORKS_PROJECT:
    raise ValueError("HOPSWORKS_PROJECT environment variable is not set!")

print("Connecting to Hopsworks...")

project = hopsworks.login(
    api_key_value=HOPSWORKS_API_KEY,
    project=HOPSWORKS_PROJECT,
)

fs = project.get_feature_store()
print("Connected!")

# -----------------------
# 2. READ LIVE FEATURES FROM FEATURE STORE
# -----------------------
print("Reading features from Hopsworks Feature Store...")

fg = fs.get_feature_group(name="aqi_features", version=1)
df_live = fg.read()

print(f"Loaded {len(df_live)} rows from Feature Store")
print(df_live.head())

# -----------------------
# 3. LOAD STATIC DATASET FOR CLASSIFICATION TRAINING
# -----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_PATH = os.path.join(BASE_DIR, "data", "dataset", "pakistan_air_quality_final_clean.csv")

df_static = pd.read_csv(STATIC_PATH)

features = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "temperature",
    "humidity", "pressure", "wind_speed"
]

df_static = df_static[features + ["aqi_category"]].dropna()
print(f"Static dataset: {len(df_static)} rows")

# -----------------------
# 4. ENCODE TARGET
# -----------------------
le = LabelEncoder()
df_static["aqi_category"] = le.fit_transform(df_static["aqi_category"])

X = df_static[features]
y = df_static["aqi_category"]

# -----------------------
# 5. SPLIT DATA
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# 6. TRAIN MODEL
# -----------------------
print("Training model...")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Test accuracy: {score:.4f}")

# -----------------------
# 7. SAVE MODEL + ENCODER LOCALLY
# -----------------------
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

model_path   = os.path.join(BASE_DIR, "models", "aqi_model.pkl")
encoder_path = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

joblib.dump(model, model_path)
joblib.dump(le, encoder_path)

print(f"Model saved to {model_path}")

# -----------------------
# 8. REGISTER MODEL IN HOPSWORKS MODEL REGISTRY
# -----------------------
print("Registering model in Hopsworks Model Registry...")

mr = project.get_model_registry()

hw_model = mr.sklearn.create_model(
    name="aqi_classifier",
    version=1,
    metrics={"accuracy": round(score, 4)},
    description="RandomForest AQI classifier trained on Pakistan air quality data",
    input_example=X_test.iloc[0].to_dict(),
)

hw_model.save(os.path.join(BASE_DIR, "models"))

print(f"Model registered in Hopsworks! Accuracy: {score:.4f}")
print("Done!")