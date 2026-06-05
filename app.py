import os
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

load_dotenv(override=False)

import hopsworks

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

if not HOPSWORKS_API_KEY:
    raise ValueError("HOPSWORKS_API_KEY environment variable is not set!")
if not HOPSWORKS_PROJECT:
    raise ValueError("HOPSWORKS_PROJECT environment variable is not set!")

# -----------------------
# 1. CONNECT TO HOPSWORKS
# -----------------------
print("Connecting to Hopsworks...")
project = hopsworks.login(
    api_key_value=HOPSWORKS_API_KEY,
    project=HOPSWORKS_PROJECT,
)
fs = project.get_feature_store()
print("Connected!")

# -----------------------
# 2. LOAD HISTORICAL DATASET
# -----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_PATH = os.path.join(BASE_DIR, "data", "dataset", "pakistan_air_quality_final_clean.csv")

df = pd.read_csv(STATIC_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"Dataset loaded: {df.shape}")

# Features — only raw pollutant/weather readings, no derived/leaky columns
features = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "temperature",
    "humidity", "pressure", "wind_speed"
]

df = df[["timestamp"] + features + ["aqi_category"]].dropna()

# -----------------------
# 3. TIME-BASED TRAIN/TEST SPLIT
#    Train on first 80% of time, test on last 20%
#    This prevents data leakage from future → past
# -----------------------
split_idx = int(len(df) * 0.8)
df_train  = df.iloc[:split_idx]
df_test   = df.iloc[split_idx:]

print(f"Train: {len(df_train)} rows ({df_train['timestamp'].min()} → {df_train['timestamp'].max()})")
print(f"Test:  {len(df_test)} rows ({df_test['timestamp'].min()} → {df_test['timestamp'].max()})")

# -----------------------
# 4. ENCODE TARGET
# -----------------------
le = LabelEncoder()
le.fit(df["aqi_category"])

X_train = df_train[features]
y_train = le.transform(df_train["aqi_category"])
X_test  = df_test[features]
y_test  = le.transform(df_test["aqi_category"])

print(f"Classes: {list(le.classes_)}")

# -----------------------
# 5. TRAIN MULTIPLE MODELS
# -----------------------
models = {
    "RandomForest":      RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "GradientBoosting":  GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
}

results = {}
best_model_name = None
best_score = -1
best_model = None

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    results[name] = {"accuracy": acc, "rmse": rmse, "mae": mae, "r2": r2}

    print(f"  Accuracy : {acc:.4f}")
    print(f"  RMSE     : {rmse:.4f}")
    print(f"  MAE      : {mae:.4f}")
    print(f"  R²       : {r2:.4f}")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        zero_division=0
    ))

    if acc > best_score:
        best_score = acc
        best_model_name = name
        best_model = model

print(f"\nBest model: {best_model_name} (accuracy={best_score:.4f})")

# -----------------------
# 6. SHAP FEATURE IMPORTANCE
# -----------------------
try:
    import shap
    import matplotlib.pyplot as plt
    print("\nComputing SHAP values...")
    rf_model = models["RandomForest"]
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test.iloc[:200])
    shap.summary_plot(shap_values, X_test.iloc[:200], feature_names=features, show=False)
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    plt.savefig(os.path.join(BASE_DIR, "models", "shap_summary.png"), bbox_inches="tight")
    plt.close()
    print("SHAP plot saved!")
except Exception as e:
    print(f"SHAP skipped: {e}")

# -----------------------
# 7. SAVE LOCALLY
# -----------------------
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

joblib.dump(best_model, os.path.join(BASE_DIR, "models", "aqi_model.pkl"))
joblib.dump(le,         os.path.join(BASE_DIR, "models", "label_encoder.pkl"))
joblib.dump(features,   os.path.join(BASE_DIR, "models", "feature_names.pkl"))

print("Model saved locally!")

# -----------------------
# 8. REGISTER IN HOPSWORKS MODEL REGISTRY
#    Auto-increment version to avoid conflicts
# -----------------------
print("\nRegistering in Hopsworks Model Registry...")
mr = project.get_model_registry()

best_metrics = results[best_model_name]

# Find next available version
for version in range(1, 20):
    try:
        hw_model = mr.sklearn.create_model(
            name="aqi_classifier",
            version=version,
            metrics={
                "accuracy": round(best_metrics["accuracy"], 4),
                "rmse":     round(best_metrics["rmse"], 4),
                "mae":      round(best_metrics["mae"], 4),
                "r2":       round(best_metrics["r2"], 4),
            },
            description=(
                f"Best model: {best_model_name} | "
                f"Time-based split | "
                f"Trained on Pakistan AQI data (21840 rows) | "
                f"Acc={best_metrics['accuracy']:.4f}"
            ),
            input_example=X_test.iloc[0].to_dict(),
        )
        hw_model.save(os.path.join(BASE_DIR, "models"))
        print(f"Model registered as version {version}!")
        break
    except Exception:
        continue

print("\n=== TRAINING COMPLETE ===")
for name, m in results.items():
    print(f"{name:25s}: acc={m['accuracy']:.4f}  rmse={m['rmse']:.4f}  mae={m['mae']:.4f}  r2={m['r2']:.4f}")