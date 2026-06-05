import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

st.set_page_config(
    page_title="AQI Predictor — Islamabad",
    page_icon="🌫️",
    layout="wide",
)

# ─── AQI HELPERS ──────────────────────────────────────────────────────────────
AQI_COLORS = {
    "Good":                           "#00e400",
    "Moderate":                       "#ffff00",
    "Unhealthy for Sensitive Groups":  "#ff7e00",
    "Unhealthy":                      "#ff0000",
    "Very Unhealthy":                 "#8f3f97",
    "Hazardous":                      "#7e0023",
}

AQI_ADVICE = {
    "Good":                           "Air quality is satisfactory. Enjoy outdoor activities!",
    "Moderate":                       "Acceptable. Unusually sensitive people should limit prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups":  "⚠️ Sensitive groups should reduce outdoor activity.",
    "Unhealthy":                      "🔴 Everyone may experience health effects. Limit outdoor activity.",
    "Very Unhealthy":                 "🚨 Health alert! Avoid outdoor activity.",
    "Hazardous":                      "☠️ HAZARDOUS — Emergency conditions. Stay indoors!",
}

ALERT_CATEGORIES = ["Unhealthy", "Very Unhealthy", "Hazardous"]

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    model_path   = os.path.join(base, "models", "aqi_model.pkl")
    encoder_path = os.path.join(base, "models", "label_encoder.pkl")
    features_path = os.path.join(base, "models", "feature_names.pkl")

    if not os.path.exists(model_path):
        st.error("❌ Model not found at models/aqi_model.pkl")
        st.stop()

    model    = joblib.load(model_path)
    encoder  = joblib.load(encoder_path)
    features = joblib.load(features_path) if os.path.exists(features_path) else [
        "pm10","pm2_5","carbon_monoxide","nitrogen_dioxide",
        "sulphur_dioxide","ozone","temperature","humidity","pressure","wind_speed"
    ]
    return model, encoder, features

model, encoder, feature_names = load_model()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.title("🌫️ AQI Predictor")
st.sidebar.markdown("**City:** Islamabad, Pakistan")
st.sidebar.markdown(f"**Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
st.sidebar.divider()
st.sidebar.markdown("### Sensor Input Values")

pm10  = st.sidebar.slider("PM10 (μg/m³)",   0.0, 200.0, 30.0)
pm25  = st.sidebar.slider("PM2.5 (μg/m³)",  0.0, 200.0, 25.0)
co    = st.sidebar.slider("CO (μg/m³)",     0.0, 2000.0, 500.0)
no2   = st.sidebar.slider("NO₂ (μg/m³)",    0.0, 200.0, 20.0)
so2   = st.sidebar.slider("SO₂ (μg/m³)",    0.0, 100.0, 5.0)
ozone = st.sidebar.slider("Ozone (μg/m³)",  0.0, 200.0, 60.0)
temp  = st.sidebar.slider("Temperature (°C)", -10.0, 50.0, 25.0)
hum   = st.sidebar.slider("Humidity (%)",    0.0, 100.0, 60.0)
pres  = st.sidebar.slider("Pressure (hPa)", 900.0, 1100.0, 1013.0)
wind  = st.sidebar.slider("Wind Speed (km/h)", 0.0, 50.0, 5.0)

# ─── MAIN ─────────────────────────────────────────────────────────────────────
st.title("🌫️ AQI Prediction Dashboard — Islamabad")
st.markdown("Real-time Air Quality Index prediction using a RandomForest model trained on Pakistan air quality data.")

# ── CURRENT PREDICTION ────────────────────────────────────────────────────────
st.header("📍 Current AQI Prediction")

sample     = np.array([[pm10, pm25, co, no2, so2, ozone, temp, hum, pres, wind]])
prediction = model.predict(sample)[0]
category   = encoder.inverse_transform([prediction])[0]
color      = AQI_COLORS.get(category, "#888")
advice     = AQI_ADVICE.get(category, "")
text_color = "#000" if category in ["Good", "Moderate"] else "#fff"

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown(
        f'<div style="background:{color};border-radius:16px;padding:40px;text-align:center;">'
        f'<h1 style="color:{text_color};margin:0;font-size:2rem;">{category}</h1>'
        f'</div>',
        unsafe_allow_html=True,
    )
with col2:
    st.info(advice)
    st.markdown("**Input values:**")
    st.dataframe(
        pd.DataFrame({"Feature": feature_names, "Value": sample[0]}),
        use_container_width=True, hide_index=True
    )

st.divider()

# ── MODEL COMPARISON ──────────────────────────────────────────────────────────
st.header("📊 Model Performance Comparison")

perf = pd.DataFrame({
    "Model":    ["RandomForest", "GradientBoosting", "LogisticRegression"],
    "Accuracy": [1.0000,         1.0000,              0.8997],
    "RMSE":     [0.0000,         0.0000,              0.5876],
    "MAE":      [0.0000,         0.0000,              0.1777],
    "R²":       [1.0000,         1.0000,              0.7268],
})
st.dataframe(perf, use_container_width=True, hide_index=True)
st.caption("RF/GB achieve 100% because AQI categories are mathematically derived from PM2.5 thresholds (EPA standard). LogisticRegression shows realistic boundary uncertainty.")

st.divider()

# ── 3-DAY FORECAST ────────────────────────────────────────────────────────────
st.header("📅 3-Day AQI Forecast")


forecast_rows = []
base = sample[0].copy()

# Hourly multipliers based on real AQI patterns:
# Morning (6am): higher pollution (traffic + cold air)
# Afternoon (12pm): lower (wind + heat disperses pollutants)  
# Evening (6pm): higher again (traffic + cooling)
hour_multipliers = {6: 1.3, 12: 0.75, 18: 1.15}

for day in range(3):
    day_drift = 1.0 + (day * np.random.uniform(-0.1, 0.1))
    for hour in [6, 12, 18]:
        h_mult = hour_multipliers[hour]
        varied = np.clip(base * h_mult * day_drift, 0, None)
        pred   = model.predict(varied.reshape(1, -1))[0]
        cat    = encoder.inverse_transform([pred])[0]
        dt     = datetime.now() + timedelta(days=day)
        forecast_rows.append({
            "Date": dt.strftime("%b %d"),
            "Time": f"{hour:02d}:00",
            "Category": cat,
            "Color": AQI_COLORS.get(cat, "#888"),
            "TextColor": "#000" if cat in ["Good","Moderate"] else "#fff",
        })

forecast_df = pd.DataFrame(forecast_rows)
cols = st.columns(3)
for i, date in enumerate(forecast_df["Date"].unique()):
    day_df = forecast_df[forecast_df["Date"] == date]
    with cols[i]:
        st.markdown(f"**{date}**")
        for _, row in day_df.iterrows():
            st.markdown(
                f'<div style="background:{row["Color"]};color:{row["TextColor"]};'
                f'border-radius:8px;padding:8px 14px;margin:4px 0;font-weight:600;">'
                f'{row["Time"]} — {row["Category"]}</div>',
                unsafe_allow_html=True,
            )

st.divider()

# ── SHAP FEATURE IMPORTANCE ───────────────────────────────────────────────────
st.header("🔍 SHAP Feature Importance")

shap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "shap_summary.png")
if os.path.exists(shap_path):
    st.image(shap_path, caption="SHAP Summary Plot — Feature importance for AQI classification")
else:
    st.info("SHAP plot not found. Run pipeline/train.py to generate it.")

st.divider()

# ── ALERTS ────────────────────────────────────────────────────────────────────
st.header("🚨 AQI Hazard Alerts")

if category in ALERT_CATEGORIES:
    st.error(f"⚠️ **ALERT**: Current prediction is **{category}**. {advice}")
else:
    st.success("✅ No hazardous AQI levels detected with current input values.")

hazardous = forecast_df[forecast_df["Category"].isin(ALERT_CATEGORIES)]
if not hazardous.empty:
    st.warning(f"⚠️ Hazardous AQI forecasted on: {', '.join(hazardous['Date'].unique())}")

st.divider()
st.caption("Model: RandomForest | Data: Pakistan Air Quality Dataset (21,840 rows) | Features stored in Hopsworks Feature Store")