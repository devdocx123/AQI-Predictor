# AQI Predictor — Islamabad

A production-grade, fully serverless Air Quality Index prediction system for Islamabad, Pakistan. Fetches live pollution data hourly, engineers features, trains ML models, and serves predictions through an interactive dashboard.

## Architecture

```
WAQI API (live)
     │
     ▼
fetch.py ──► raw_aqi_data.csv (staging)
     │
     ▼
features.py ──► Hopsworks Feature Store
     │
     ▼
train.py ──► Hopsworks Model Registry
     │
     ▼
app.py ──► Streamlit Dashboard (live)
```

All pipeline steps are automated via **GitHub Actions** — features update hourly, model retrains daily.

## Live Dashboard

[saeed-aqi-predictor.streamlit.app](https://aqi-predictor-by-saeed.streamlit.app/)

## Project Structure

```
AQI-Predictor/
├── pipeline/
│   ├── fetch.py          # Fetch live AQI data from WAQI API
│   ├── features.py       # Feature engineering → Hopsworks Feature Store
│   ├── train.py          # Train models → Hopsworks Model Registry
│   └── predict.py        # Prediction utility
├── data/
│   ├── dataset/          # Historical Pakistan air quality dataset (21,840 rows)
│   ├── raw/              # Raw API data (staging buffer)
│   └── processed/        # Processed features
├── models/               # Saved model artifacts
├── notebooks/
    ├── training.ipynb    # Actual model training
│   └── EDA.ipynb         # Exploratory data analysis
├── app.py                # Streamlit dashboard
├── requirements.txt      # App dependencies
├── requirements-pipeline.txt  # Pipeline dependencies
└── .github/workflows/
    ├── fetch_data.yml    # Hourly feature pipeline
    └── train.yml         # Daily training pipeline
```

## Features Engineered

| Feature | Description |
|---|---|
| `pm2_5` | Fine particulate matter (primary AQI driver) |
| `pm10` | Coarse particulate matter |
| `carbon_monoxide` | CO concentration |
| `nitrogen_dioxide` | NO₂ concentration |
| `sulphur_dioxide` | SO₂ concentration |
| `ozone` | O₃ concentration |
| `temperature` | Ambient temperature |
| `humidity` | Relative humidity |
| `pressure` | Atmospheric pressure |
| `wind_speed` | Wind speed |
| `hour`, `day`, `month`, `weekday` | Time-based features |
| `aqi_lag_1/2/3` | Lag features |
| `aqi_rolling_mean_3` | Rolling average |
| `aqi_change` | Rate of change |

## Model Performance

| Model | Accuracy | RMSE | MAE | R² |
|---|---|---|---|---|
| Random Forest | 100% | 0.0000 | 0.0000 | 1.00 |
| Gradient Boosting | 100% | 0.0000 | 0.0000 | 1.00 |
| Logistic Regression | 89.97% | 0.5876 | 0.1777 | 0.7268 |

> Random Forest and Gradient Boosting achieve perfect scores because AQI categories are mathematically derived from PM2.5 thresholds using EPA breakpoints. Logistic Regression (90%) shows realistic linear boundary uncertainty.

## Setup

```bash
# Clone repo
git clone https://github.com/devdocx123/AQI-Predictor.git
cd AQI-Predictor

# Install dependencies
pip install -r requirements-pipeline.txt

# Set environment variables
cp .env.example .env
# Add your API keys to .env

# Run pipeline
python pipeline/fetch.py
python pipeline/features.py
python pipeline/train.py

# Launch dashboard
pip install streamlit
streamlit run app.py
```

## Environment Variables

```
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_PROJECT=your_project_name
WAQI_API_TOKEN=your_waqi_token
```

## Data Sources

- **Live data:** [WAQI API](https://aqicn.org/api/) — World Air Quality Index
- **Historical data:** Pakistan Air Quality Dataset — 21,840 records across 6 AQI categories

## Tech Stack

| Component | Technology |
|---|---|
| Feature Store | Hopsworks |
| Model Registry | Hopsworks |
| CI/CD | GitHub Actions |
| Dashboard | Streamlit |
| ML Models | scikit-learn (RandomForest, GradientBoosting, LogisticRegression) |
| Feature Importance | SHAP |
| Data Source | WAQI API |
