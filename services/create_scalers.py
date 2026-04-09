# create_scalers.py
import os
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Adjust these paths if your project is elsewhere
PROJECT_ROOT = r"D:\BreatheAI"
SRC_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "breatheai_6state_2020_2025_realistic.csv"),
    os.path.join(PROJECT_ROOT, "breatheai_aqi_10yr_dataset_enriched.csv"),
    "/mnt/data/breatheai_6state_2020_2025_realistic.csv"
]
DST_DIR = os.path.join(PROJECT_ROOT, "services")

# find dataset
src = None
for p in SRC_CANDIDATES:
    if p and os.path.exists(p):
        src = p
        break

if src is None:
    raise FileNotFoundError(f"No source CSV found. Tried: {SRC_CANDIDATES}")

print("Using dataset:", src)
df = pd.read_csv(src)
df.columns = df.columns.str.strip().str.lower()
print("Dataset shape:", df.shape)

# choose features: exclude common metadata and the target 'aqi'
excluded = {'aqi', 'city', 'station_type', 'station_id', 'station', 'timestamp', 'date', 'time', 'season'}
feature_cols = [c for c in df.columns if c not in excluded]

if not feature_cols:
    raise RuntimeError("No feature columns found - review excluded list and dataset columns.")

print("Detected feature columns (count={}):".format(len(feature_cols)))
print(feature_cols)

# build X and y (drop rows with NaN in target)
df = df.dropna(subset=['aqi'])
X = df[feature_cols].values
y = df['aqi'].values.reshape(-1, 1)

print("Fitting scalers to X shape", X.shape, "and y shape", y.shape)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_X.fit(X)
scaler_y.fit(y)

os.makedirs(DST_DIR, exist_ok=True)
fx = os.path.join(DST_DIR, "scaler_features.pkl")
fy = os.path.join(DST_DIR, "scaler_y.pkl")
fp = os.path.join(DST_DIR, "preprocessing_columns.pkl")

joblib.dump(scaler_X, fx)
joblib.dump(scaler_y, fy)
joblib.dump(feature_cols, fp)

print("Saved scaler_features ->", fx)
print("Saved scaler_y ->", fy)
print("Saved preprocessing_columns ->", fp)
