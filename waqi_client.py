# waqi_client.py
import os, time, requests, pandas as pd, numpy as np
from datetime import timezone

# --- WAQI Configuration ---
# Direct token assignment (you can replace this later with os.getenv if needed)
WAQI_TOKEN = "5e7c6a5bf8c15728ddf31317348af5ed01d672e8"

# File to store historical data (can be changed to absolute path)
HISTORY_CSV = os.getenv("WAQI_HISTORY_CSV", "waqi_history.csv")


# ---------- Helper Functions ---------- #

def _get_with_backoff(url, max_tries=5, timeout=10):
    """Perform GET request with exponential backoff for rate limits and network retries."""
    tries = 0
    while True:
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 429:  # rate limit
                tries += 1
                if tries >= max_tries:
                    r.raise_for_status()
                time.sleep(2 ** tries)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            tries += 1
            if tries >= max_tries:
                raise
            time.sleep(2 ** tries)


# ---------- Core WAQI Data Fetch ---------- #

def fetch_waqi_by_city(city):
    """
    Fetch AQI data for a given city from WAQI API.
    Returns a single-row pandas DataFrame indexed by timestamp.
    """
    import urllib.parse
    city_enc = urllib.parse.quote(city)
    url = f"https://api.waqi.info/feed/{city_enc}/?token={WAQI_TOKEN}"
    resp = _get_with_backoff(url)
    data = resp.json()

    if data.get("status") != "ok":
        raise RuntimeError(f"WAQI error: {data}")

    d = data["data"]
    ts_str = d.get("time", {}).get("s")
    ts = pd.to_datetime(ts_str) if ts_str else pd.Timestamp.now(tz=timezone.utc)

    # Build a row with main pollutants
    row = {
        "aqi": d.get("aqi"),
        "pm25": d.get("iaqi", {}).get("pm25", {}).get("v"),
        "pm10": d.get("iaqi", {}).get("pm10", {}).get("v"),
        "o3": d.get("iaqi", {}).get("o3", {}).get("v"),
        "no2": d.get("iaqi", {}).get("no2", {}).get("v"),
        "so2": d.get("iaqi", {}).get("so2", {}).get("v"),
        "co": d.get("iaqi", {}).get("co", {}).get("v"),
        "city_name": d.get("city", {}).get("name"),
        "idx": d.get("idx"),
        "raw_json": str(d)
    }

    df = pd.DataFrame([row], index=[pd.to_datetime(ts)])
    df.index.name = "timestamp"

    # Normalize timezone to UTC naive for consistency
    try:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    except Exception:
        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            pass

    return df


# ---------- Append to History ---------- #

def append_history(df_row, path=HISTORY_CSV):
    """Append the new row to historical CSV, deduplicate by timestamp."""
    try:
        hist = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    except FileNotFoundError:
        hist = pd.DataFrame()

    combined = pd.concat([hist, df_row])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_csv(path)
    return combined


# ---------- High-Level Helper ---------- #

def fetch_and_store(city, history_csv_path=HISTORY_CSV):
    """
    Fetch latest WAQI data for the city and store it to CSV history file.
    Returns the updated DataFrame.
    """
    df = fetch_waqi_by_city(city)
    return append_history(df, history_csv_path)


# ---------- Prepare Data for LSTM ---------- #

def prepare_lstm_input(history_csv_path=HISTORY_CSV, city=None, lookback_hours=24, features=None):
    """
    Prepare WAQI data for LSTM model prediction.
    Returns:
        X (1, lookback_hours, n_features)
        last_ts (timestamp of last observation)
    """
    df = pd.read_csv(history_csv_path, parse_dates=["timestamp"], index_col="timestamp")

    # Filter by city if provided
    if city and "city_name" in df.columns:
        df = df[df["city_name"].str.contains(city, case=False, na=False)]

    # Hourly resampling, interpolate small gaps
    df_hour = df.resample("H").mean().interpolate(limit=3).ffill(limit=3)

    if features is None:
        features = [c for c in df_hour.columns if c in ("aqi", "pm25", "pm10", "o3", "no2", "so2", "co")]

    # Ensure missing features exist
    for f in features:
        if f not in df_hour.columns:
            df_hour[f] = np.nan

    # Extract last lookback_hours
    tail = df_hour[-lookback_hours:]

    # Create zero array for model input
    X = np.zeros((1, lookback_hours, len(features)))

    # Right-align data & fill missing
    for i, f in enumerate(features):
        vals = tail[f].values
        if len(vals) < lookback_hours:
            arr = np.full(lookback_hours, np.nan)
            arr[-len(vals):] = vals
            arr = pd.Series(arr).fillna(method="ffill").fillna(method="bfill").fillna(0).values
        else:
            arr = vals
        X[0, :, i] = arr

    last_ts = df_hour.index[-1] if len(df_hour.index) else pd.Timestamp.now()

    return X, last_ts
