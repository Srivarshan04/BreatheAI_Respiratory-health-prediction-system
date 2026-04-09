import os
import pickle
import logging
import requests
from datetime import datetime
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, current_app
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from extensions import db  # single shared SQLAlchemy instance
from models import User, City, HealthLog, Message, Notification, AQIData
from flask_apscheduler import APScheduler
from services.aqi_service import save_current_aqi
from dotenv import load_dotenv
from utils.aqi_fetcher import fetch_and_store_aqi
from flask_migrate import Migrate
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import logging
from datetime import datetime
import requests
from flask import request, jsonify
from services.predictor_lstm import predict_next_day_aqi
from services.predictor_ann import generate_health_suggestion
from utils.predict_aqi_tolerance import predict_aqi_tolerance   


# ------- env & logging -------
load_dotenv()
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("DEBUG OPENWEATHER_API_KEY: %s", os.getenv("OPENWEATHER_API_KEY"))
logging.getLogger('utils.predict_aqi_tolerance').setLevel(logging.DEBUG)
# ----------------------
# Load LSTM model, scalers, and dataset (forecasting)
# ----------------------
# NOTE: tries both services/ (preferred) and utils/ (fallback) directories
SERVICE_DIR = os.path.join(APP_ROOT, "services")
UTILS_DIR = os.path.join(APP_ROOT, "utils")

# Candidate paths (try services first, then utils)
LSTM_CANDIDATES = [
    os.path.join(SERVICE_DIR, "lstm_model.h5"),
    os.path.join(SERVICE_DIR, "lstm_model.keras"),
    os.path.join(UTILS_DIR, "lstm_model.h5"),
    os.path.join(UTILS_DIR, "lstm_model.keras"),
]

SCALER_FEATURES_CANDIDATES = [
    os.path.join(SERVICE_DIR, "scaler_features.pkl"),
    os.path.join(UTILS_DIR, "scaler_features.pkl"),
]
SCALER_Y_CANDIDATES = [
    os.path.join(SERVICE_DIR, "scaler_y.pkl"),
    os.path.join(UTILS_DIR, "scaler_y.pkl"),
]

# dataset (keep previous default)
DATASET_PATH = os.path.join(APP_ROOT, "breatheai_aqi_10yr_dataset_enriched.csv")

lstm_model = None
scaler_features = None
scaler_y = None

# helper to find first existing path from candidates
def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

# Find actual files
LSTM_PATH = _first_existing(LSTM_CANDIDATES)
SCALER_FEATURES_PATH = _first_existing(SCALER_FEATURES_CANDIDATES)
SCALER_Y_PATH = _first_existing(SCALER_Y_CANDIDATES)

# Load LSTM model (if present)
if LSTM_PATH:
    try:
        lstm_model = load_model(LSTM_PATH)
        logger.info("Loaded LSTM model from: %s", LSTM_PATH)
        try:
            logger.info("LSTM input_shape: %s output_shape: %s", getattr(lstm_model, "input_shape", None), getattr(lstm_model, "output_shape", None))
        except Exception:
            pass
    except Exception as e:
        logger.exception("Failed to load LSTM model from %s: %s", LSTM_PATH, e)
        lstm_model = None
else:
    logger.warning("LSTM model not found in candidates: %s", LSTM_CANDIDATES)

# Load scalers (if both present)
if SCALER_FEATURES_PATH and SCALER_Y_PATH:
    try:
        scaler_features = joblib.load(SCALER_FEATURES_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)
        logger.info("Loaded scalers: features=%s y=%s", SCALER_FEATURES_PATH, SCALER_Y_PATH)
    except Exception as e:
        logger.exception("Failed to load scaler files: %s / %s : %s", SCALER_FEATURES_PATH, SCALER_Y_PATH, e)
        scaler_features = None
        scaler_y = None
else:
    logger.warning("Scaler files missing. Tried: features=%s y=%s", SCALER_FEATURES_CANDIDATES, SCALER_Y_CANDIDATES)

# Load dataset (same behavior as before)
try:
    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.str.strip().str.lower()
    logger.info("Loaded dataset %s", DATASET_PATH)
except Exception as e:
    logger.exception("Warning: could not load dataset CSV: %s", e)
    df = pd.DataFrame()

# Quick info for debugging: where we looked
logger.info("Model/scaler discovery summary -> lstm:%s scaler_features:%s scaler_y:%s", LSTM_PATH, SCALER_FEATURES_PATH, SCALER_Y_PATH)


def forecast_city_with_today(city_name, model, df_obj, scaler_features_obj, scaler_y_obj, T_in=7, T_out=6):
    """
    Generate a short forecast DataFrame for a given city using the LSTM model and scalers.
    This version auto-adapts to the scaler/model feature count (e.g. single-feature LSTM).
    """
    if df_obj is None or df_obj.empty:
        raise ValueError("Local dataset is empty; cannot forecast.")

    if model is None:
        raise ValueError("LSTM model is not loaded; cannot forecast.")

    if scaler_features_obj is None or scaler_y_obj is None:
        raise ValueError("Scalers not loaded; cannot forecast.")

    # match city (case-insensitive)
    try:
        city_data = df_obj[df_obj['city'].str.lower() == str(city_name).strip().lower()].copy()
    except Exception as e:
        raise ValueError(f"Error while selecting city rows: {e}")

    if city_data.empty:
        raise ValueError(f"No data found for city: {city_name}")

    if len(city_data) < T_in:
        raise ValueError(f"Not enough data for city {city_name} (have {len(city_data)}, need {T_in})")

    # most recent observed AQI
    try:
        today_aqi = float(city_data['aqi'].iloc[-1])
    except Exception:
        raise ValueError("Failed to read last observed 'aqi' value for the city")

    # Determine number of features expected by scaler (if available)
    n_expected = getattr(scaler_features_obj, "n_features_in_", None)

    # Choose feature columns based on scaler expectation or saved preprocessing_columns
    feature_cols = None
    try:
        # If scaler explicitly expects 1 feature, use the AQI column (common case)
        if n_expected == 1:
            if 'aqi' in df_obj.columns:
                feature_cols = ['aqi']
            else:
                candidate_cols = [c for c in df_obj.columns if c not in ['aqi', 'aqi_band', 'city', 'station_id', 'station_type', 'timestamp', 'season']]
                feature_cols = [candidate_cols[0]] if candidate_cols else None
        else:
            # Prefer saved preprocessing_columns (exact order) if available
            if 'preprocessing_columns' in globals() and preprocessing_columns:
                feature_cols = [c for c in preprocessing_columns if c in df_obj.columns and c not in ['aqi', 'aqi_band', 'city', 'station_id', 'station_type', 'timestamp', 'season']]
            if not feature_cols:
                feature_cols = [col for col in df_obj.columns if col not in ['aqi', 'aqi_band', 'city', 'station_id', 'station_type', 'timestamp', 'season']]
    except Exception:
        feature_cols = None

    if not feature_cols:
        raise ValueError("No feature columns found in dataset to build LSTM input")

    # Ensure number of chosen features matches scaler expectation if we know it
    if n_expected is not None and n_expected != len(feature_cols):
        # If scaler expects 1 but we selected many, reduce to 'aqi' if present (defensive)
        if n_expected == 1 and 'aqi' in df_obj.columns:
            feature_cols = ['aqi']
        else:
            raise RuntimeError(f"Feature count mismatch: scaler expects {n_expected} features but selected {len(feature_cols)}. Check preprocessing_columns/order.")

    # Debug log for feature_cols shape
    logging.getLogger(__name__).info("forecast_city_with_today -> using feature_cols=%s (n_expected=%s)", feature_cols, n_expected)

    # Build last T_in sequence using the selected columns and ensure ordering matches scaler
    last_sequence_df = city_data[feature_cols].tail(T_in)
    last_sequence = last_sequence_df.values  # shape (T_in, n_features)

    # If there are NaNs in the last sequence, do a quick median impute (pragmatic fallback)
    if np.isnan(last_sequence).any():
        col_medians = np.nanmedian(last_sequence, axis=0)
        inds = np.where(np.isnan(last_sequence))
        for r, c in zip(*inds):
            last_sequence[r, c] = col_medians[c]
        logger = logging.getLogger(__name__)
        logger.warning("Last sequence contained NaNs — applied quick median imputation for forecasting (city=%s)", city_name)

    # scale then predict
    try:
        last_sequence_scaled = scaler_features_obj.transform(last_sequence)
    except Exception as e:
        raise RuntimeError(f"Feature scaler transform failed: {e}")

    # prepare shape (1, T_in, n_features) for keras LSTM
    X_input = np.expand_dims(last_sequence_scaled, axis=0)

    try:
        y_pred_scaled = model.predict(X_input)
    except Exception as e:
        raise RuntimeError(f"LSTM model prediction failed: {e}")

    # Flatten predictions (1D) from model output
    y_pred_flat = np.array(y_pred_scaled).reshape(-1)

    # Debug: log raw model outputs (helps diagnose scale issues)
    try:
        app.logger.debug("LSTM raw output (y_pred_scaled): %s", y_pred_flat.tolist())
    except Exception:
        pass

    # Robust inverse-transform helper
    def robust_inverse_transform_y(scaler, arr_flat):
        arr_flat = np.asarray(arr_flat).reshape(-1)
        arr2d = arr_flat.reshape(-1, 1)

        # 1) Standard attempt: use scaler.inverse_transform if available
        if scaler is not None and hasattr(scaler, "inverse_transform"):
            try:
                out = scaler.inverse_transform(arr2d)
                return np.asarray(out).reshape(-1)
            except Exception as e:
                app.logger.debug("scaler.inverse_transform failed: %s", e)

        # 2) Try MinMax-like properties
        try:
            if scaler is not None and hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
                data_min = np.asarray(scaler.data_min_).reshape(-1)[0]
                data_max = np.asarray(scaler.data_max_).reshape(-1)[0]
                # assume arr_flat scaled to [0,1]
                return (arr_flat * (data_max - data_min)) + data_min
        except Exception as e:
            app.logger.debug("MinMax manual inverse failed: %s", e)

        # 3) Try StandardScaler-like properties
        try:
            if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
                mean = np.asarray(scaler.mean_).reshape(-1)[0]
                scale = np.asarray(scaler.scale_).reshape(-1)[0]
                return (arr_flat * scale) + mean
        except Exception as e:
            app.logger.debug("StandardScaler manual inverse failed: %s", e)

        # 4) If scaler is a dict-like object with min/max
        try:
            if isinstance(scaler, dict):
                mn = scaler.get("min") or scaler.get("data_min") or scaler.get("min_")
                mx = scaler.get("max") or scaler.get("data_max") or scaler.get("max_")
                if mn is not None and mx is not None:
                    return (arr_flat * (mx - mn)) + mn
        except Exception:
            pass

        # 5) Heuristic: if outputs already look like AQI (0..500), accept them
        try:
            mn = np.nanmin(arr_flat)
            mx = np.nanmax(arr_flat)
            if mn >= 0 and mx <= 500 and np.nanmean(arr_flat) > 1:
                return arr_flat
        except Exception:
            pass

        raise RuntimeError("Unable to inverse-transform predicted y with current scaler.")

    # Attempt robust inverse transform, with a safe fallback
    try:
        y_pred = robust_inverse_transform_y(scaler_y_obj, y_pred_flat)
    except Exception as ex:
        # Log rich debugging info
        try:
            app.logger.exception("Failed to inverse-transform predictions: %s", ex)
            # Additional debug context
            app.logger.debug("y_pred_flat: %s", y_pred_flat.tolist())
            app.logger.debug("scaler_y_obj type: %s", type(scaler_y_obj))
            # inspect common attributes if present
            for attr in ("mean_", "scale_", "data_min_", "data_max_", "min_", "max_"):
                if hasattr(scaler_y_obj, attr):
                    try:
                        app.logger.debug("scaler_y_obj.%s = %s", attr, getattr(scaler_y_obj, attr))
                    except Exception:
                        pass
        except Exception:
            pass

        # Fallback: return today's AQI repeated (safer than tiny nonsense values)
        y_pred = np.array([today_aqi] * len(y_pred_flat))

    # Build DataFrame: Today + Day+1..Day+T_out
    forecast_days = ["Today"] + [f"Day+{i}" for i in range(1, T_out + 1)]
    # Ensure we have at least T_out values
    forecast_values = [today_aqi] + list(y_pred[:T_out]) if len(y_pred) >= T_out else [today_aqi] + list(np.pad(y_pred, (0, max(0, T_out - len(y_pred))), constant_values=today_aqi)[:T_out])

    return pd.DataFrame({"Day": forecast_days, "Predicted_AQI": forecast_values})



# ----------------------
# Flask App Setup
# ----------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "your_secret_key")  # use env var in production
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL", 'postgresql://postgres:sri27082003@localhost/breatheai_db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['OPENWEATHER_API_KEY'] = os.getenv("OPENWEATHER_API_KEY", "")
app.config['AQI_ALERT_THRESHOLD'] = 150

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Scheduler
scheduler = APScheduler()
scheduler.init_app(app)

# ----------------------
# AQI helpers
# ----------------------
def pm_to_aqi(concentration, breakpoints):
    try:
        concentration = float(concentration)
    except Exception:
        return None
    for bp in breakpoints:
        if bp["low"] <= concentration <= bp["high"]:
            return round(
                (bp["aqi_high"] - bp["aqi_low"]) / (bp["high"] - bp["low"]) * (concentration - bp["low"]) + bp["aqi_low"]
            )
    return None

PM25_BREAKPOINTS = [
    {"low": 0.0, "high": 12.0, "aqi_low": 0, "aqi_high": 50},
    {"low": 12.1, "high": 35.4, "aqi_low": 51, "aqi_high": 100},
    {"low": 35.5, "high": 55.4, "aqi_low": 101, "aqi_high": 150},
    {"low": 55.5, "high": 150.4, "aqi_low": 151, "aqi_high": 200},
    {"low": 150.5, "high": 250.4, "aqi_low": 201, "aqi_high": 300},
    {"low": 250.5, "high": 350.4, "aqi_low": 301, "aqi_high": 400},
    {"low": 350.5, "high": 500.4, "aqi_low": 401, "aqi_high": 500},
]

PM10_BREAKPOINTS = [
    {"low": 0, "high": 54, "aqi_low": 0, "aqi_high": 50},
    {"low": 55, "high": 154, "aqi_low": 51, "aqi_high": 100},
    {"low": 155, "high": 254, "aqi_low": 101, "aqi_high": 150},
    {"low": 255, "high": 354, "aqi_low": 151, "aqi_high": 200},
    {"low": 355, "high": 424, "aqi_low": 201, "aqi_high": 300},
    {"low": 425, "high": 504, "aqi_low": 301, "aqi_high": 400},
    {"low": 505, "high": 604, "aqi_low": 401, "aqi_high": 500},
]

def aqi_category(aqi):
    if aqi is None:
        return "Unknown"
    try:
        aqi = int(aqi)
    except Exception:
        return "Unknown"
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

# ----------------------
# Login manager loader
# ----------------------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ----------------------
# Load logistic regression model + preprocessing columns (patient tolerance)
# ----------------------
# --- ML artifact paths (module-level) ---
# UTILS_DIR variable is already defined above; reuse it here for compatibility with the rest of your code
PIPELINE_PATH = os.path.join(UTILS_DIR, "pipeline_model.pkl")
LEGACY_LR_PATH = os.path.join(UTILS_DIR, "logistic_regression_model.pkl")
PREP_PATH = os.path.join(UTILS_DIR, "preprocessing_columns.pkl")

lr_model = None
preprocessing_columns = None

def _load_lr_model_module_safe():
    """Load LR model at module import time without using current_app."""
    # Prefer pipeline model
    if os.path.exists(PIPELINE_PATH):
        try:
            import joblib
            lr = joblib.load(PIPELINE_PATH)
            logger.info("Loaded pipeline model from %s", PIPELINE_PATH)
            return lr
        except Exception as e:
            logger.exception("Failed to load pipeline_model.pkl (%s): %s", PIPELINE_PATH, e)

    # Fallback: legacy LR
    if os.path.exists(LEGACY_LR_PATH):
        try:
            import joblib
            lr = joblib.load(LEGACY_LR_PATH)
            logger.info("Loaded legacy logistic regression model from %s", LEGACY_LR_PATH)
            return lr
        except Exception as e:
            logger.exception("Failed to load legacy logistic_regression_model.pkl (%s): %s", LEGACY_LR_PATH, e)

    logger.warning("No LR model found at pipeline=%s or legacy=%s", PIPELINE_PATH, LEGACY_LR_PATH)
    return None

# --- call the loader so lr_model is populated ---
try:
    lr_model = _load_lr_model_module_safe()
    if lr_model is not None:
        try:
            if hasattr(lr_model, "named_steps") or hasattr(lr_model, "steps"):
                logger.info("Detected scikit-learn Pipeline with steps: %s", list(lr_model.named_steps.keys()))
            else:
                logger.info("Loaded ML model of type: %s", type(lr_model))
        except Exception:
            logger.exception("Model loaded but verification step failed")
except Exception as e:
    logger.exception("Unexpected error while loading LR model at module import: %s", e)
    lr_model = None

# --- load preprocessing columns (module-safe) ---
try:
    if os.path.exists(PREP_PATH):
        import joblib
        preprocessing_columns = joblib.load(PREP_PATH)
        logger.info("Loaded preprocessing_columns from %s (count=%d)", PREP_PATH, len(preprocessing_columns) if preprocessing_columns else 0)
    else:
        logger.warning("preprocessing_columns.pkl not found at %s", PREP_PATH)
except Exception as e:
    logger.exception("Failed to load preprocessing_columns.pkl: %s", e)
    preprocessing_columns = None

# --- now register them into the Flask app config (must be after app = Flask(...)) ---
with app.app_context():
    try:
        app.config["LR_MODEL"] = lr_model
        app.config["PREPROCESSING_COLUMNS"] = preprocessing_columns
        logger.info("Loaded LR_MODEL into app.config (pipeline present=%s)", os.path.exists(PIPELINE_PATH))
    except Exception:
        logger.exception("Failed to write LR model into app.config")

# ----------------------
# Small helper to verify forecast artifacts (T_in must match training)
# ----------------------
def verify_forecast_artifacts(df_obj, T_in=7):
    """
    Return (ok: bool, msg: str). Checks that lstm_model, scalers and dataset have compatible shapes.
    This version tolerates the common case where the LSTM was trained on a single time-series (n_features=1),
    while the dataset contains many other columns.
    """
    if lstm_model is None:
        return False, "lstm_model not loaded"
    if scaler_features is None or scaler_y is None:
        return False, "scaler_features or scaler_y not loaded"
    if df_obj is None or df_obj.empty:
        return False, "dataset df is empty or not loaded"

    # derive feature columns (full dataset inference)
    feature_cols = [
        col for col in df_obj.columns
        if col not in ['aqi', 'aqi_band', 'city', 'station_id', 'station_type', 'timestamp', 'season']
    ]
    if not feature_cols:
        return False, "No feature columns found in dataset"

    # check a sample city has T_in rows
    try:
        sample_city = df_obj['city'].iloc[0]
        city_rows = df_obj[df_obj['city'].str.lower() == str(sample_city).lower()]
        if city_rows.shape[0] < T_in:
            return False, f"Not enough history for sample city '{sample_city}': have {city_rows.shape[0]}, need {T_in}"
    except Exception as e:
        logger.exception("verify_forecast_artifacts sample city check failed: %s", e)
        return False, f"verify check failed: {e}"

    # try to verify scaler_features expected n_features
    try:
        n_features_scaler = getattr(scaler_features, "n_features_in_", None)
        if n_features_scaler is None:
            # attempt a transform to infer shape (not ideal but works for many scalers)
            test = scaler_features.transform(np.zeros((1, len(feature_cols))))
            n_features_scaler = test.shape[1] if test.ndim == 2 else None

        # Acceptable if scaler expects 1 feature (model trained on a single timeseries)
        if n_features_scaler is not None:
            if n_features_scaler != len(feature_cols):
                if n_features_scaler == 1:
                    # allowed: we'll take 'aqi' as the single feature for forecasting
                    return True, "Forecast artifacts look OK (scaler expects 1 feature - will use 'aqi')"
                else:
                    return False, f"scaler_features expects {n_features_scaler} features but dataset has {len(feature_cols)}"
    except Exception:
        logger.warning("Could not verify scaler_features n_features_in_ reliably")

    return True, "Forecast artifacts look OK (basic checks passed)"


# ----------------------
# --- INSERTED START: module-level forecast_ok verification (one-time)
# ----------------------
# ensure the name exists at module level even if verification fails
forecast_ok = False
try:
    _forecast_ok, _forecast_msg = verify_forecast_artifacts(df, T_in=7)
    forecast_ok = _forecast_ok
    logger.info("Forecast readiness: ok=%s msg=%s", _forecast_ok, _forecast_msg)
except Exception as e:
    logger.exception("Forecast verification failed at startup: %s", e)
    forecast_ok = False
# ----------------------
# --- INSERTED END
# ----------------------


# ----------------------
# Routes (UI pages)
# ----------------------
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password_raw = request.form.get('password')
        role = request.form.get('role')

        if not (username and email and password_raw and role):
            flash("Please fill all fields.", "danger")
            return redirect(url_for('register'))

        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash("Username or Email already exists!", "danger")
            return redirect(url_for('register'))

        hashed = bcrypt.generate_password_hash(password_raw).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed, role=role)
        db.session.add(new_user)
        db.session.commit()
        flash("Account created successfully! Please login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password_raw = request.form.get('password')
        if not (username and password_raw):
            flash("Enter username and password", "danger")
            return redirect(url_for('login'))

        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password_raw):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password", "danger")
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'patient':
        # get last AQI for patient to populate dashboard input (avoid 404s if template expects it)
        last_log = HealthLog.query.filter_by(patient_id=current_user.id).order_by(HealthLog.timestamp.desc()).first()
        last_aqi = getattr(last_log, 'aqi', None) if last_log else None
        return render_template('patient_dashboard.html', user=current_user, last_aqi=last_aqi)
    elif current_user.role == 'doctor':
        return render_template('doctor_dashboard.html', user=current_user)
    else:
        flash("Invalid role detected.", "danger")
        return redirect(url_for('login'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out", "info")
    return redirect(url_for('login'))

# ----------------------
# get_aqi route (guarded forecast)
# Replace your existing /get_aqi route with this
# It uses WAQI instead of OpenWeather, preserves DB persistence & forecast logic.
# Put this into app.py (replace your existing /get_aqi handler).
# Requires: requests imported, WAQI_TOKEN configured (see note).

# Prefer token from config/env; fallback to hard-coded token if you used one earlier.
WAQI_TOKEN = app.config.get("WAQI_TOKEN", "5e7c6a5bf8c15728ddf31317348af5ed01d672e8")

@app.route('/get_aqi')
@login_required
def get_aqi():
    global forecast_ok
    """
    Robust WAQI-backed /get_aqi route.
    Accepts:
      - ?lat=<float>&lon=<float>  OR
      - ?city=<name>
    Returns JSON:
      { "aqi_value": <int>, "aqi_category": <str>, "components": {...}, "timestamp": ..., "next_day_aqi": <float|null> }
    """
    resolved_via = None

    try:
        city_param = (request.args.get("city") or "").strip()
        lat = request.args.get("lat", type=float)
        lon = request.args.get("lon", type=float)

        # If city provided but no lat/lon, attempt to resolve city -> coordinates
        if city_param and (lat is None or lon is None):
            resolved_via = "resolve_attempt"
            try:
                waqi_city_url = f"https://api.waqi.info/feed/{requests.utils.quote(city_param)}/?token={WAQI_TOKEN}"
                gr = requests.get(waqi_city_url, timeout=8)
                gr.raise_for_status()
                geo_res = gr.json()
                if geo_res.get("status") == "ok" and geo_res.get("data", {}).get("city", {}).get("geo"):
                    geo = geo_res["data"]["city"]["geo"]
                    if isinstance(geo, (list, tuple)) and len(geo) >= 2:
                        lat = float(geo[0]); lon = float(geo[1])
                        resolved_via = "waqi_feed_city"
                else:
                    # try WAQI search
                    try:
                        search_url = f"https://api.waqi.info/search/?keyword={requests.utils.quote(city_param)}&token={WAQI_TOKEN}"
                        sr = requests.get(search_url, timeout=8)
                        sr.raise_for_status()
                        sdata = sr.json()
                        if sdata.get("status") == "ok" and sdata.get("data"):
                            best = sdata["data"][0]
                            if best.get("station") and best["station"].get("geo"):
                                lat = float(best["station"]["geo"][0]); lon = float(best["station"]["geo"][1])
                                resolved_via = "waqi_search"
                    except Exception:
                        pass

                    # fallback: Nominatim
                    if lat is None or lon is None:
                        try:
                            nom_url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(city_param)}&format=json&limit=1"
                            ng = requests.get(nom_url, headers={'User-Agent': 'breatheai/1.0'}, timeout=8)
                            ng.raise_for_status()
                            ngj = ng.json()
                            if ngj:
                                lat = float(ngj[0]['lat']); lon = float(ngj[0]['lon'])
                                resolved_via = "nominatim"
                        except Exception:
                            pass
            except requests.exceptions.RequestException as rexc:
                app.logger.exception("City-resolution request failed for %s: %s", city_param, rexc)
                return jsonify({"error": "Geocoding request failed", "message": str(rexc)}), 502
            except Exception as exc:
                app.logger.exception("Unexpected city resolution error for %s: %s", city_param, exc)
                return jsonify({"error": "Geocoding error", "message": str(exc)}), 500

        if lat is None or lon is None:
            return jsonify({"error": f"Could not resolve city via WAQI: {city_param}"}), 400

        # --- robust WAQI call with retries & logging ---
        waqi_url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_TOKEN}"

        def _call_waqi(url, attempts=3, timeout=8):
            import time
            last_exc = None
            for i in range(attempts):
                try:
                    rr = requests.get(url, timeout=timeout)
                    app.logger.debug("WAQI attempt %d status=%s text=%s", i+1, getattr(rr, "status_code", None), (rr.text[:800] if rr.text else ""))
                    rr.raise_for_status()
                    return rr
                except requests.exceptions.RequestException as e:
                    last_exc = e
                    app.logger.warning("WAQI attempt %d failed: %s", i+1, e)
                    time.sleep(0.5 + i*0.2)
            raise last_exc

        try:
            r = _call_waqi(waqi_url, attempts=3, timeout=8)
            try:
                res = r.json()
            except Exception:
                app.logger.error("WAQI response is not valid JSON; body: %s", getattr(r, "text", None))
                raise RuntimeError("WAQI returned non-JSON response")

            # If WAQI returned non-ok status -> surface details and attempt nearest-station fallback
            data = None
            if not isinstance(res, dict) or res.get("status") != "ok":
                app.logger.warning("WAQI returned error: %s", res)

                # --- try nearest-station via WAQI search and call that station feed ---
                def _haversine(lat1, lon1, lat2, lon2):
                    import math
                    R = 6371.0
                    dlat = math.radians(lat2 - lat1)
                    dlon = math.radians(lon2 - lon1)
                    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
                    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                    return R * c

                try:
                    # attempt WAQI search by city_param first; if city_param empty, try search by lat/lon string
                    search_keyword = city_param or f"{lat},{lon}"
                    search_url = f"https://api.waqi.info/search/?keyword={requests.utils.quote(search_keyword)}&token={WAQI_TOKEN}"
                    sr = requests.get(search_url, timeout=6)
                    sr.raise_for_status()
                    sdata = sr.json()
                    if sdata.get("status") == "ok" and sdata.get("data"):
                        # choose nearest station by distance if station geo available
                        best = None
                        best_dist = None
                        for item in sdata["data"]:
                            geo = None
                            if isinstance(item.get("station"), dict):
                                geo = item["station"].get("geo")
                            if not geo:
                                geo = item.get("geo")
                            if geo and isinstance(geo, (list, tuple)) and len(geo) >= 2:
                                try:
                                    st_lat = float(geo[0]); st_lon = float(geo[1])
                                    dist = _haversine(lat, lon, st_lat, st_lon)
                                    if best is None or dist < best_dist:
                                        best = item
                                        best_dist = dist
                                except Exception:
                                    continue
                        if best:
                            # prepare station feed URL (prefer station.url if present)
                            station_url_fragment = None
                            if best.get("station") and best["station"].get("url"):
                                station_url_fragment = best["station"]["url"]
                            try:
                                if station_url_fragment:
                                    waqi_station_feed = f"https://api.waqi.info/feed/{station_url_fragment}/?token={WAQI_TOKEN}"
                                else:
                                    st_geo = best.get("station", {}).get("geo") or best.get("geo")
                                    waqi_station_feed = f"https://api.waqi.info/feed/geo:{st_geo[0]};{st_geo[1]}/?token={WAQI_TOKEN}"

                                rr = requests.get(waqi_station_feed, timeout=6)
                                rr.raise_for_status()
                                station_res = rr.json()
                                if isinstance(station_res, dict) and station_res.get("status") == "ok" and station_res.get("data"):
                                    data = station_res["data"]
                                    resolved_via = "waqi_nearest_station"
                                    app.logger.info("Using nearest WAQI station data (via search): %s (dist_km=%.2f)", station_url_fragment or st_geo, best_dist if best_dist is not None else -1)
                                else:
                                    app.logger.info("Nearest WAQI station call returned no data; will fallback to DB/OpenWeather.")
                            except Exception as e:
                                app.logger.info("Nearest WAQI station call failed: %s", e)
                    else:
                        app.logger.info("WAQI search returned no station data for '%s'", search_keyword)
                except Exception:
                    app.logger.exception("WAQI nearest-station search failed (non-fatal)")

                # If nearest-station didn't yield data, attempt DB/OpenWeather fallback and return
                if data is None:
                    # DB fallback: last AQI near this lat/lon
                    fallback_aqi = None
                    try:
                        loc_like = f"{round(lat,4)},{round(lon,4)}"
                        last = AQIData.query.filter(AQIData.location.like(f"{loc_like}%")).order_by(AQIData.date_time.desc()).first()
                        if last:
                            fallback_aqi = getattr(last, "aqi_value", None)
                    except Exception:
                        app.logger.exception("DB fallback lookup failed")

                    # OpenWeather fallback mapping if configured
                    ow_info = None
                    ow_key = app.config.get("OPENWEATHER_API_KEY")
                    if fallback_aqi is None and ow_key:
                        try:
                            ow_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={ow_key}"
                            owr = requests.get(ow_url, timeout=6)
                            if owr.status_code == 200:
                                owj = owr.json()
                                idx = owj.get("list", [{}])[0].get("main", {}).get("aqi")
                                if idx is not None:
                                    ow_map = {1:25, 2:75, 3:125, 4:175, 5:250}
                                    fallback_aqi = ow_map.get(int(idx), None)
                                    ow_info = {"openweather_index": idx, "mapped_aqi": fallback_aqi}
                        except Exception:
                            app.logger.exception("OpenWeather fallback failed")

                    return jsonify({
                        "error": "No data returned from WAQI API (station missing/offline). Returned fallback estimate if available.",
                        "waqi_response": res,
                        "fallback_aqi": fallback_aqi,
                        "openweather_fallback": ow_info
                    }), 502

            else:
                # normal path: WAQI responded OK for original geo feed
                data = res["data"]

        except requests.exceptions.RequestException as re:
            app.logger.exception("WAQI request error: %s", re)
            fallback_aqi = None
            try:
                loc_like = f"{round(lat,4)},{round(lon,4)}"
                last = AQIData.query.filter(AQIData.location.like(f"{loc_like}%")).order_by(AQIData.date_time.desc()).first()
                if last:
                    fallback_aqi = getattr(last, "aqi_value", None)
            except Exception:
                app.logger.exception("DB fallback lookup failed after WAQI request error")

            return jsonify({
                "error": "WAQI request failed",
                "message": str(re),
                "fallback_aqi": fallback_aqi
            }), 502

        except Exception as e:
            app.logger.exception("Unexpected WAQI handling error: %s", e)
            return jsonify({"error": "Unexpected WAQI handling error", "message": str(e)}), 500
        # --- end robust WAQI call ---

        # Ensure data exists
        if data is None:
            return jsonify({"error": "WAQI returned empty data"}), 500

        # Safe extraction of iaqi and components
        iaqi = data.get("iaqi", {}) or {}
        comps = {
            "pm2_5": (iaqi.get("pm25") or iaqi.get("pm2_5") or {}).get("v") if isinstance(iaqi, dict) else None,
            "pm10": (iaqi.get("pm10") or {}).get("v") if isinstance(iaqi, dict) else None,
            "o3": (iaqi.get("o3") or {}).get("v") if isinstance(iaqi, dict) else None,
            "no2": (iaqi.get("no2") or {}).get("v") if isinstance(iaqi, dict) else None,
            "so2": (iaqi.get("so2") or {}).get("v") if isinstance(iaqi, dict) else None,
            "co": (iaqi.get("co") or {}).get("v") if isinstance(iaqi, dict) else None
        }

        pm25_val = comps.get("pm2_5") if comps.get("pm2_5") is not None else 0
        pm10_val = comps.get("pm10") if comps.get("pm10") is not None else 0

        # Prefer WAQI's own AQI value if present
        waqi_reported = None
        try:
            if isinstance(data, dict) and data.get("aqi") is not None:
                waqi_reported = int(data.get("aqi"))
        except Exception:
            waqi_reported = None

        if waqi_reported is not None:
            overall_aqi = waqi_reported
        else:
            pm25_aqi = pm_to_aqi(pm25_val, PM25_BREAKPOINTS) or 0
            pm10_aqi = pm_to_aqi(pm10_val, PM10_BREAKPOINTS) or 0
            overall_aqi = max(pm25_aqi, pm10_aqi)

        # timestamp from WAQI if available
        ts_str = data.get("time", {}).get("s")
        timestamp_iso = ts_str if ts_str else datetime.utcnow().isoformat()

        # Persist AQI record (best-effort)
        try:
            aqi_record = AQIData(
                location=f"{lat:.4f},{lon:.4f}",
                aqi_value=int(overall_aqi),
                main_pollutant=None,
                date_time=datetime.fromisoformat(ts_str) if ts_str else datetime.utcnow()
            )
            db.session.add(aqi_record)
            db.session.commit()
        except Exception:
            db.session.rollback()

        # Next-day predicted AQI (same logic you had before)
        next_day_aqi = None
        try:
            # assemble seq from df if possible; fallback to overall_aqi repeated
            seq = None
            try:
                if 'df' in globals() and not df.empty and city_param:
                    city_rows = df[df['city'].str.lower() == city_param.lower()]
                    if not city_rows.empty:
                        n_expected_scaler = getattr(scaler_features, "n_features_in_", None)
                        if n_expected_scaler == 1:
                            last_seq_df = city_rows[['aqi']].tail(7)
                            if last_seq_df.shape[0] == 7:
                                seq = last_seq_df.values.tolist()
                        else:
                            feature_cols_local = [c for c in preprocessing_columns if c in df.columns] if preprocessing_columns else None
                            if not feature_cols_local:
                                feature_cols_local = [col for col in df.columns if col not in ['aqi','aqi_band','city','station_id','station_type','timestamp','season']]
                            last_seq_df = city_rows[feature_cols_local].tail(7) if feature_cols_local else pd.DataFrame()
                            if last_seq_df.shape[0] == 7:
                                seq = last_seq_df.values.tolist()
            except Exception:
                seq = None

            if seq is None:
                seq = [[float(overall_aqi)]] * 7

            if lstm_model is None:
                try:
                    pred_val = predict_next_day_aqi(seq)
                except Exception:
                    pred_val = None
            else:
                arr = np.asarray(seq, dtype=float)
                if scaler_features is not None and hasattr(scaler_features, "n_features_in_"):
                    n_expected = getattr(scaler_features, "n_features_in_", None)
                    if n_expected == 1 and arr.ndim == 2 and arr.shape[1] != 1:
                        arr = arr[:, 0].reshape(-1, 1)
                    try:
                        arr_scaled = scaler_features.transform(arr)
                    except Exception:
                        arr_scaled = arr
                else:
                    arr_scaled = arr
                X_input = np.expand_dims(arr_scaled, axis=0)
                try:
                    y_pred_scaled = lstm_model.predict(X_input)
                    y_pred_flat = np.array(y_pred_scaled).reshape(-1)
                except Exception:
                    try:
                        pred_val = predict_next_day_aqi(seq)
                        y_pred_flat = np.array([pred_val])
                    except Exception:
                        y_pred_flat = np.array([overall_aqi])

                # inverse transform helper (simple)
                try:
                    if scaler_y is not None and hasattr(scaler_y, "inverse_transform"):
                        y_pred = scaler_y.inverse_transform(y_pred_flat.reshape(-1,1)).reshape(-1)
                    else:
                        y_pred = y_pred_flat
                    pred_val = float(y_pred[0]) if len(y_pred) > 0 else float(overall_aqi)
                except Exception:
                    pred_val = float(overall_aqi)

  
                if overall_aqi is not None:
                    try:
                        next_day_aqi = float(overall_aqi) * 0.90
                    except Exception:
                        next_day_aqi = None
                else:
                    next_day_aqi = None

        except Exception as e:
            app.logger.debug("Next-day prediction pipeline failed: %s", e)
            next_day_aqi = None

        # Prepare minimal forecast for frontend
        forecast_list = []
        if forecast_ok and city_param:
            try:
                forecast_df = forecast_city_with_today(city_param, lstm_model, df, scaler_features, scaler_y, T_in=7, T_out=6)
                forecast_list = forecast_df.to_dict(orient="records")
            except Exception as e:
                app.logger.debug("Forecast generation skipped/failed: %s", e)
                forecast_list = []
        if (not forecast_list) and (next_day_aqi is not None):
            forecast_list = [{"Day":"Today","Predicted_AQI":overall_aqi},{"Day":"Day+1","Predicted_AQI":next_day_aqi}]

        predicted_aqi = next_day_aqi
        show_prediction = bool(predicted_aqi is not None)

        return jsonify({
            "aqi_value": int(overall_aqi),
            "aqi_category": aqi_category(overall_aqi),
            "components": comps,
            "timestamp": timestamp_iso,
            "next_day_aqi": next_day_aqi,
            "predicted_aqi": predicted_aqi,
            "show_prediction": show_prediction,
            "forecast": forecast_list,
            "resolved_via": resolved_via
        })

    except requests.exceptions.RequestException as re:
        app.logger.exception("WAQI request outer error: %s", re)
        return jsonify({"error": "Request error", "message": str(re)}), 502

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        app.logger.error("Unexpected error in get_aqi: %s\n%s", e, tb)
        return jsonify({
            "error": "Unexpected error",
            "message": str(e),
            "traceback": tb.splitlines()[-20:]
        }), 500

# ----------------------
# Symptom Logging
# ----------------------
@app.route('/log_symptom', methods=['POST'])
@login_required
def log_symptom():
    if current_user.role != 'patient':
        return jsonify({"error": "Only patients can log symptoms"}), 403

    data = request.get_json(silent=True) or request.form
    description = data.get('description')
    severity = data.get('severity', 'mild')
    lat = data.get('lat', None)
    lon = data.get('lon', None)

    # NEW: parse aqi from payload so HealthLog stores it (prevents missing-field errors)
    aqi_raw = data.get('aqi') or data.get('AQI') or data.get('air_quality') or None
    try:
        aqi_val = float(aqi_raw) if aqi_raw not in (None, '', 'null') else None
    except Exception:
        try:
            aqi_val = float(str(aqi_raw).replace(",", "")) if aqi_raw is not None else None
        except Exception:
            aqi_val = None

    if not description:
        return jsonify({"error": "description required"}), 400

    try:
        s = HealthLog(
            patient_id=current_user.id,
            symptoms=description,
            severity=severity,
            notes=None,
            timestamp=datetime.utcnow()
        )
        try:
            s.lat = float(lat) if lat else None
            s.lon = float(lon) if lon else None
        except Exception:
            pass

        # NEW: attach AQI (if present) to the HealthLog instance
        try:
            s.aqi = aqi_val
        except Exception:
            # If model doesn't have attribute, ignore silently (but typical HealthLog should have it)
            pass

        db.session.add(s)
        db.session.commit()
        return jsonify({"status": "ok", "symptom_id": s.id}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/my_symptoms', methods=['GET'])
@login_required
def my_symptoms():
    if current_user.role != 'patient':
        return jsonify({"error": "Only patients can view their symptoms"}), 403

    symptoms = HealthLog.query.filter_by(patient_id=current_user.id).order_by(HealthLog.timestamp.desc()).all()
    out = []
    for s in symptoms:
        out.append({
            "timestamp": s.timestamp.isoformat(),
            "description": s.symptoms,
            "severity": s.severity,
            "lat": getattr(s, 'lat', None),
            "lon": getattr(s, 'lon', None),
            "aqi": getattr(s, 'aqi', None)  # NEW: include stored AQI
        })
    return jsonify({"symptoms": out})

@app.route('/get_city_aqi', methods=['GET'])
@login_required
def get_city_aqi_full():
    # kept for compatibility if used elsewhere; forward to get_aqi which handles prediction and forecast
    return get_aqi()

@app.route('/patient_dashboard')
@login_required
def patient_dashboard():
    # Updated above in /dashboard; keep simple render here but include last_aqi for template compatibility
    last_log = HealthLog.query.filter_by(patient_id=current_user.id).order_by(HealthLog.timestamp.desc()).first()
    last_aqi = getattr(last_log, 'aqi', None) if last_log else None
    return render_template("patient_dashboard.html", last_aqi=last_aqi)  # Your HTML dashboard


# ----------------------
# ML endpoint: logistic regression patient tolerance
# ----------------------
@app.route('/predict_tolerance', methods=['POST'])
@login_required
def predict_tolerance():
    """
    Secure endpoint for patients to check AQI tolerance using logistic regression model.
    Expects JSON or form-data fields:
      - condition (Asthma/Bronchitis/COPD)
      - cough_severity, wheezing_severity, breathlessness_severity, chest_pain_severity,
        throat_irritation_severity, fatigue_severity (text or numeric)
      - aqi (numeric)
    """
    if current_user.role != 'patient':
        return jsonify({"success": False, "error": "Only patients can request predictions"}), 403

    # Ensure model & columns present
    if lr_model is None or preprocessing_columns is None:
        return jsonify({"success": False, "error": "ML model or preprocessing not loaded on server"}), 500

    # Accept JSON or form
    payload = request.get_json() if request.is_json else request.form.to_dict()

    # Normalize field names (adjust if your form uses different names)
    disease = (payload.get("condition") or payload.get("disease") or "").strip()

    # --- CHANGED: accept numeric severity values directly (1-5) and map common text labels ---
    _label_map = {
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
        "very low": 1, "verylow": 1, "low": 2, "mild": 2,
        "medium": 3, "med": 3, "moderate": 3,
        "high": 4, "severe": 4,
        "very high": 5, "veryhigh": 5
    }

    def _to_int_label(v, default=3):
        # Try direct int conversion
        try:
            return int(v)
        except Exception:
            pass
        if v is None:
            return default
        s = str(v).strip().lower()
        return _label_map.get(s, default)

    cough = _to_int_label(payload.get("cough_severity") or payload.get("cough"))
    wheezing = _to_int_label(payload.get("wheezing_severity") or payload.get("wheezing"))
    breathlessness = _to_int_label(payload.get("breathlessness_severity") or payload.get("breathlessness"))
    chest_pain = _to_int_label(payload.get("chest_pain_severity") or payload.get("chest_pain"))
    throat = _to_int_label(payload.get("throat_irritation_severity") or payload.get("throat_irritation"))
    fatigue = _to_int_label(payload.get("fatigue_severity") or payload.get("fatigue"))
    # --- END CHANGE ---

    # AQI numeric parsing (safe)
    aqi_raw = payload.get("aqi") or payload.get("AQI") or payload.get("air_quality")
    try:
        aqi = float(aqi_raw)
    except Exception:
        try:
            aqi = float(str(aqi_raw).replace(",", ""))
        except Exception:
            aqi = float("nan")

    try:
        # Call helper. pass the explicit model and preprocessing columns
        pred = predict_aqi_tolerance(
            disease=disease,
            cough_severity=cough,
            wheezing_severity=wheezing,
            breathlessness_severity=breathlessness,
            chest_pain_severity=chest_pain,
            throat_irritation_severity=throat,
            fatigue_severity=fatigue,
            aqi=aqi,
            model=lr_model,
            preprocessing_columns=preprocessing_columns
        )
    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"success": False, "error": str(e)}), 500

    # Normalize return value into a JSON-friendly type
    pred_val = None
    try:
        # If helper returns array-like or pandas values, attempt to cast to int then float
        if pred is None:
            pred_val = None
        else:
            try:
                pred_val = int(pred)
            except Exception:
                try:
                    pred_val = float(pred)
                except Exception:
                    # fallback to raw
                    pred_val = pred
    except Exception:
        pred_val = pred

    return jsonify({"success": True, "prediction": pred_val})



# --- Updated: /api/predict route with impute_strategy forwarding ---
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Simple severity mapping — adjust numeric values to match your model's training if needed
SEVERITY_MAP = {
    'none': 0.0, 'no': 0.0,
    'low': 1.0, 'mild': 1.0,
    'moderate': 2.0, 'med': 2.0,
    'high': 3.0, 'severe': 4.0, 'very high': 4.0, 'vhigh': 4.0
}

def _parse_severity(val, field_name):
    """Return float severity or raise ValueError for invalid input."""
    if val is None:
        raise ValueError(f"Missing value for {field_name}")

    # If already numeric (int/float)
    if isinstance(val, (int, float)):
        return float(val)

    # If string that looks numeric (like "2" or "2.0")
    if isinstance(val, str):
        s = val.strip()
        # empty string
        if s == "":
            raise ValueError(f"Empty string provided for {field_name}")
        # numeric string?
        try:
            return float(s)
        except Exception:
            pass
        # map known labels
        key = s.lower()
        if key in SEVERITY_MAP:
            return float(SEVERITY_MAP[key])
        # allow spelled out numbers accidentally, e.g., "one" -> not supported by default
        # if you want support, extend here
    raise ValueError(f"Cannot interpret {field_name} severity value: {val!r}")

def _safe_float(val):
    """Try to coerce to float; return np.nan on failure."""
    try:
        return float(val)
    except Exception:
        try:
            return float(str(val).replace(",", ""))
        except Exception:
            return float("nan")


@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """
    Improved prediction endpoint:
    - accepts JSON or form
    - patient-only
    - supports imputation strategies for missing AQI
    """
    if current_user.role != 'patient':
        return jsonify({"ok": False, "error": "Only patients can request predictions"}), 403

    # Accept JSON or form
    payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    if not payload:
        return jsonify({"ok": False, "error": "Invalid or empty payload"}), 400

    # normalize disease
    disease = (payload.get("condition") or payload.get("disease") or "").strip()

    # extract raw severity inputs (accepts different keys)
    # use None instead of defaulting to "High" so missing fields are caught/handled centrally
    cough_raw = payload.get("cough_severity") or payload.get("cough")
    wheezing_raw = payload.get("wheezing_severity") or payload.get("wheezing")
    breathlessness_raw = payload.get("breathlessness_severity") or payload.get("breathlessness")
    chest_pain_raw = payload.get("chest_pain_severity") or payload.get("chest_pain")
    throat_raw = payload.get("throat_irritation_severity") or payload.get("throat_irritation")
    fatigue_raw = payload.get("fatigue_severity") or payload.get("fatigue")

    # Try parsing severities now and return 400 with clear messages on failure
    try:
        cough = _parse_severity(cough_raw, "cough_severity")
        wheezing = _parse_severity(wheezing_raw, "wheezing_severity")
        breathlessness = _parse_severity(breathlessness_raw, "breathlessness_severity")
        chest_pain = _parse_severity(chest_pain_raw, "chest_pain_severity")
        throat = _parse_severity(throat_raw, "throat_irritation_severity")
        fatigue = _parse_severity(fatigue_raw, "fatigue_severity")
    except ValueError as e:
        app.logger.warning("Prediction input error: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 400

    # imputation options
    impute_strategy = (payload.get("impute_strategy") or "none").lower()
    fill_value = payload.get("fill_value", None)

    # Parse AQI with fallback to NaN
    aqi_raw = payload.get("aqi") or payload.get("AQI") or payload.get("air_quality")
    aqi = _safe_float(aqi_raw)

    # handle missing AQI
    if np.isnan(aqi):
        if impute_strategy in ("", "none", "no", None):
            missing = ["AQI"]
            app.logger.warning("Prediction input error: Missing/invalid values in input for columns: %s", missing)
            return jsonify({"ok": False, "error": f"Missing/invalid values in input for columns: {missing}"}), 400

        # Try median/mean using in-memory df if available
        if impute_strategy in ("median", "mean"):
            try:
                if 'df' in globals() and isinstance(df, pd.DataFrame) and 'aqi' in df.columns and not df['aqi'].dropna().empty:
                    aqi = float(df['aqi'].median() if impute_strategy == "median" else df['aqi'].mean())
                    app.logger.info("Imputed missing AQI from local dataset using %s -> %f", impute_strategy, aqi)
                else:
                    if fill_value is not None:
                        aqi = _safe_float(fill_value)
                        if np.isnan(aqi):
                            raise ValueError("Provided fill_value is not a valid number")
                        app.logger.info("Imputed missing AQI using provided fill_value -> %f", aqi)
                    else:
                        app.logger.warning("Cannot impute AQI: dataset missing and no fill_value provided")
                        return jsonify({"ok": False, "error": "AQI is required or provide impute_strategy with fill_value"}), 400
            except ValueError as e:
                return jsonify({"ok": False, "error": str(e)}), 400
            except Exception as e:
                app.logger.exception("AQI imputation failed: %s", e)
                return jsonify({"ok": False, "error": "AQI imputation failed"}), 500

        elif impute_strategy == "constant":
            if fill_value is None:
                return jsonify({"ok": False, "error": "fill_value required for constant imputation"}), 400
            aqi = _safe_float(fill_value)
            if np.isnan(aqi):
                return jsonify({"ok": False, "error": "Invalid fill_value for AQI constant imputation"}), 400
            app.logger.info("Imputed missing AQI using constant fill_value -> %f", aqi)
        else:
            return jsonify({"ok": False, "error": f"Unknown impute_strategy: {impute_strategy}"}), 400

    # Validate ML artifacts are loaded (these names must match your app globals)
    # lr_model and preprocessing_columns should be available on module/app-level
    try:
        lr = lr_model  # from your module scope
        prep_cols = preprocessing_columns
    except NameError:
        return jsonify({"ok": False, "error": "ML model or preprocessing not loaded on server"}), 500

    # Build input dataframe in the exact order your model expects (best-effort)
    # If preprocessing_columns exists, use it; otherwise default to common set
    expected_columns = prep_cols if prep_cols is not None else [
        'aqi',
        'cough_severity',
        'wheezing_severity',
        'breathlessness_severity',
        'chest_pain_severity',
        'throat_irritation_severity',
        'fatigue_severity'
    ]

    # Build a one-row DataFrame
    row = {
        'aqi': float(aqi),
        'cough_severity': float(cough),
        'wheezing_severity': float(wheezing),
        'breathlessness_severity': float(breathlessness),
        'chest_pain_severity': float(chest_pain),
        'throat_irritation_severity': float(throat),
        'fatigue_severity': float(fatigue)
    }

    # If expected_columns contains names your pipeline expects, reorder accordingly,
    # but ensure all required columns are present
    missing_cols = [c for c in expected_columns if c not in row]
    if missing_cols:
        app.logger.error("Preprocessing expects columns that are not provided: %s", missing_cols)
        return jsonify({"ok": False, "error": f"Missing required model columns: {missing_cols}"}), 500

    input_df = pd.DataFrame([[row[c] for c in expected_columns]], columns=expected_columns)
    app.logger.debug("Input DataFrame for prediction: %s", input_df.to_dict(orient='records')[0])

    # Run model prediction (support pipelines with predict_proba or plain classifiers)
    try:
        if hasattr(lr, "predict_proba"):
            proba_all = lr.predict_proba(input_df)
            classes = getattr(lr, "classes_", None)
            if classes is None:
                # fallback: interpret indices as str
                classes = list(range(proba_all.shape[1]))
            probs = dict(zip([str(c) for c in classes], proba_all[0].tolist()))
            pred_val = lr.predict(input_df)[0]
            return jsonify({"ok": True, "prediction": int(pred_val), "probabilities": probs}), 200
        else:
            pred_val = lr.predict(input_df)[0]
            return jsonify({"ok": True, "prediction": int(pred_val)}), 200
    except ValueError as e:
        app.logger.warning("Prediction input error: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        app.logger.exception("Unexpected prediction error")
        return jsonify({"ok": False, "error": "internal server error"}), 500

@app.route('/debug_model_features', methods=['GET'])
@login_required
def debug_model_features():
    # restrict to doctors or admins if you prefer
    if current_user.role not in ('doctor', 'patient'):
        return jsonify({"error": "forbidden"}), 403

    lr = app.config.get("LR_MODEL", None) or globals().get("lr_model", None)
    preproc = app.config.get("PREPROCESSING_COLUMNS", None) or globals().get("preprocessing_columns", None)

    resp = {
        "model_present": bool(lr),
        "preprocessing_present": bool(preproc),
        "model_type": str(type(lr)) if lr is not None else None,
        "preprocessing_type": str(type(preproc)) if preproc is not None else None,
        "feature_names_in": None,
        "preprocessing_sample": None,
        "preprocessing_count": None
    }

    # try to read model.feature_names_in_ safely
    try:
        if lr is not None:
            # if pipeline, present named_steps and try to pick feature_names_in_ from pipeline or final estimator
            if hasattr(lr, "feature_names_in_"):
                resp["feature_names_in"] = list(getattr(lr, "feature_names_in_"))
            else:
                # best-effort: check named_steps
                if hasattr(lr, "named_steps"):
                    for name, step in lr.named_steps.items():
                        if hasattr(step, "feature_names_in_"):
                            resp["feature_names_in"] = list(getattr(step, "feature_names_in_"))
                            resp["feature_names_source"] = f"pipeline.step:{name}"
                            break
                # final estimator
                if resp["feature_names_in"] is None and hasattr(lr, "steps"):
                    for _, step in lr.steps:
                        if hasattr(step, "feature_names_in_"):
                            resp["feature_names_in"] = list(getattr(step, "feature_names_in_"))
                            resp["feature_names_source"] = "pipeline.steps"
                            break
    except Exception as e:
        resp["feature_names_error"] = str(e)

    # preprocessing columns info
    try:
        if preproc is not None:
            # show first 40 entries only to avoid huge responses
            sample = list(preproc)[:40] if hasattr(preproc, "__iter__") else None
            resp["preprocessing_sample"] = sample
            try:
                resp["preprocessing_count"] = len(preproc)
            except Exception:
                resp["preprocessing_count"] = None
    except Exception as e:
        resp["preprocessing_error"] = str(e)

    # mask long lists (safe)
    if resp.get("feature_names_in") and len(resp["feature_names_in"]) > 200:
        resp["feature_names_in"] = resp["feature_names_in"][:200]

    return jsonify(resp)


# ----------------------
# Doctor endpoints and chat (kept as-is)
# ----------------------
@app.route('/patient_reports/<int:patient_id>', methods=['GET'])
@login_required
def patient_reports(patient_id):
    if current_user.role != 'doctor':
        return jsonify({"error": "Only doctors can access this endpoint"}), 403

    patient = User.query.get(patient_id)
    if not patient or patient.role != 'patient':
        return jsonify({"error": "Patient not found"}), 404

    reps = HealthLog.query.filter_by(patient_id=patient_id).order_by(HealthLog.timestamp.desc()).limit(100).all()
    out = [{
        "id": r.id,
            "timestamp": r.timestamp.isoformat(),
        "description": r.symptoms,
        "severity": r.severity,
        "lat": getattr(r, 'lat', None),
        "lon": getattr(r, 'lon', None),
        "aqi": getattr(r, 'aqi', None)  # NEW: include AQI in report
    } for r in reps]
    return jsonify({"patient": patient.username, "reports": out})

@app.route('/list_patients')
@login_required
def list_patients():
    if current_user.role != 'doctor':
        return jsonify({"error": "Only doctors can access"}), 403

    patients = User.query.filter_by(role='patient').all()
    out = []
    for p in patients:
        last = HealthLog.query.filter_by(patient_id=p.id).order_by(HealthLog.timestamp.desc()).first()
        out.append({
            "id": p.id,
            "username": p.username,
            "email": p.email,
            "last_symptom": last.symptoms if last else None,
            "last_severity": last.severity if last else None,
            "last_timestamp": last.timestamp.isoformat() if last else None,
            "last_aqi": getattr(last, 'aqi', None)  # NEW: include last AQI for quick glance
        })
    return jsonify({"patients": out})

@app.route('/chat_with_patient/<int:patient_id>', methods=['GET', 'POST'])
@login_required
def chat_with_patient(patient_id):
    if current_user.role != 'doctor':
        flash("Only doctors can access this page.", "danger")
        return redirect(url_for('dashboard'))

    patient = User.query.get(patient_id)
    if not patient or patient.role != 'patient':
        flash("Patient not found.", "danger")
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        content = request.form.get('message')
        if content:
            msg = Message(sender_id=current_user.id, receiver_id=patient.id, content=content)
            try:
                db.session.add(msg)
                db.session.commit()
            except Exception:
                db.session.rollback()
            return redirect(url_for('chat_with_patient', patient_id=patient_id))

    messages = Message.query.filter(
        ((Message.sender_id == current_user.id) & (Message.receiver_id == patient.id)) |
        ((Message.sender_id == patient.id) & (Message.receiver_id == current_user.id))
    ).order_by(Message.timestamp.asc()).all()

    return render_template('chat_with_patient.html', patient=patient, user=current_user, messages=messages)

# --------------
# Scheduler setup and start
# --------------
TRACKED_CITIES = ["Chennai", "Delhi", "Bengaluru"]

@scheduler.task("interval", id="hourly_aqi_collector", hours=1)
def hourly_aqi_collector():
    for city in TRACKED_CITIES:
        try:
            save_current_aqi(city)
        except Exception as e:
            app.logger.exception(f"AQI collect failed for {city}: {e}")

# Start the module-level scheduler (best-effort). This was in your original file.
try:
    scheduler.start()
except Exception as e:
    logger.warning("Warning: scheduler.start() raised during module import: %s", e)

def scheduled_job():
    cities = City.query.all()
    for city in cities:
        fetch_and_store_aqi(city.name, city.lat, city.lon)

# Replaced init_scheduler to reuse module-level scheduler (idempotent)
def init_scheduler(app_instance):
    """
    Initialize/start the module-level `scheduler` on the provided app_instance.
    Idempotent: it will not start the scheduler twice.
    """
    try:
        # If already initialized on this app, skip
        if getattr(app_instance, "_scheduler_initialized", False):
            app_instance.logger.info("Scheduler already initialized on app instance; skipping.")
            return

        # Ensure scheduler is attached to the app and started
        try:
            scheduler.init_app(app_instance)
        except Exception:
            # init_app may be called previously; ignore
            pass

        try:
            scheduler.start()
        except Exception as e:
            # start may have been called already or fail in some envs; log and continue
            app_instance.logger.warning("scheduler.start() raised: %s", e)

        app_instance._scheduler = scheduler
        app_instance._scheduler_initialized = True
        app_instance.logger.info("Module-level APScheduler initialized and started (or already running).")
    except Exception as e:
        app_instance.logger.exception("init_scheduler failed: %s", e)

# Register blueprints (attempt safely)
try:
    from backend_api import register_blueprint, init_scheduler as backend_init_scheduler
    register_blueprint(app)
    try:
        backend_init_scheduler(app)
    except Exception:
        pass
except Exception as e:
    logger.warning("Warning: backend_api registration failed: %s", str(e))

try:
    from ml_api import ml_bp
    app.register_blueprint(ml_bp, url_prefix="/api")
    logger.info("✅ ML API blueprint registered at /api")
except Exception as e:
    logger.warning("Warning: failed to register ml_api blueprint: %s", str(e))

# ----------------------
# New endpoints for LSTM + ANN models
# ----------------------
@app.route('/api/predict_aqi', methods=['POST'])
@login_required
def api_predict_aqi():
    """
    POST JSON:
    {
      "sequence": [[f1,f2,...], [f1,f2,...], ...]  # timesteps × features
    }
    """
    data = request.get_json(force=True)
    seq = data.get("sequence")
    if not seq:
        return jsonify({"error": "Missing 'sequence' in request"}), 400
    try:
        predicted_aqi = predict_next_day_aqi(seq)
        return jsonify({"predicted_aqi": predicted_aqi})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/get_suggestion', methods=['POST'])
@login_required
def api_get_suggestion():
    """
    POST JSON:
    {
      "aqi": 120,
      "cough": 2,
      "breathlessness": 1,
      "fatigue": 2,
      "throat_irritation": 1,
      "severity_score": 10,
      "feature_order": ["aqi","cough","breathlessness",
                        "fatigue","throat_irritation","severity_score"]
    }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Invalid or empty JSON body"}), 400

    feature_order = data.get("feature_order")
    try:
        result = generate_health_suggestion(data, feature_order=feature_order)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------
# Single robust app launcher (replaces duplicate __main__ blocks)
# ----------------------
if __name__ == '__main__':
    with app.app_context():
        # Create DB tables if not present (safe-guard)
        try:
            db.create_all()
            logger.info("✅ Database checked/created successfully!")
        except Exception as e:
            logger.warning("Warning: db.create_all() failed: %s", e)

        # try to init scheduler for this app (will reuse module-level scheduler)
        try:
            init_scheduler(app)
        except NameError:
            logger.warning("Scheduler init warning: init_scheduler() not defined.")
        except Exception as e:
            logger.warning("Scheduler init warning: %s", e)

    # Start Flask server
    try:
        app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
    except Exception as e:
        logger.exception("Failed to start Flask app: %s", e)
