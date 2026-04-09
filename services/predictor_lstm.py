# services/predictor_lstm.py
"""
LSTM predictor for next-day AQI.
Loads:
 - utils/lstm_model.h5    (required)
 - utils/scaler.pkl       (optional)  -- scaler used for input features (fit on feature columns)
Usage:
  from services.predictor_lstm import predict_next_day_aqi, ensure_timesteps
  pred = predict_next_day_aqi(sequence, expected_timesteps=7)
"""
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UTILS_DIR = os.path.join(BASE, "utils")

LSTM_MODEL_FNAME = "lstm_model.h5"
SCALER_FNAME = "scaler.pkl"

LSTM_MODEL_PATH = os.path.join(UTILS_DIR, LSTM_MODEL_FNAME)
SCALER_PATH = os.path.join(UTILS_DIR, SCALER_FNAME)

_lstm = None
_scaler = None


def _load_once():
    global _lstm, _scaler
    if _lstm is None:
        if not os.path.exists(LSTM_MODEL_PATH):
            raise FileNotFoundError(f"LSTM model not found at {LSTM_MODEL_PATH}")
        _lstm = load_model(LSTM_MODEL_PATH)
    if _scaler is None and os.path.exists(SCALER_PATH):
        try:
            with open(SCALER_PATH, "rb") as f:
                _scaler = pickle.load(f)
        except Exception:
            _scaler = None


def ensure_timesteps(seq, expected=7, pad_value=None):
    """
    Ensure seq (list of lists) has length == expected.
    If shorter, pad by repeating the last row (or pad_value).
    If longer, keep the last `expected` timesteps (most recent).
    """
    if not isinstance(seq, list):
        raise ValueError("Sequence must be a list of timesteps (lists).")
    if len(seq) == 0:
        raise ValueError("Empty sequence provided.")
    s = list(seq)
    if pad_value is None:
        pad_value = s[-1]
    if len(s) < expected:
        while len(s) < expected:
            s.insert(0, pad_value)
    elif len(s) > expected:
        s = s[-expected:]
    return s


def predict_next_day_aqi(sequence, expected_timesteps=None):
    """
    Predict next-day AQI.
    Args:
      sequence: list (timesteps x features) or numpy array.
      expected_timesteps: int or None. If provided, sequence will be padded/truncated to this length.
    Returns:
      float predicted AQI (if model returns single value) or list (if model returns multi-value).
    """
    _load_once()
    X = np.array(sequence, dtype=float)

    # if expected_timesteps provided, enforce it by padding/truncating
    if expected_timesteps is not None:
        # convert to list-of-lists first to use ensure_timesteps
        if X.ndim == 3:
            # assume shape (1, timesteps, features)
            X = X.reshape(X.shape[1], X.shape[2])
        seq_list = X.tolist()
        seq_list = ensure_timesteps(seq_list, expected=expected_timesteps)
        X = np.array(seq_list, dtype=float)

    # normalize shape to (1, timesteps, features)
    if X.ndim == 2:
        X = X.reshape(1, X.shape[0], X.shape[1])
    elif X.ndim == 3:
        pass
    else:
        raise ValueError("sequence must be shape (timesteps, features) or (1, timesteps, features)")

    # apply scaler if present (scaler expected shape: (n_samples, n_features) -> we reshape)
    if _scaler is not None:
        b, t, f = X.shape
        flat = X.reshape(-1, f)  # (timesteps, features) if b==1
        try:
            flat_scaled = _scaler.transform(flat)
            X = flat_scaled.reshape(b, t, f)
        except Exception:
            # If transform fails, keep original X but warn
            print("[predictor_lstm] Warning: scaler.transform failed; using raw features")

    preds = _lstm.predict(X)
    # return scalar if model outputs single value
    try:
        val = float(np.array(preds).flatten()[0])
        return val
    except Exception:
        return preds.tolist()
