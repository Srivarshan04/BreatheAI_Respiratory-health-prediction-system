# services/predictor_ann.py
"""
ANN predictor for generating health suggestions.
Loads:
 - utils/ann_model.h5         (required)
 - utils/scaler_symptom.pkl   (optional)  <-- if you used a separate scaler name use that or keep none
 - utils/label_encoder.pkl    (optional)  <-- if you used LabelEncoder during training
Usage:
  from services.predictor_ann import generate_health_suggestion
  result = generate_health_suggestion(input_dict, feature_order=[...])
  result -> dict { label, label_index, probabilities, suggestion }
"""
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UTILS_DIR = os.path.join(BASE, "utils")

ANN_MODEL_FNAME = "ann_model.h5"
SCALER_FNAME = "scaler_symptom.pkl"   # change if your scaler filename is different
LABEL_ENCODER_FNAME = "label_encoder.pkl"

ANN_MODEL_PATH = os.path.join(UTILS_DIR, ANN_MODEL_FNAME)
SCALER_PATH = os.path.join(UTILS_DIR, SCALER_FNAME)
LABEL_ENCODER_PATH = os.path.join(UTILS_DIR, LABEL_ENCODER_FNAME)

_ann = None
_scaler = None
_label_encoder = None

# Default feature order recommended for your ANN.
# Replace or pass feature_order when calling generate_health_suggestion if your training order differs.
DEFAULT_FEATURE_ORDER = [
    "aqi",
    "cough",
    "breathlessness",
    "fatigue",
    "throat_irritation",
    "severity_score"
]


def _load_once():
    global _ann, _scaler, _label_encoder
    if _ann is None:
        if not os.path.exists(ANN_MODEL_PATH):
            raise FileNotFoundError(f"ANN model not found at {ANN_MODEL_PATH}")
        _ann = load_model(ANN_MODEL_PATH)
    if _scaler is None and os.path.exists(SCALER_PATH):
        try:
            with open(SCALER_PATH, "rb") as f:
                _scaler = pickle.load(f)
        except Exception:
            _scaler = None
    if _label_encoder is None and os.path.exists(LABEL_ENCODER_PATH):
        try:
            with open(LABEL_ENCODER_PATH, "rb") as f:
                _label_encoder = pickle.load(f)
        except Exception:
            _label_encoder = None


def _suggestion_text(label):
    mapping = {
        "Low Risk": "Air quality acceptable. Normal activities OK.",
        "Moderate Risk": "Limit prolonged outdoor exposure. Wear mask if sensitive.",
        "High Risk": "Stay indoors, use air purifier, avoid strenuous activity.",
        "Critical": "Stay inside; seek medical attention if symptoms worsen."
    }
    return mapping.get(label, "Monitor AQI and symptoms; take precautions.")


def generate_health_suggestion(input_dict, feature_order=None):
    """
    input_dict: dict containing numeric features. Must include 'aqi' key.
    feature_order: list specifying exact column order the ANN expects.
                   If not provided, DEFAULT_FEATURE_ORDER is used and missing keys will raise.
    Returns:
      { label, label_index, probabilities, suggestion }
    """
    _load_once()

    if feature_order is None:
        feature_order = DEFAULT_FEATURE_ORDER

    # Validate presence of keys
    missing = [k for k in feature_order if k not in input_dict]
    if missing:
        raise KeyError(f"Missing keys in input_dict required by feature_order: {missing}")

    # Build input vector in the exact order
    X = np.array([[float(input_dict[k]) for k in feature_order]], dtype=float)

    # Apply scaler if present
    if _scaler is not None:
        try:
            X = _scaler.transform(X)
        except Exception:
            print("[predictor_ann] Warning: scaler.transform failed; using raw features")

    preds = _ann.predict(X)
    probs = preds[0].tolist() if hasattr(preds, "__len__") else [float(preds)]

    # Determine predicted index/label
    if len(probs) == 1:
        idx = 1 if probs[0] >= 0.5 else 0
    else:
        idx = int(np.argmax(probs))

    label = idx
    if _label_encoder is not None:
        try:
            label = _label_encoder.inverse_transform([idx])[0]
        except Exception:
            label = idx

    suggestion = _suggestion_text(label)
    return {"label": label, "label_index": int(idx), "probabilities": probs, "suggestion": suggestion}
