# ml_api.py
import os
import traceback
import joblib
import numpy as np
import pandas as pd
from flask import Blueprint, current_app, request, jsonify

ml_bp = Blueprint("ml_api", __name__)

# Configure paths (override with env var if needed)
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "/content/drive/MyDrive/breatheai/logistic_regression_model.pkl")
DEFAULT_LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH", "/content/drive/MyDrive/breatheai/label_encoder.pkl")
# Optional scaler path (if you saved scalers separately)
DEFAULT_SCALER_PATH = os.getenv("ML_SCALER_PATH", "/content/drive/MyDrive/breatheai/scaler_features.pkl")


def load_ml_model(model_path=DEFAULT_MODEL_PATH, label_encoder_path=DEFAULT_LABEL_ENCODER_PATH, scaler_path=DEFAULT_SCALER_PATH):
    """
    Load model, optional label encoder, and optional scaler.
    Returns (model, label_encoder or None, scaler or None).
    """
    model = None
    le = None
    scaler = None
    try:
        model = joblib.load(model_path)
    except Exception as e:
        # If joblib fails, try pickle or raise
        try:
            import pickle
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        except Exception:
            raise RuntimeError(f"Failed to load model at {model_path}: {e}")

    if os.path.exists(label_encoder_path):
        try:
            le = joblib.load(label_encoder_path)
        except Exception:
            le = None

    # load scaler if provided and exists (optional)
    if scaler_path and os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            scaler = None

    return model, le, scaler


# lazy load into blueprint state when first request comes
def get_model():
    # cache on current_app.config to avoid reloading
    if current_app.config.get("ML_MODEL") is None:
        try:
            model, le, scaler = load_ml_model()
            current_app.config["ML_MODEL"] = model
            current_app.config["ML_LABEL_ENCODER"] = le
            current_app.config["ML_SCALER"] = scaler
            current_app.logger.info("ML model loaded from default path.")
        except Exception as e:
            current_app.logger.error(f"Failed to load ML model: {e}")
            current_app.config["ML_MODEL"] = None
            current_app.config["ML_LABEL_ENCODER"] = None
            current_app.config["ML_SCALER"] = None
    return current_app.config.get("ML_MODEL"), current_app.config.get("ML_LABEL_ENCODER")


def validate_input_json(data):
    """
    Required fields updated to match the patient dashboard inputs.
    """
    required = [
        "disease",
        "cough_severity",
        "wheezing_severity",
        "breathlessness_severity",
        "chest_pain_severity",
        "throat_irritation_severity",
        "fatigue_severity",
        "current_aqi"
    ]
    for r in required:
        if r not in data:
            return False, f"Missing required field: {r}"
    return True, ""


def prepare_features(data, label_encoder=None):
    """
    Convert incoming payload dict to features for the model.

    - Builds a DataFrame row with fields coming from the dashboard.
    - If a sklearn Pipeline is loaded (model has .named_steps or .steps),
      returns the DataFrame row so the pipeline handles encoding/scaling.
    - Otherwise returns a numpy array matching feature_order and applies scaler if available.

    Returns (X, feature_names) where X can be a DataFrame or numpy array.
    """

    # Build canonical payload with defaults
    # Accept both numeric-coded severities and string-like (we'll coerce later)
    payload = {
        "disease": data.get("disease", ""),
        "cough_severity": data.get("cough_severity", 0),
        "wheezing_severity": data.get("wheezing_severity", 0),
        "breathlessness_severity": data.get("breathlessness_severity", 0),
        "chest_pain_severity": data.get("chest_pain_severity", 0),
        "throat_irritation_severity": data.get("throat_irritation_severity", 0),
        "fatigue_severity": data.get("fatigue_severity", 0),
        "current_aqi": data.get("current_aqi", 0)
    }

    df_row = pd.DataFrame([payload])

    # If label encoder present, try to create an encoded disease column.
    if label_encoder is not None and "disease" in df_row.columns:
        try:
            # label_encoder.transform expects array-like
            df_row["disease_encoded"] = label_encoder.transform(df_row["disease"])
            # Note: depending on your model, you might want to drop the original disease column.
        except Exception:
            # leave disease column as-is if encoder fails
            current_app.logger.debug("Label encoder failed to transform disease; leaving original value.")

    # Define the feature order expected by your non-pipeline model.
    feature_order = [
        # If your model expects encoded disease instead of string, ensure "disease_encoded" is used.
        # "disease_encoded",
        "disease",
        "cough_severity",
        "wheezing_severity",
        "breathlessness_severity",
        "chest_pain_severity",
        "throat_irritation_severity",
        "fatigue_severity",
        "current_aqi"
    ]

    # If the loaded model is a sklearn Pipeline, return DataFrame row (pipeline will handle transforms)
    model = current_app.config.get("ML_MODEL")
    is_pipeline = False
    try:
        if model is not None and (hasattr(model, "named_steps") or hasattr(model, "steps")):
            is_pipeline = True
    except Exception:
        is_pipeline = False

    if is_pipeline:
        # Return the DataFrame so the pipeline can accept column names and apply encoders/scalers.
        return df_row, feature_order

    # Otherwise produce numpy array in the required order
    available = [c for c in feature_order if c in df_row.columns]
    if not available:
        X = df_row.values
    else:
        # try to coerce numeric columns; keep disease as-is if present
        df_numeric = df_row.copy()
        for col in available:
            if col != "disease":
                df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce').fillna(0)
        X = df_numeric[available].values

    # Apply scaler if present (optional)
    scaler = current_app.config.get("ML_SCALER")
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            # if scaler fails, continue with unscaled X
            current_app.logger.warning("Scaler present but transform failed; continuing without scaler.")

    return X, available


@ml_bp.route("/health", methods=["GET"])
def health():
    model, _ = get_model()
    ok = model is not None
    return jsonify({"status": "ok" if ok else "error", "model_loaded": ok})


@ml_bp.route("/predict", methods=["POST"])
def predict():
    model, label_encoder = get_model()
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    # Parse JSON safely and log
    try:
        payload = request.get_json(force=True)
        current_app.logger.info("🔍 /api/predict received payload: %s", repr(payload))
        if not payload:
            current_app.logger.warning("⚠️ /api/predict - Empty or invalid JSON payload received.")
            return jsonify({"error": "Empty or invalid JSON payload"}), 400
    except Exception:
        current_app.logger.exception("Failed to parse JSON payload for /api/predict")
        return jsonify({"error": "Invalid JSON"}), 400

    single = isinstance(payload, dict)
    items = [payload] if single else payload

    preds = []
    probs = []
    errors = []

    for i, item in enumerate(items):
        ok, msg = validate_input_json(item)
        current_app.logger.info("🔍 Validation result for item %d: %s, message: %s", i, ok, msg)

        if not ok:
            errors.append({"index": i, "error": msg})
            preds.append(None)
            probs.append(None)
            continue

        try:
            X, feat_names = prepare_features(item, label_encoder)
            # If prepare_features returned a DataFrame (pipeline case), pass it directly to predict
            if isinstance(X, pd.DataFrame):
                X_input = X
            else:
                X_input = X
                if hasattr(X_input, "ndim") and X_input.ndim == 1:
                    X_input = X_input.reshape(1, -1)

            # Debug: log the prepared input shape / columns
            try:
                if isinstance(X_input, pd.DataFrame):
                    current_app.logger.debug("Prepared DataFrame for model:\n%s", X_input.to_string())
                else:
                    current_app.logger.debug("Prepared numpy array for model shape=%s, features=%s", getattr(X_input, "shape", None), feat_names)
            except Exception:
                pass

            # If model is a scikit-learn pipeline or accepts DataFrame, pass X_input directly
            y_pred = model.predict(X_input)[0]
            prob = None
            if hasattr(model, "predict_proba"):
                try:
                    prob = float(model.predict_proba(X_input).max(axis=1)[0])
                except Exception:
                    prob = None
            else:
                # fallback: decision_function if available
                if hasattr(model, "decision_function"):
                    try:
                        dfun = model.decision_function(X_input)
                        # decision_function can return scalar or array; handle common cases
                        if np.isscalar(dfun):
                            prob = float(dfun)
                        elif hasattr(dfun, "__len__") and len(dfun.shape) == 1:
                            prob = float(dfun[0])
                        else:
                            prob = None
                    except Exception:
                        prob = None

            # decode label if label_encoder exists and output is int-coded
            try:
                if label_encoder is not None:
                    try:
                        pred_decoded = label_encoder.inverse_transform([y_pred])[0]
                    except Exception:
                        pred_decoded = y_pred
                else:
                    pred_decoded = y_pred
            except Exception:
                pred_decoded = y_pred

            preds.append(pred_decoded)
            probs.append(prob)
        except Exception as e:
            current_app.logger.error(traceback.format_exc())
            errors.append({"index": i, "error": str(e)})
            preds.append(None)
            probs.append(None)

    code = 200 if not errors or len(errors) < len(items) else 400
    return jsonify({"predictions": preds, "probabilities": probs, "errors": errors}), code
