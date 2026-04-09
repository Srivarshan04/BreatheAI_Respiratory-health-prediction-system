# routes/ml_routes.py
from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from utils.predict_aqi_tolerance import predict_aqi_tolerance

ml_bp = Blueprint("ml", __name__)

def _parse_payload(payload):
    """Normalize incoming payload keys to expected variable names."""
    disease = (payload.get("condition") or payload.get("disease") or "").strip()
    cough = payload.get("cough_severity") or payload.get("cough") or "High"
    wheezing = payload.get("wheezing_severity") or payload.get("wheezing") or "High"
    breathlessness = payload.get("breathlessness_severity") or payload.get("breathlessness") or "High"
    chest_pain = payload.get("chest_pain_severity") or payload.get("chest_pain") or "High"
    throat = payload.get("throat_irritation_severity") or payload.get("throat_irritation") or "High"
    fatigue = payload.get("fatigue_severity") or payload.get("fatigue") or "High"

    aqi_raw = payload.get("aqi") or payload.get("AQI") or payload.get("air_quality")
    try:
        aqi = float(aqi_raw)
    except Exception:
        try:
            aqi = float(str(aqi_raw).replace(",", ""))
        except Exception:
            aqi = float("nan")

    return {
        "disease": disease,
        "cough": cough,
        "wheezing": wheezing,
        "breathlessness": breathlessness,
        "chest_pain": chest_pain,
        "throat": throat,
        "fatigue": fatigue,
        "aqi": aqi,
    }


@ml_bp.route("/predict_tolerance", methods=["POST"])
@login_required
def predict_tolerance():
    """Protected endpoint for patient prediction (uses Flask-Login session)."""
    if current_user.role != "patient":
        return jsonify({"success": False, "error": "Only patients can request predictions"}), 403

    payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    if not payload:
        return jsonify({"success": False, "error": "No input provided"}), 400

    p = _parse_payload(payload)
    try:
        pred = predict_aqi_tolerance(
            disease=p["disease"],
            cough_severity=p["cough"],
            wheezing_severity=p["wheezing"],
            breathlessness_severity=p["breathlessness"],
            chest_pain_severity=p["chest_pain"],
            throat_irritation_severity=p["throat"],
            fatigue_severity=p["fatigue"],
            aqi=p["aqi"],
        )
    except Exception as e:
        current_app.logger.exception("Prediction failed")
        return jsonify({"success": False, "error": str(e)}), 500

    try:
        pred_int = int(pred)
    except Exception:
        pred_int = pred

    return jsonify({"success": True, "prediction": pred_int})


# Backwards-compatible alias for clients that POST to /api/predict
@ml_bp.route("/api/predict", methods=["POST"])
@login_required
def api_predict_alias():
    return predict_tolerance()


# Debug endpoint for local testing (NOT protected) — remove before production
@ml_bp.route("/debug/predict_tolerance", methods=["POST"])
def debug_predict_tolerance():
    payload = request.get_json(force=True) if request.is_json else request.form.to_dict()
    if not payload:
        return jsonify({"success": False, "error": "No input provided"}), 400
    p = _parse_payload(payload)
    try:
        pred = predict_aqi_tolerance(
            disease=p["disease"],
            cough_severity=p["cough"],
            wheezing_severity=p["wheezing"],
            breathlessness_severity=p["breathlessness"],
            chest_pain_severity=p["chest_pain"],
            throat_irritation_severity=p["throat"],
            fatigue_severity=p["fatigue"],
            aqi=p["aqi"],
        )
    except Exception as e:
        current_app.logger.exception("Debug prediction failed")
        return jsonify({"success": False, "error": str(e)}), 500

    try:
        pred_int = int(pred)
    except Exception:
        pred_int = pred
    return jsonify({"success": True, "prediction": pred_int})
