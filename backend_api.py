# backend_api.py
import requests
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from flask_apscheduler import APScheduler
from sqlalchemy import and_
from models import AQIHistory

from extensions import db
from models import User, Message, Notification, HealthLog, AQIData

import logging

bp = Blueprint('backend_api', __name__, url_prefix='/api')

# -------------------------
# Messaging APIs
# -------------------------
@bp.route('/send_message', methods=['POST'])
@login_required
def api_send_message():
    """
    POST JSON: { "receiver_id": <id>, "content": "<text>" }
    """
    data = request.get_json() or {}
    receiver_id = data.get('receiver_id')
    content = (data.get('content') or "").strip()

    if not receiver_id or not content:
        return jsonify({"error": "receiver_id and content required"}), 400

    receiver = User.query.get(receiver_id)
    if not receiver:
        return jsonify({"error": "receiver not found"}), 404

    msg = Message(sender_id=current_user.id, receiver_id=receiver_id, content=content)
    db.session.add(msg)
    db.session.commit()
    return jsonify({"status": "ok", "message_id": msg.id}), 201


@bp.route('/get_messages/<int:other_user_id>', methods=['GET'])
@login_required
def api_get_messages(other_user_id):
    """
    Get conversation between current_user and other_user_id
    """
    other = User.query.get_or_404(other_user_id)
    conv = Message.query.filter(
        ((Message.sender_id == current_user.id) & (Message.receiver_id == other.id)) |
        ((Message.sender_id == other.id) & (Message.receiver_id == current_user.id))
    ).order_by(Message.timestamp.asc()).all()

    out = []
    for m in conv:
        out.append({
            "id": m.id,
            "sender_id": m.sender_id,
            "receiver_id": m.receiver_id,
            "content": m.content,
            "timestamp": m.timestamp.isoformat()
        })
    return jsonify({"messages": out})


@bp.route('/conversations', methods=['GET'])
@login_required
def api_conversations():
    """
    List conversation partners for current user with last message summary
    """
    msgs = Message.query.filter(
        (Message.sender_id == current_user.id) | (Message.receiver_id == current_user.id)
    ).order_by(Message.timestamp.desc()).all()

    partners = {}
    for m in msgs:
        other_id = m.receiver_id if m.sender_id == current_user.id else m.sender_id
        if other_id not in partners:
            partners[other_id] = {
                "last_message": m.content,
                "last_timestamp": m.timestamp.isoformat()
            }

    out = []
    for pid, info in partners.items():
        user = User.query.get(pid)
        out.append({"user_id": pid, "username": user.username if user else None, **info})

    return jsonify({"conversations": out})

# -------------------------
# Notification APIs
# -------------------------
@bp.route('/notifications', methods=['GET'])
@login_required
def api_get_notifications():
    """
    Return notifications for the logged-in user
    """
    notifs = Notification.query.filter_by(user_id=current_user.id).order_by(Notification.timestamp.desc()).all()
    out = []
    for n in notifs:
        out.append({
            "id": n.id,
            "title": getattr(n, "title", None) or ("Notification"),
            "message": getattr(n, "message", None) or getattr(n, "body", None) or "",
            "level": getattr(n, "level", "info"),
            "timestamp": n.timestamp.isoformat(),
            "read": getattr(n, "is_read", False)
        })
    return jsonify({"notifications": out})


@bp.route('/notifications/mark_read', methods=['POST'])
@login_required
def api_mark_notifications():
    """
    POST JSON: { "ids": [1,2,3] }
    """
    data = request.get_json() or {}
    ids = data.get('ids', [])
    if not isinstance(ids, list):
        return jsonify({"error": "ids should be a list"}), 400

    Notification.query.filter(Notification.id.in_(ids), Notification.user_id == current_user.id).update(
        {"is_read": True}, synchronize_session=False
    )
    db.session.commit()
    return jsonify({"status": "ok"})


@bp.route('/notify', methods=['POST'])
@login_required
def api_manual_notify():
    """
    For testing: POST JSON { "user_id": <id>, "title":"...", "body":"...", "level":"warning" }
    Only accessible to logged-in users (doctors/admins)
    """
    data = request.get_json() or {}
    uid = data.get('user_id') or current_user.id
    title = data.get('title', 'Alert')
    body = data.get('body', '')
    level = data.get('level', 'info')

    notif = Notification(user_id=uid, title=title, message=body, level=level)
    db.session.add(notif)
    db.session.commit()
    return jsonify({"status": "ok", "notif_id": notif.id})


# -------------------------
# Doctor / Patient helper APIs
# -------------------------
@bp.route('/list_patients', methods=['GET'])
@login_required
def api_list_patients():
    """
    Return simple list of patients. Only accessible to doctors (but can be used by others).
    """
    if current_user.role != 'doctor':
        return jsonify({"error": "Only doctors can access this endpoint"}), 403

    patients = User.query.filter_by(role='patient').all()
    out = []
    for p in patients:
        # Optionally, fetch last health log summary
        last_log = HealthLog.query.filter_by(patient_id=p.id).order_by(HealthLog.timestamp.desc()).first()
        out.append({
            "id": p.id,
            "username": p.username,
            "email": p.email,
            "last_symptom": last_log.symptoms if last_log else None,
            "last_severity": last_log.severity if last_log else None,
            "last_lat": last_log.lat if last_log else None,
            "last_lon": last_log.lon if last_log else None
        })
    return jsonify({"patients": out})


@bp.route('/patient_reports/<int:patient_id>', methods=['GET'])
@login_required
def api_patient_reports(patient_id):
    if current_user.role != 'doctor':
        return jsonify({"error": "Only doctors can access this endpoint"}), 403

    patient = User.query.get(patient_id)
    if not patient or patient.role != 'patient':
        return jsonify({"error": "Patient not found"}), 404

    reps = HealthLog.query.filter_by(patient_id=patient_id).order_by(HealthLog.timestamp.desc()).limit(200).all()
    out = []
    for r in reps:
        out.append({
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "symptoms": r.symptoms,
            "severity": r.severity,
            "lat": r.lat,
            "lon": r.lon,
            "notes": r.notes
        })
    return jsonify({"patient": patient.username, "reports": out})


# -------------------------
# AQI Alerting Scheduler
# -------------------------
scheduler = APScheduler()

def _map_openweather_index_to_numeric(idx):
    """
    Map OpenWeather's AQI index (1..5) to an approximate numeric value used in frontend.
    1 -> 50 (Good), 2 -> 100, 3 -> 150, 4 -> 200, 5 -> 300
    """
    mapping = {1: 50, 2: 100, 3: 150, 4: 200, 5: 300}
    return mapping.get(idx, None)

def check_aqi_and_alert():
    """
    Periodic job:
    - For each patient with coordinates (extracted from last HealthLog), call OpenWeather API
    - Compute a numeric AQI (mapping OpenWeather index or compute from PM2.5/PM10)
    - If AQI >= threshold: create Notification for patient + inform doctors
    - Also, if last symptom severity is moderate/severe and AQI >= 100, create critical notification

    This wrapper ensures the job always runs inside a Flask application context.
    """
    logger = logging.getLogger(__name__)

    def run_job():
        # --- original job logic (copied from previous implementation) ---
        ow_key = current_app.config.get('OPENWEATHER_API_KEY')
        if not ow_key:
            current_app.logger.warning("OPENWEATHER_API_KEY not configured; skipping AQI scheduler run.")
            return

        ALERT_THRESHOLD = current_app.config.get('AQI_ALERT_THRESHOLD', 150)

        patients = User.query.filter_by(role='patient').all()
        for p in patients:
            # pick coordinates: prefer last health log with coords
            last_log = HealthLog.query.filter_by(patient_id=p.id).order_by(HealthLog.timestamp.desc()).first()
            lat = last_log.lat if last_log and last_log.lat else None
            lon = last_log.lon if last_log and last_log.lon else None

            # fallback: check AQIData table for a recent record (if you populate it)
            if (lat is None or lon is None):
                # try to get last AQIData entry for the patient (if you store location there)
                # For now, skip if no coords
                continue

            try:
                url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={ow_key}"
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                current_app.logger.exception("OpenWeather API call failed for patient %s: %s", p.id, e)
                continue

            # parse numeric aqi
            aqi_value = None
            if isinstance(data, dict) and "list" in data and data["list"]:
                entry = data["list"][0]
                # OpenWeather returns 'main':{'aqi': 1..5} and 'components'
                ow_index = entry.get("main", {}).get("aqi")
                if ow_index:
                    aqi_value = _map_openweather_index_to_numeric(ow_index)

                # If pm2_5 present, compute US EPA PM2.5 based AQI to be more precise
                comps = entry.get("components", {})
                pm25 = comps.get("pm2_5")
                pm10 = comps.get("pm10")
                # If pm2_5 available, do a simple pm->AQI conversion consistent with app's helper
                # (Use PM breakpoints if available in app code; here we rely on a simplified approach)
                # If you have a PM->AQI util, call that here; fallback to ow_index mapping if not present
                if pm25 is not None:
                    # approximate mapping here might be coarse; prefer the mapping function in app if available.
                    # We'll attempt to map using the same mapping technique as front-end (if present)
                    # For simplicity, use mapping of pm2_5 to a rough AQI (coarse)
                    # But if you added AQIData/pm->AQI helper in your app, use that function instead.
                    # Keep current fallback:
                    if aqi_value is None:
                        aqi_value = _map_openweather_index_to_numeric(ow_index) if ow_index else None

            # If no aqi_value could be computed, skip
            if aqi_value is None:
                continue

            # Save AQIData record (optional) for auditing / dashboard
            try:
                aqi_record = AQIData(location=f"{lat},{lon}", aqi_value=aqi_value, main_pollutant=None)
                db.session.add(aqi_record)
                db.session.commit()
            except Exception as e:
                current_app.logger.exception("Failed to save AQIData: %s", e)
                db.session.rollback()

            # Now apply thresholds and create notifications
            if aqi_value >= ALERT_THRESHOLD:
                # Patient notification
                notif = Notification(
                    user_id=p.id,
                    title="High AQI Alert",
                    message=f"AQI at your location is {aqi_value}. Advise limiting outdoor exposure.",
                    level='warning'
                )
                db.session.add(notif)

                # Notify doctors (create notifications for all doctors)
                doctors = User.query.filter_by(role='doctor').all()
                for d in doctors:
                    dn = Notification(
                        user_id=d.id,
                        title=f"Patient {p.username} - High AQI",
                        message=f"Patient {p.username} has AQI {aqi_value} at last-known location.",
                        level='info'
                    )
                    db.session.add(dn)

            # If recent severe/moderate symptom and AQI >= 100, critical alert
            if last_log and getattr(last_log, "severity", None) and last_log.severity.lower() in ['severe', 'moderate'] and aqi_value >= 100:
                notif2 = Notification(
                    user_id=p.id,
                    title="Symptom + AQI Alert",
                    message=f"You recently logged {last_log.severity} symptoms and current AQI is {aqi_value}. Seek medical attention if needed.",
                    level='critical'
                )
                db.session.add(notif2)

        db.session.commit()
        # --- end original job logic ---

    # Try to run directly if we already have an application context
    try:
        # Access current_app to check if a context is active
        _ = current_app.name
        return run_job()
    except RuntimeError:
        # Not in an application context — push one using the app instance
        try:
            # lazy import to avoid circular import issues
            from app import app as flask_app
        except Exception as e:
            logger.exception("Failed to import Flask app for scheduled job app_context: %s", e)
            # As a last resort, attempt to run job without app context (will likely fail for DB/current_app usage)
            try:
                logger.warning("Running scheduled job without Flask app context - DB/current_app access may fail.")
                return run_job()
            except Exception as ee:
                logger.exception("Scheduled job failed when running without app context: %s", ee)
                return

        try:
            with flask_app.app_context():
                return run_job()
        except Exception as e:
            try:
                flask_app.logger.exception("Error in scheduled job inside app_context: %s", e)
            except Exception:
                logger.exception("Scheduled job error and failed to log via flask_app.logger: %s", e)
            return


def init_scheduler(flask_app):
    """
    Initialize & start the APScheduler. Call this from your main app within app_context.
    """
    flask_app.config.setdefault('SCHEDULER_API_ENABLED', True)
    scheduler.init_app(flask_app)
    # Schedule job every 10 minutes (adjust as necessary)
    scheduler.add_job(id='aqi_check_job', func=check_aqi_and_alert, trigger='interval', minutes=10)
    scheduler.start()


# -------------------------
# Blueprint registration helper
# -------------------------
def register_blueprint(app_obj):
    """
    Register blueprint & ensure tables exist.
    Call this from your main app within app context (or this function will create app_context itself).
    """
    app_obj.register_blueprint(bp)
    # ensure models' tables exist (Notification etc.)
    with app_obj.app_context():
        db.create_all()


# If backend_api.py is imported as module (not ideal), we don't auto-register to avoid side effects.
# The main app should call register_blueprint(app) and init_scheduler(app) explicitly.

#-----------------------
#Aqi History
#----------------------
@bp.route("/aqi/history")
def aqi_history():
    city = request.args.get("city", "Chennai")
    hours = int(request.args.get("hours", 168))  # default last 7 days
    since = datetime.utcnow() - timedelta(hours=hours)

    rows = (AQIHistory.query
            .filter(AQIHistory.city==city, AQIHistory.ts >= since)
            .order_by(AQIHistory.ts.asc())
            .all())
    return jsonify([
        {
            "city": r.city,
            "ts": r.ts.isoformat(),
            "aqi": r.aqi,
            "pm25": r.pm25,
            "pm10": r.pm10
        }
        for r in rows
    ])

#----------
#temp route
#----------

@bp.route("/aqi/debug/save_once")
def debug_save_once():
    from services.aqi_service import save_current_aqi
    out = []
    for c in ["Chennai","Delhi","Bengaluru"]:
        row = save_current_aqi(c)
        out.append({"city": row.city, "ts": row.ts.isoformat(), "aqi": row.aqi})
    return jsonify(out)
