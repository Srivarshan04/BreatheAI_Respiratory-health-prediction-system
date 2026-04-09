# models.py
from extensions import db
from flask_login import UserMixin
from datetime import datetime

# ------------------------------
# USER MODEL
# ------------------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(10), nullable=False)  # 'patient' or 'doctor'

    # Relationships
    health_logs = db.relationship('HealthLog', backref='patient', lazy=True, foreign_keys='HealthLog.patient_id')
    doctor_notes = db.relationship('DoctorNote', backref='doctor', lazy=True, foreign_keys='DoctorNote.doctor_id')
    sent_messages = db.relationship('Message', foreign_keys='Message.sender_id', backref='sender', lazy=True)
    received_messages = db.relationship('Message', foreign_keys='Message.receiver_id', backref='receiver', lazy=True)
    notifications = db.relationship('Notification', backref='user', lazy=True)

# ------------------------------
# AIR QUALITY DATA
# ------------------------------
class AQIData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(150), nullable=False)  # e.g., "12.9716,77.5946"
    aqi_value = db.Column(db.Integer, nullable=False)
    main_pollutant = db.Column(db.String(50), nullable=True)
    date_time = db.Column(db.DateTime, default=datetime.utcnow)

# ------------------------------
# PATIENT HEALTH LOGS
# ------------------------------
class HealthLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symptoms = db.Column(db.Text, nullable=False)
    severity = db.Column(db.String(50), nullable=False)  # e.g., 'mild', 'moderate', 'severe'
    notes = db.Column(db.Text, nullable=True)
    lat = db.Column(db.Float, nullable=True)
    lon = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ------------------------------
# DOCTOR NOTES
# ------------------------------
class DoctorNote(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    recommendation = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ------------------------------
# AI PREDICTION RESULTS (Future Use)
# ------------------------------
class AIPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction_text = db.Column(db.Text, nullable=False)
    risk_level = db.Column(db.String(50), nullable=False)  # e.g., 'low', 'medium', 'high'
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ------------------------------
# CHAT MESSAGES (Doctor <-> Patient)
# ------------------------------
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ------------------------------
# NOTIFICATIONS (AQI Alerts, Health Reminders, etc.)
# ------------------------------
class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_read = db.Column(db.Boolean, default=False)
