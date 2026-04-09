# auth_routes.py
from flask import Blueprint, request, render_template, redirect, url_for, flash, current_app, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from extensions import db
from models import User
from flask_login import login_user, logout_user, login_required, current_user

# Import the prediction helper (this module lazy-loads artifacts if not provided)
from utils.predict_aqi_tolerance import predict_aqi_tolerance

auth_bp = Blueprint('auth', __name__)

# Helper to lazily get a Bcrypt instance bound to the current app context.
# This avoids circular imports and will use the app's config.
def get_bcrypt():
    """
    Return a Flask-Bcrypt instance initialized with the current_app.
    We create it lazily so this module does not import 'app' at import-time.
    """
    try:
        from flask_bcrypt import Bcrypt
    except Exception:
        return None

    # store on current_app for reuse
    if not hasattr(current_app, "_bcrypt_instance"):
        current_app._bcrypt_instance = Bcrypt(current_app)
    return current_app._bcrypt_instance


# -------------------------
# Register Route
# -------------------------
@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = (request.form.get('username') or "").strip()
        email = (request.form.get('email') or "").strip().lower()
        password = request.form.get('password') or ""
        role = request.form.get('role')  # patient or doctor

        if not (username and email and password and role):
            flash("Please fill all fields.", "danger")
            return redirect(url_for('auth.register'))

        # check if user already exists
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash("Username or Email already exists!", "danger")
            return redirect(url_for('auth.register'))

        # hash password: prefer app's Bcrypt if available, else use werkzeug
        bcrypt = get_bcrypt()
        if bcrypt:
            hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        else:
            hashed_pw = generate_password_hash(password)

        # create new user
        new_user = User(username=username, email=email, password=hashed_pw, role=role)
        try:
            db.session.add(new_user)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            current_app.logger.exception("Failed to create user")
            flash("Registration failed. Try again.", "danger")
            return redirect(url_for('auth.register'))

        flash("Registration successful! Please login.", "success")
        return redirect(url_for('auth.login'))

    return render_template("register.html")


# -------------------------
# Login Route
# -------------------------
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # allow login via username or email if you want; here we use email
        email = (request.form.get('email') or "").strip().lower()
        password = request.form.get('password') or ""

        user = User.query.filter_by(email=email).first()
        if not user:
            flash("Invalid email or password", "danger")
            return redirect(url_for('auth.login'))

        # Try checking password with app Bcrypt first (if available), otherwise werkzeug
        bcrypt = get_bcrypt()
        password_ok = False
        try:
            if bcrypt:
                password_ok = bcrypt.check_password_hash(user.password, password)
            else:
                password_ok = check_password_hash(user.password, password)
        except Exception:
            # Fallback: try werkzeug check (helps if hashes were created differently)
            try:
                password_ok = check_password_hash(user.password, password)
            except Exception:
                password_ok = False

        if user and password_ok:
            # Use Flask-Login to log user in
            login_user(user)
            flash("Login successful!", "success")
            # single redirect point — app.py handles role-based dashboard routing
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid email or password", "danger")
            return redirect(url_for('auth.login'))

    return render_template("login.html")


# -------------------------
# Logout Route
# -------------------------
@auth_bp.route('/logout')
def logout():
    # Use Flask-Login logout
    try:
        logout_user()
    except Exception:
        # fallback: just clear session keys if any problem
        current_app.logger.exception("logout_user() raised an exception")
    flash("You have been logged out.", "info")
    return redirect(url_for('auth.login'))


# -------------------------
# Backwards-compatible API alias for ML prediction
# This allows clients that post to /api/predict to keep working.
# -------------------------
@auth_bp.route('/api/predict', methods=['POST'])
@login_required
def api_predict_alias():
    """
    Alias endpoint that forwards form/json payload to the existing predict helper.
    This endpoint requires authentication and only allows users with role 'patient'.
    """
    if current_user.role != 'patient':
        return jsonify({"success": False, "error": "Only patients can request predictions"}), 403

    # Accept JSON or form data
    payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()

    # Map/normalize expected fields (adjust keys here if your frontend uses different names)
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

    try:
        # Call the helper; it will lazy-load the model & preprocessing columns if needed.
        pred = predict_aqi_tolerance(
            disease=disease,
            cough_severity=cough,
            wheezing_severity=wheezing,
            breathlessness_severity=breathlessness,
            chest_pain_severity=chest_pain,
            throat_irritation_severity=throat,
            fatigue_severity=fatigue,
            aqi=aqi,
        )
    except Exception as e:
        current_app.logger.exception("Prediction failed in /api/predict")
        return jsonify({"success": False, "error": str(e)}), 500

    try:
        pred_int = int(pred)
    except Exception:
        pred_int = pred

    return jsonify({"success": True, "prediction": pred_int})
