# ML/app.py
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
import json
import os
import logging
from flask_wtf import CSRFProtect
from flask_wtf.csrf import generate_csrf

# Local imports (use the modules present in your repo)
# - BilingualAuthenticationSystem handles keystroke register/authenticate
# - BehavioralAuthSystem orchestrates multi-modal monitoring
from keystroke_model import BilingualAuthenticationSystem
from main_authentication_system import BehavioralAuthSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("brocode_app")

# Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Secret management
secret = os.environ.get("BROCODE_SECRET")
if not secret:
    logger.warning("BROCODE_SECRET not set; using dev fallback. Set BROCODE_SECRET env var for production.")
    secret = "brocode_keystroke_secret_dev"
app.config['SECRET_KEY'] = secret

# CSRF protection
csrf = CSRFProtect(app)

# expose a template function to include CSRF token easily
@app.context_processor
def inject_csrf():
    return dict(csrf_token=generate_csrf)

# Models / systems
MODEL_DIR = os.environ.get("BROCODE_MODEL_DIR", "models")
bilingual_system = BilingualAuthenticationSystem(model_dir=MODEL_DIR)

# Behavioral orchestrator for continuous monitoring
behavioral_system = BehavioralAuthSystem()

# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")

# Register route (keeps your original flow)
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")
    # POST
    username = request.form.get("username", "").strip()
    language = request.form.get("language", "english")
    keystrokes_raw = request.form.get("keystrokes", "")
    passphrase = request.form.get("passphrase", "")
    if not username or len(passphrase) < 3:
        flash("Invalid username or passphrase", "error")
        return redirect(url_for("register"))

    try:
        keystrokes = json.loads(keystrokes_raw) if keystrokes_raw else []
    except Exception:
        keystrokes = []

    # For registration we accept multiple repetitions; here we store single sample as list-of-events
    samples = [keystrokes]
    labels = [username]

    ok = bilingual_system.register_user(username, language, samples, labels)
    if ok:
        flash("Registration succeeded", "success")
    else:
        flash("Registration failed", "error")
    return redirect(url_for("index"))

# Authenticate route (keystroke-based login)
@app.route("/authenticate", methods=["GET", "POST"])
def authenticate():
    if request.method == "GET":
        return render_template("authenticate.html")
    username = request.form.get("username", "").strip()
    language = request.form.get("language", "english")
    keystrokes_raw = request.form.get("keystrokes", "")
    passphrase = request.form.get("passphrase", "")

    try:
        keystrokes = json.loads(keystrokes_raw) if keystrokes_raw else []
    except Exception:
        keystrokes = []

    accepted, score = bilingual_system.authenticate_user(username, language, keystrokes)
    if accepted:
        flash(f"Authenticated (score={score:.2f})", "success")
    else:
        flash(f"Authentication failed (score={score:.2f})", "error")
    return redirect(url_for("index"))

# Dashboard route (visualization)
@app.route("/dashboard")
def dashboard():
    # You can pass session/user info here if needed
    return render_template("dashboard.html")

# Serve models or data for debug (optional)
@app.route("/models/<path:filename>")
def models(filename):
    return send_from_directory(MODEL_DIR, filename)

# --- New: monitor_activity endpoint for continuous monitoring ---
@app.route("/monitor_activity", methods=["POST"])
def monitor_activity():
    """
    Receives continuous data (keystrokes, frame info, device info) from frontend,
    and returns trust score + risk label.
    Expected payload:
    {
      "user_id": "username",
      "keystrokes": [ { "key": "a", "t": 1234567, "type": "keydown" }, ... ],
      "frame_data": { ... }                 # optional, can be empty or basic metrics
      "device_info": { "ua": "...", ... }   # optional
    }
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid json"}), 400

    user_id = data.get("user_id", "guest")
    keystrokes = data.get("keystrokes", [])
    frame_data = data.get("frame_data", {})  # optional; your gaze/pose modules expect frames, but browser can't send images easily
    device_info = data.get("device_info", {})

    # Prepare a simple behavioral_data dict to feed the monitor.
    # If you have real gaze/pose data from backend/video processing, include their scores here.
    # For browser-side only keystroke monitoring, we will send the keystroke-derived score via your bilingual_system
    # and then combine with any provided frame_data metrics (if present).
    # 1) Keystroke score (0..1)
    ks_score = 0.0
    try:
        # Attempt to authenticate using bilingual_system to get score
        # Note: bilingual_system.authenticate_user returns (accepted, score)
        accepted, ks_score = bilingual_system.authenticate_user(user_id, device_info.get("language", "english"), keystrokes)
        # ks_score is used as a proxy trust contribution
    except Exception:
        ks_score = 0.0

    # Build behavioral_data input for the main behavioral_system.monitor_session
    # main_authentication_system.monitor_session expects (user_id, frame, keystrokes)
    # But frame may be None or a placeholder dict — we will pass frame_data dict and keystrokes list
    try:
        trust_score, anomaly_flag = behavioral_system.monitor_session(user_id, frame_data, keystrokes)
    except Exception as e:
        # Fallback: compute simple combined trust using ks_score and any numeric 'frame' trust passed
        fallback_frame_trust = float(frame_data.get("frame_trust", 0.5)) if isinstance(frame_data, dict) else 0.0
        # simple weighted fusion; keep consistent with fusion_engine weights if available
        trust_score = round(0.6 * ks_score + 0.4 * fallback_frame_trust, 3)
        anomaly_flag = False

    # Respond with JSON
    return jsonify({
        "user_id": user_id,
        "trust_score": trust_score,
        "risk": "high" if (anomaly_flag is True or trust_score < 0.5) else "normal",
        "anomaly": bool(anomaly_flag)
    })

# Run
if __name__ == "__main__":
    # If you want to pre-start any sessions or models, do that here.
    logger.info("Starting BroCode ML app (development mode)")
    app.run(debug=True)
