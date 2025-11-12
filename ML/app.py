from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import json
import os
import logging
from flask_wtf import CSRFProtect
from flask import current_app

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("brocode_app")

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
from flask_wtf.csrf import generate_csrf
@app.context_processor
def inject_csrf():
    return dict(csrf_token=generate_csrf)

# Create bilingual system instance (uses KeystrokeAuthenticationModel/BilingualAuthenticationSystem)
from keystroke_model import BilingualAuthenticationSystem
MODEL_DIR = os.environ.get("BROCODE_MODEL_DIR", "models")
bilingual_system = BilingualAuthenticationSystem(model_dir=MODEL_DIR)

@app.route("/")
def index():
    return render_template("index.html")

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

# serve models or data for debug (optional)
@app.route("/models/<path:filename>")
def models(filename):
    return send_from_directory(MODEL_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)