# BroCode — Keystroke Behavioral Authentication (Prototype)

Status
- Prototype with Flask UI, keystroke capture, simple sklearn training, unified model I/O, and tests.
- Feature extraction: basic IKI + hold-time features.
- Not production-ready: security, privacy, storage, and extensive tests still required.

Quickstart
1. Install dependencies:
   python -m pip install -r ML/requirements.txt

2. Set secret (PowerShell):
   $env:BROCODE_SECRET="your_secret_here"

3. Run the app:
   python ML/app.py

4. Run tests:
   pytest ML/tests -q

Development notes
- Models persisted with `ML/model_io.py` (joblib payload + metadata).
- Run migration helper to convert ad-hoc joblib usage:
   python ML/scripts/migrate_to_model_io.py

- Improve keystroke features in `ML/keystroke_model.py::_extract_features_from_events`.
- Add/integrate more unit tests under `ML/tests/`.

Contact
- This repository is a local prototype. Review and harden before sharing or production use.