# 🧠 BroCode – Adaptive Behavioral Authentication System

<p align="center">
  <b>Hackathon Problem Statement 3: Sentinel – The Continuous Trust Engine</b><br>
  <i>AI-driven, continuous behavioral authentication based on human patterns of interaction.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Flask-2.3-blue?logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/OpenCV-4.8-green?logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/AI-Behavioral%20Trust-yellow" alt="AI Behavioral Trust">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-success" alt="Status">
</p>

---

## 🧩 Overview

**BroCode** is an **AI-driven Adaptive Behavioral Authentication System** designed to continuously verify user identity by analyzing behavioral patterns such as **keystrokes, gaze, posture, and facial emotions** — forming a unified, real-time **trust score**.

Unlike traditional one-time login systems, BroCode provides **continuous, passive authentication** powered by multi-modal machine learning and explainable AI.

> 🏆 Built for Hackathon Problem Statement 3 (**Sentinel – The Continuous Trust Engine**)

---

## 🚀 Features

✅ Continuous Authentication – Real-time monitoring of user behavior  
✅ Multi-Modal Analysis – Combines gaze, pose, emotion, and keystroke data  
✅ Adaptive Trust Engine – Dynamic trust scoring via weighted fusion  
✅ Behavioral Drift Detection – Detects deviation or impersonation  
✅ Explainability – Highlights key anomalies and reasons  
✅ Live Trust Dashboard – Real-time graph with Chart.js  
✅ Modular Architecture – Fully scalable, pluggable ML modules

---

## 🧠 System Architecture

```
┌───────────────────────────────┐
│ FRONTEND                     │
│  - monitor.js                │
│  - dashboard.html            │
│ Sends → /monitor_activity    │
└──────────────┬────────────────┘
               │
┌──────────────▼────────────────┐
│ BACKEND (Flask)               │
│  - app.py                     │
│  - main_authentication_system │
│  - continuous_monitor.py      │
│  - anomaly_detector.py        │
│  - fusion_engine.py           │
└──────────────┬────────────────┘
               │
┌──────────────▼────────────────┐
│ AI / ML Modules               │
│  - Gaze Analysis              │
│  - Pose Estimation            │
│  - Emotion Classification     │
│  - Keystroke Dynamics         │
│ Fusion → Trust + Risk Score   │
└───────────────────────────────┘
```

---

## 🧰 Tech Stack

| Category | Tools |
|-----------|--------|
| Backend | Flask, Python 3.10+, OpenCV |
| Frontend | HTML, CSS, JS, Chart.js |
| ML/AI | scikit-learn, numpy, joblib |
| Explainability | Custom feature drift explainability |
| Security | Encrypted model storage |

---

## ⚙️ Setup

```bash
git clone https://github.com/Amitgupta0001/BroCode.git
cd BroCode/ML
pip install -r requirements.txt
python app.py
```

Open → http://127.0.0.1:5000/dashboard

---

## 📈 Dashboard Preview

```
Trust ───────────────────────────────
1.0 |      ╭──╮
0.7 |   ╭──╯  ╰──╮
0.5 | ╭─╯       ╰─╮ ⚠️
0.3 |╭╯           ╰─────────
    └────────────────────────
     Time → Continuous Session
```

---

## 🧠 Explainable AI Output

```
Key deviations → attention_score: 0.21, emotion_variability: 0.89, movement_smoothness: 0.31
Detected anomaly type: Emotional instability (confidence: 0.82)
```

---

## 🔒 Security

- Behavioral templates encrypted via `model_io.py`  
- Real-time anomaly & drift detection  
- Reauthentication triggered when trust < threshold

---

## 🧠 Future Enhancements

- Real webcam gaze + pose via WebRTC  
- Federated learning personalization  
- Blockchain-based trust logs  
- Enterprise IAM integration

---

## 👨‍💻 Author

**Amit Kumar Gupta** – Project Lead, AI & Backend Developer  
GitHub: [@Amitgupta0001](https://github.com/Amitgupta0001)

---

## 📜 License

MIT License – Free to use for research and innovation.

---

### 🚀 “BroCode – Where Trust Never Sleeps.”
