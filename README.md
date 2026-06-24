# 😴 NapNot - Advanced AI fatigue Detection System

> **High-accuracy multi-person fatigue and driver attention tracking using MediaPipe Face Mesh, 3D Pose Estimation, and a multi-stage alert engine.**

NapNot is a professional driver monitoring system (DMS) that tracks multiple faces simultaneously. It extracts 3D eye, mouth, and head pose metrics to calculate an intelligent 0-100 Fatigue Index, triggering staged alarms when fatigue is detected.

---

## ✨ Features & Architecture

| Module | Core Features | Why it Improves Accuracy |
|--------|---------------|--------------------------|
| **`camera.py`** | Multi-threaded capture + Adaptive Low-light Enhancer (CLAHE/Gamma) | Removes camera I/O lag and improves landmark extraction in dark vehicle cabins. |
| **`face_detector.py`** | MediaPipe Face Mesh wrapper (468/478 refined 3D landmarks) | Lightweight CPU processing; provides true depth coordinates without heavy dependencies. |
| **`tracker.py`** | Centroid-based face tracker | Persistently maps driver profiles across frames (IDs remain stable as drivers move). |
| **`eye.py`** | Dynamic EAR Calibration + Blink Analyzer (duration & BPM) | Adjusts EAR threshold per user (`avg_ear * 0.75`). Classifies blinks vs. micro-sleeps. |
| **`mouth.py`** | Mouth Aspect Ratio (MAR) + Yawn tracker | Logs yawns when MAR is high for >2 seconds (prevents false triggers from talking). |
| **`head_pose.py`** | 3D SolvePnP Head Pose Estimation (Pitch/Yaw/Roll) | Detects nodding off (pitch down) and distraction (looking left/right/away). |
| **`perclos.py`** | Percentage of Eye Closure (PERCLOS) | Computes the proportion of eye closure time over a sliding 30/60 second window. |
| **`fatigue.py`** | Multi-feature Fatigue Score (0-100) + Swappable AI Classifier | Integrates EAR, MAR, PERCLOS, blinks, and head pose. Swappable with ML models. |
| **`alarm.py`** | Threaded multi-stage alarms & SAPI Voice Synthesizer | Non-blocking warning stages: Beep (2s) -> Siren (4s) -> Voice speech warnings (6s). |
| **`logger.py`** | SQLite, CSV, and Markdown Summary Reporting | Logs telemetry in real-time. Automatically writes a markdown summary report on exit. |
| **`ui.py`** | Dark Glassmorphic Dashboard HUD | Extends webcam view with sidebar showing individual driver cards, sparklines, and meters. |

---

## 📁 Project Structure

```
NapDetector/
├── napnot.py          # Main coordinator (app entry point)
├── config.py          # Centralized configuration (thresholds, weights, styling)
├── camera.py          # Threaded camera stream & Low-Light Enhancement
├── face_detector.py   # MediaPipe Face Mesh landmark extractor
├── tracker.py         # Centroid tracker & Person State Manager
├── eye.py             # EAR, dynamic calibration, blink classification
├── mouth.py           # MAR calculation & Yawn detection
├── head_pose.py       # Perspective-n-Point 3D head pose estimator
├── perclos.py         # PERCLOS sliding window buffer
├── fatigue.py         # Weighted fatigue score & classifier
├── alarm.py           # Audio alarm playback & PowerShell speech synthesis
├── logger.py          # CSV, SQLite, and Markdown session summary writer
├── ui.py              # Dashboard sidebar panel, graphs, & 3D coordinate axes
├── requirements.txt   # Project dependencies (no dlib required)
└── README.md          # This documentation
```

---

## 🧠 Fatigue Scoring Matrix (0-100)

Fatigue is classified into 5 levels based on an Exponential Moving Average (EMA) of a weighted combination:
- **0 - 20: Alert**
- **20 - 40: Normal**
- **40 - 60: Fatigue Starting**
- **60 - 80: Drowsy**
- **80 - 100: Sleeping**

---

## 🚀 Quick Start

### 1. Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Start NapNot
Start the application from your interactive shell:
```bash
python napnot.py
```

---

## 🎮 Interface Controls
- **`q`** : Safe shutdown. Saves CSV, SQLite telemetry, and compiles a `session_report.md` file.
- **`r`** : Reset blink counters, active alarms, and fatigue scores.
- **`+` / `-`** : Manually increase/decrease the base EAR threshold sensitivity live.
