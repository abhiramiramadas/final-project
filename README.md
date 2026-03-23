# 🚗💥 AiSeeYou — AI-Based Real-Time Accident Detection System

An AI-powered road accident detection and emergency response system that detects vehicle collisions in real-time, estimates severity, and instantly notifies emergency services, family members, and a role-based web dashboard.

---

## 👥 Team

| Name | Role |
|------|------|
| Gopika | Core Detection Engine (YOLO, collision logic, severity estimation, email alerts) |
| Abhirami | Notification System (real-time dashboard, SocketIO, role-based login, insurance flow) |

**Supervised by:** Dr. Bharathi Mohan G — Amrita School of Computing, Chennai

---

## ✨ Features

| Category | Feature |
|----------|---------|
| **Detection** | YOLOv8 vehicle detection + frame-by-frame tracking |
| **Collision Detection** | IoU overlap + speed + temporal consistency |
| **Severity Grading** | CRITICAL / HIGH / MEDIUM / LOW with impact scoring |
| **Real-Time Alerts** | Browser push notifications via SocketIO |
| **Role-Based Dashboard** | Admin / Police / Ambulance views with login |
| **Email Alerts** | Emergency services, family, insurance evidence notice |
| **Emergency Services** | Nearest hospital & police via OpenStreetMap |
| **Weather Data** | Live weather conditions at accident location |
| **PDF Reports** | Auto-generated accident reports |
| **Database** | SQLite storage for all accident records |
| **Excel Export** | Export accident data to styled Excel files |
| **Interactive Maps** | Accident location maps with markers |
| **Insurance Flow** | Evidence notice (auto) + formal claim (admin-triggered) |
| **Video Clips** | Auto-save video clips of accident events |
| **Desktop GUI** | Tkinter-based GUI with live video & controls |

---

## 🏗️ Project Structure

```
aiseeyou/
├── detection.py          # Core: YOLO detection, SocketIO, Flask API, login
├── alert.py              # Email alert system (emergency, family, insurance)
├── OSM.PY                # Emergency services locator + weather
├── haversine_gui.py      # Tkinter desktop GUI
├── main.py               # CLI entry point
├── config.example.py     # Template config — copy to config.py and fill in keys
├── Requirements.txt      # Python dependencies
│
├── templates/
│   └── dashboard.html    # Role-based web dashboard (Admin/Police/Ambulance)
│
├── features/
│   ├── video_clip.py     # Save accident video clips
│   ├── audio_alert.py    # Audio notifications
│   ├── pdf_report.py     # PDF report generator
│   ├── database.py       # SQLite accident storage
│   ├── excel_export.py   # Excel export
│   └── map_generator.py  # Interactive map generation
│
└── uploads/              # Saved accident frames, clips, reports (auto-created)
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/abhiramiramadas/final-project.git
cd final-project

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r Requirements.txt
pip install flask-socketio
```

### 2. Configure

```bash
copy config.example.py config.py   # Windows
# cp config.example.py config.py   # Linux/Mac
```

Edit `config.py` and fill in:
- Gmail address + App Password for email alerts
- OpenWeatherMap API key (free at openweathermap.org)
- Role passwords for dashboard login

### 3. Run

```bash
# Start web dashboard (recommended)
python detection.py --api

# Process a specific video file
python detection.py --video path/to/video.mp4

# Live webcam detection
python detection.py --camera 0

# Desktop GUI
python main.py --gui
```

Open **http://127.0.0.1:5000** in your browser.

---

## 🔐 Role-Based Dashboard

The web dashboard has three roles, each with a separate login and view:

| Role | Default Password | View |
|------|-----------------|------|
| Admin | `admin@123` | Full dashboard — all accidents, charts, exports, insurance claim |
| Police | `police@123` | Incident list with vehicle details, number plate, location |
| Ambulance | `amb@123` | Dispatch view — severity, nearest hospital, navigate button |

> Change passwords in `config.py` under `ROLE_PASSWORDS`.

**For demo on multiple systems:** All teammates connect to the same server over WiFi:
```
Admin     → http://localhost:5000
Police    → http://<your-ip>:5000
Ambulance → http://<your-ip>:5000
```

---

## 🧠 How Detection Works

```
Frame → YOLO Detection → Vehicle Tracking → Speed Estimation
                                                   ↓
                      Alert ← Severity Calc ← Collision Check (IoU + Speed)
                        ↓
              SocketIO → Browser (instant)
              Email    → Emergency services, Family, Insurance
```

1. **YOLOv8** detects vehicles (car, truck, bus, motorcycle, bicycle)
2. **Tracker** assigns IDs and follows vehicles across frames
3. **Collision detector** checks IoU overlap + minimum speed threshold
4. **Severity calculator** grades: CRITICAL / HIGH / MEDIUM / LOW
5. **SocketIO** pushes instant browser notification to all connected dashboards
6. **Email system** sends alerts in correct logical order (emergency → family → insurance notice)

---

## 📧 Alert Flow (Logical Order)

| Step | Alert | Triggered When |
|------|-------|---------------|
| 1 | Emergency services email | Every accident |
| 2 | Family / nominee email | HIGH and CRITICAL only |
| 3 | Insurance evidence notice | Every accident (not a claim) |
| 4 | Blood donation request | CRITICAL only |
| 5 | Organ donation alert | CRITICAL only |
| 6 | Formal insurance claim | Admin manually triggers from dashboard |

> Insurance claim is **not automatic** — admin must review and approve from dashboard first.

---

## 🌐 API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/` | ✅ | Web dashboard |
| GET | `/login` | — | Login page |
| POST | `/login` | — | Authenticate |
| GET | `/logout` | — | Logout |
| GET | `/health` | — | Health check |
| GET | `/accidents` | ✅ | List all accidents |
| POST | `/detect` | ✅ | Upload video for analysis |
| GET | `/stream` | ✅ | Live webcam stream |
| POST | `/claim/initiate` | ✅ | Initiate insurance claim (admin) |
| GET | `/export/excel` | ✅ | Export to Excel |
| GET | `/statistics` | — | Accident statistics |

---

## ⚙️ Key Config Settings

```python
# Detection
IOU_THRESHOLD_LOW = 0.30        # Minimum IoU to flag collision
MIN_SPEED_FOR_ACCIDENT = 5      # Ignore parked cars (px/frame)
ALERT_COOLDOWN = 10             # Seconds between alerts

# Location (Kerala default)
DEFAULT_LATITUDE = 9.9312
DEFAULT_LONGITUDE = 76.2673

# Dashboard roles
ROLE_PASSWORDS = {
    'admin':     'admin@123',
    'police':    'police@123',
    'ambulance': 'amb@123'
}
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **YOLOv8** (Ultralytics) — object detection
- **OpenCV** — video processing
- **Flask + Flask-SocketIO** — REST API, real-time notifications
- **NumPy** — numerical operations
- **SQLite** — data persistence
- **ReportLab** — PDF generation
- **Folium** — interactive maps
- **OpenStreetMap** — emergency services lookup
- **OpenWeatherMap** — weather data

---

## 📄 License

Built for educational purposes as a final year project — Amrita Vishwa Vidyapeetham, Chennai.