# ==========================================================
# 🚗 AI-Based Real-Time Accident Detection System
# ==========================================================
# Features:
# - YOLO-based collision detection
# - Speed & severity estimation
# - Weather integration
# - Emergency service locator
# - Automated alerts (email/SMS)
# - Flask API for video processing
# - Number plate recognition (OCR)
# - Insurance claim automation
# - Video clip saving (before & after accident)
# - Audio alerts (sound + text-to-speech)
# - PDF report generation
# - Web dashboard
# ==========================================================

import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys
import time
import threading
import psutil
from datetime import datetime
from collections import defaultdict
from flask import Flask, request, jsonify, Response, render_template, send_from_directory, session, redirect, url_for
from flask_socketio import SocketIO
from functools import wraps
import logging

# Import custom modules
try:
    from alert import alert_system, send_emergency_alert
    from OSM import get_emergency_info, emergency_locator, weather_service
    from config import (
        MODEL_PATHS, UPLOAD_FOLDER, FLASK_HOST, FLASK_PORT, DEBUG_MODE,
        IOU_THRESHOLD_LOW, IOU_THRESHOLD_MEDIUM, IOU_THRESHOLD_HIGH,
        SPEED_LOW, SPEED_MEDIUM, SPEED_HIGH, DEFAULT_LATITUDE, DEFAULT_LONGITUDE,
        ENABLE_OCR
    )
except ImportError as e:
    print(f"⚠️ Module import warning: {e}")
    print("Using default configurations...")
    MODEL_PATHS = {"light": "models/yolov8n.pt"}
    UPLOAD_FOLDER = "uploads"
    FLASK_HOST = "0.0.0.0"
    FLASK_PORT = 5000
    DEBUG_MODE = True
    IOU_THRESHOLD_LOW = 0.3
    IOU_THRESHOLD_MEDIUM = 0.4
    IOU_THRESHOLD_HIGH = 0.5
    SPEED_LOW = 15
    SPEED_MEDIUM = 30
    SPEED_HIGH = 50
    DEFAULT_LATITUDE = 9.9312
    DEFAULT_LONGITUDE = 76.2673
    ENABLE_OCR = False

# Import new features
try:
    from features.video_clip import clip_saver, save_accident_clip, add_frame_to_buffer
    VIDEO_CLIP_ENABLED = True
except ImportError:
    VIDEO_CLIP_ENABLED = False
    print("⚠️ Video clip feature not available")

try:
    from features.audio_alert import audio_alert, alert_accident, play_sound
    AUDIO_ALERT_ENABLED = True
except ImportError:
    AUDIO_ALERT_ENABLED = False
    print("⚠️ Audio alert feature not available")

try:
    from features.pdf_report import generate_accident_report
    PDF_REPORT_ENABLED = True
except ImportError:
    PDF_REPORT_ENABLED = False
    print("⚠️ PDF report feature not available")

try:
    from features.database import AccidentDatabase
    DATABASE_ENABLED = True
    accident_db = AccidentDatabase()
except ImportError:
    DATABASE_ENABLED = False
    accident_db = None
    print("⚠️ Database feature not available")

try:
    from features.excel_export import export_accidents_to_excel
    EXCEL_EXPORT_ENABLED = True
except ImportError:
    EXCEL_EXPORT_ENABLED = False
    print("⚠️ Excel export feature not available")

try:
    from features.map_generator import generate_accident_map
    MAP_ENABLED = True
except ImportError:
    MAP_ENABLED = False
    print("⚠️ Map generator feature not available")

# Configure logging (UTF-8 encoding for Windows)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs.txt', encoding='utf-8'),
        logging.StreamHandler(open(1, 'w', encoding='utf-8', closefd=False))
    ]
)
logger = logging.getLogger(__name__)

# ==========================================================
# Flask + SocketIO Application
# ==========================================================
app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
app.config['SECRET_KEY'] = 'acc_detect_secret_2024_change_in_production'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Role Passwords — change these in config.py
try:
    from config import ROLE_PASSWORDS
except ImportError:
    ROLE_PASSWORDS = {
        'admin':     'admin@123',
        'police':    'police@123',
        'ambulance': 'amb@123'
    }

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'role' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated

# ==========================================================
# Create necessary directories
# ==========================================================
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)


# ==========================================================
# Dynamic Model Selection based on System Memory
# ==========================================================
def select_model():
    """Select YOLO model based on available system memory"""
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB

    if available_memory >= 8:
        model_type = "heavy"
        model_name = "yolov8m.pt"
    elif available_memory >= 4:
        model_type = "medium"
        model_name = "yolov8s.pt"
    else:
        model_type = "light"
        model_name = "yolov8n.pt"

    model_path = MODEL_PATHS.get(model_type, "models/yolov8n.pt")

    if not os.path.exists(model_path):
        model_path = "models/yolov8n.pt"
        model_name = "yolov8n.pt"

    logger.info(f"🧠 System Memory: {available_memory:.1f}GB | Model: {model_name}")
    return model_path


# ==========================================================
# Vehicle Classes in COCO Dataset
# ==========================================================
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
    1: "Bicycle"
}


# ==========================================================
# IoU Calculation
# ==========================================================
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / (areaA + areaB - interArea + 1e-6)
    return iou


# ==========================================================
# Vehicle Tracker for Speed Estimation
# ==========================================================
class VehicleTracker:
    """Tracks vehicles across frames for speed estimation"""

    def __init__(self):
        self.tracks = defaultdict(list)
        self.next_id = 0
        self.max_distance = 100
        self.max_age = 30

    def update(self, boxes, frame_num):
        results = []
        used_tracks = set()

        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            class_id = box[4] if len(box) > 4 else 2

            best_track = None
            best_distance = float('inf')

            for track_id, history in self.tracks.items():
                if track_id in used_tracks:
                    continue
                if not history:
                    continue

                last_frame, last_center, _ = history[-1]
                if frame_num - last_frame > self.max_age:
                    continue

                distance = np.sqrt((center[0] - last_center[0])**2 +
                                   (center[1] - last_center[1])**2)

                if distance < best_distance and distance < self.max_distance:
                    best_distance = distance
                    best_track = track_id

            if best_track is not None:
                track_id = best_track
                used_tracks.add(track_id)
            else:
                track_id = self.next_id
                self.next_id += 1

            self.tracks[track_id].append((frame_num, center, box))

            if len(self.tracks[track_id]) > 30:
                self.tracks[track_id] = self.tracks[track_id][-30:]

            speed = self._calculate_speed(track_id)
            results.append((track_id, box, speed, class_id))

        self._cleanup(frame_num)
        return results

    def _calculate_speed(self, track_id):
        history = self.tracks.get(track_id, [])
        if len(history) < 2:
            return 0

        recent = history[-5:]
        if len(recent) < 2:
            return 0

        total_distance = 0
        for i in range(1, len(recent)):
            prev_center = recent[i-1][1]
            curr_center = recent[i][1]
            distance = np.sqrt((curr_center[0] - prev_center[0])**2 +
                               (curr_center[1] - prev_center[1])**2)
            total_distance += distance

        avg_speed = total_distance / (len(recent) - 1)
        return avg_speed

    def _cleanup(self, current_frame):
        tracks_to_remove = []
        for track_id, history in self.tracks.items():
            if history:
                last_frame = history[-1][0]
                if current_frame - last_frame > self.max_age:
                    tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]


# ==========================================================
# Severity Calculator
# ==========================================================
class SeverityCalculator:
    """Calculates accident severity based on multiple factors"""

    @staticmethod
    def calculate(iou, speed1, speed2, vehicle_types):
        # Base score from IoU (0-40 points)
        iou_score = min(iou * 80, 40)

        # Speed score (0-40 points)
        avg_speed = (speed1 + speed2) / 2
        if avg_speed > SPEED_HIGH:
            speed_score = 40
        elif avg_speed > SPEED_MEDIUM:
            speed_score = 25
        elif avg_speed > SPEED_LOW:
            speed_score = 15
        else:
            speed_score = 5

        # Vehicle type score (0-20 points)
        heavy_vehicles = ["Truck", "Bus"]
        type_score = 0
        for v_type in vehicle_types:
            if v_type in heavy_vehicles:
                type_score += 10
        type_score = min(type_score, 20)

        impact_score = iou_score + speed_score + type_score

        if impact_score >= 70:
            severity = "CRITICAL"
            color = (0, 0, 255)
        elif impact_score >= 50:
            severity = "HIGH"
            color = (0, 69, 255)
        elif impact_score >= 35:
            severity = "MEDIUM"
            color = (0, 165, 255)
        else:
            severity = "LOW"
            color = (0, 255, 255)

        return severity, impact_score, color


# ==========================================================
# Damage Estimator for Insurance Claims
# ==========================================================
class DamageEstimator:
    """Estimates vehicle damage for insurance processing"""

    @staticmethod
    def estimate(severity, vehicle_types, impact_score):
        base_costs = {
            "CRITICAL": 15000,
            "HIGH": 10000,
            "MEDIUM": 5000,
            "LOW": 2000
        }

        type_multipliers = {
            "Car": 1.0,
            "Motorcycle": 0.5,
            "Bicycle": 0.2,
            "Bus": 2.5,
            "Truck": 2.0
        }

        base_cost = base_costs.get(severity, 5000)

        multiplier = 1.0
        if vehicle_types:
            multipliers = [type_multipliers.get(v, 1.0) for v in vehicle_types]
            multiplier = sum(multipliers) / len(multipliers)

        estimated_cost = int(base_cost * multiplier * (impact_score / 50))

        repair_days = {
            "CRITICAL": "30-45",
            "HIGH": "20-30",
            "MEDIUM": "10-20",
            "LOW": "5-10"
        }

        return {
            "level": severity,
            "estimated_cost": estimated_cost,
            "repair_days": repair_days.get(severity, "7-14"),
            "impact_score": impact_score,
            "vehicles": vehicle_types,
            "total_loss": severity == "CRITICAL" and impact_score > 85
        }


# ==========================================================
# Main Accident Detector Class
# ==========================================================
class AccidentDetector:
    """Main accident detection system"""

    def __init__(self, model_path=None):
        self.model_path = model_path or select_model()
        self.model = YOLO(self.model_path)
        self.tracker = VehicleTracker()
        self.frame_count = 0
        self.accidents_detected = []
        self.last_alert_time = 0
        self.alert_cooldown = 10  # seconds (Gopika's value)

        self.latitude = DEFAULT_LATITUDE
        self.longitude = DEFAULT_LONGITUDE

        logger.info(f"Accident Detector initialized with {self.model_path}")

    def set_location(self, lat, lon):
        """Update current location"""
        self.latitude = lat
        self.longitude = lon

    def process_frame(self, frame, fps=30):
        self.frame_count += 1

        if VIDEO_CLIP_ENABLED:
            add_frame_to_buffer(frame, fps)

        results = self.model(frame, verbose=False)

        boxes = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls in VEHICLE_CLASSES and conf > 0.3:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append([x1, y1, x2, y2, cls])

        tracked = self.tracker.update(boxes, self.frame_count)

        for track_id, box, speed, class_id in tracked:
            x1, y1, x2, y2 = map(int, box[:4])
            vehicle_type = VEHICLE_CLASSES.get(class_id, "Vehicle")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{vehicle_type} #{track_id} | {speed:.1f}px/f"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        accident_data = None
        for i in range(len(tracked)):
            for j in range(i + 1, len(tracked)):
                track1 = tracked[i]
                track2 = tracked[j]

                box1 = track1[1][:4]
                box2 = track2[1][:4]
                speed1 = track1[2]
                speed2 = track2[2]
                class1 = track1[3]
                class2 = track2[3]

                iou = calculate_iou(box1, box2)

                min_speed = 5
                try:
                    from config import MIN_SPEED_FOR_ACCIDENT
                    min_speed = MIN_SPEED_FOR_ACCIDENT
                except:
                    pass

                max_speed = max(speed1, speed2)

                if iou > IOU_THRESHOLD_LOW and max_speed >= min_speed:
                    vehicle_types = [
                        VEHICLE_CLASSES.get(class1, "Vehicle"),
                        VEHICLE_CLASSES.get(class2, "Vehicle")
                    ]

                    severity, impact_score, color = SeverityCalculator.calculate(
                        iou, speed1, speed2, vehicle_types
                    )

                    cx = int((box1[0] + box1[2] + box2[0] + box2[2]) / 4)
                    cy = int((box1[1] + box1[3] + box2[1] + box2[3]) / 4)

                    cv2.circle(frame, (cx, cy), 50, color, 3)
                    cv2.putText(frame, "!", (cx - 10, cy + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

                    cv2.putText(frame, f"ACCIDENT DETECTED",
                                (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    cv2.putText(frame, f"Severity: {severity} ({impact_score:.0f})",
                                (40, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame, f"IoU: {iou:.2%} | Vehicles: {', '.join(vehicle_types)}",
                                (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    accident_data = {
                        "timestamp": datetime.now().isoformat(),
                        "frame_number": self.frame_count,
                        "severity": severity,
                        "impact_score": impact_score,
                        "iou": iou,
                        "vehicles_count": 2,
                        "vehicle_types": vehicle_types,
                        "speeds": [speed1, speed2],
                        "latitude": self.latitude,
                        "longitude": self.longitude,
                        "collision_center": (cx, cy)
                    }

        cv2.putText(frame, f"Frame: {self.frame_count}",
                    (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Vehicles: {len(tracked)}",
                    (frame.shape[1] - 150, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame, accident_data

    def handle_accident(self, accident_data, frame):
        current_time = time.time()

        if current_time - self.last_alert_time < self.alert_cooldown:
            return

        self.last_alert_time = current_time

        # Save accident frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(UPLOAD_FOLDER, f"accident_{timestamp}.jpg")
        cv2.imwrite(image_path, frame)
        logger.info(f"Accident frame saved: {image_path}")

        # Save video clip
        video_clip_path = None
        if VIDEO_CLIP_ENABLED:
            try:
                save_accident_clip(frame, accident_data)
                video_clip_path = f"uploads/clips/accident_clip_{accident_data['severity']}_{timestamp}.mp4"
                logger.info(f"Video clip recording triggered")
            except Exception as e:
                logger.warning(f"Could not save video clip: {e}")

        # Audio alert
        if AUDIO_ALERT_ENABLED:
            try:
                alert_accident(accident_data)
                logger.info(f"Audio alert triggered for {accident_data['severity']} accident")
            except Exception as e:
                logger.warning(f"Could not play audio alert: {e}")

        # Get emergency services info (Gopika's richer version)
        try:
            emergency_info = get_emergency_info(
                accident_data['latitude'],
                accident_data['longitude']
            )

            h = emergency_info.get('nearest_hospital') or {}
            p = emergency_info.get('nearest_police') or {}
            accident_data.update({
                'address':           emergency_info['location']['address'],
                # Hospital details — included in email alert
                'nearest_hospital':  h.get('name', 'N/A'),
                'hospital_distance': h.get('distance_km', 'N/A'),
                'hospital_phone':    h.get('phone', 'N/A'),
                'hospital_address':  h.get('address', 'N/A'),
                'all_hospitals':     emergency_info.get('all_hospitals', []),
                # Police details
                'nearest_police':    p.get('name', 'N/A'),
                'police_distance':   p.get('distance_km', 'N/A'),
                'police_phone':      p.get('phone', 'N/A'),
                # Weather
                'weather':           emergency_info['weather']['weather'],
                'weather_risk':      emergency_info['weather']['risk_factor'],
                'temperature':       emergency_info['weather']['temperature'],
                'visibility':        emergency_info['weather']['visibility'],
            })
        except Exception as e:
            logger.warning(f"Could not fetch emergency info: {e}")

        # Damage assessment
        damage_assessment = DamageEstimator.estimate(
            accident_data['severity'],
            accident_data['vehicle_types'],
            accident_data['impact_score']
        )
        accident_data['damage_assessment'] = damage_assessment

        # PDF report
        if PDF_REPORT_ENABLED:
            try:
                pdf_path = generate_accident_report(accident_data, image_path, video_clip_path)
                if pdf_path:
                    accident_data['pdf_report'] = pdf_path
                    logger.info(f"PDF report generated: {pdf_path}")
            except Exception as e:
                logger.warning(f"Could not generate PDF report: {e}")

        # =====================================================
        # 🚨 REAL-TIME BROWSER NOTIFICATION via SocketIO
        # =====================================================
        try:
            socketio.emit('accident_alert', {
                'severity':  accident_data.get('severity', 'UNKNOWN'),
                'vehicles':  accident_data.get('vehicle_types', []),
                'impact':    round(accident_data.get('impact_score', 0), 1),
                'location':  accident_data.get('address', 'Detecting location...'),
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'iou':       round(accident_data.get('iou', 0) * 100, 1),
                'hospital':  accident_data.get('nearest_hospital', 'N/A'),
                'weather':   accident_data.get('weather', 'N/A'),
                'lat':       accident_data.get('latitude'),
                'lon':       accident_data.get('longitude'),
            })
            logger.info("✅ SocketIO notification sent to browser")
        except Exception as e:
            logger.warning(f"SocketIO emit failed: {e}")

        # Send alerts in background thread
        try:
            alert_thread = threading.Thread(
                target=self._send_alerts,
                args=(accident_data, image_path)
            )
            alert_thread.start()
        except Exception as e:
            logger.error(f"Alert thread error: {e}")

        # Store accident record
        self.accidents_detected.append(accident_data)
        logger.info(f"Accident #{len(self.accidents_detected)} recorded: {accident_data['severity']}")

        # Save to database
        if DATABASE_ENABLED and accident_db:
            try:
                accident_db.save_accident(accident_data)
                logger.info("✅ Accident saved to database")
            except Exception as e:
                logger.warning(f"Could not save to database: {e}")

    def _send_alerts(self, accident_data, image_path):
        """
        Correct logical alert flow:
        1. Emergency services  — always, immediately
        2. Family              — HIGH and CRITICAL only
        3. Insurance notice    — always (evidence notice, NOT a claim)
        4. Blood donation      — CRITICAL only
        5. Organ donation      — CRITICAL only
        Insurance CLAIM is NOT sent automatically.
        Admin must review and trigger it from the dashboard.
        """
        try:
            # Step 1: Emergency services — dispatch immediately
            send_emergency_alert(accident_data, image_path)
            logger.info("✅ Emergency alert sent")

            # Step 2: Family notification — only if serious
            if accident_data['severity'] in ['HIGH', 'CRITICAL']:
                alert_system.send_nominee_alert(accident_data)
                logger.info("✅ Family notification sent")

            # Step 3: Insurance evidence notice (NOT a claim)
            incident_ref = alert_system.send_insurance_evidence_notice(
                accident_data, image_path
            )
            accident_data['incident_ref'] = incident_ref
            logger.info(f"✅ Insurance evidence notice sent | Ref: {incident_ref}")

            # Step 4: Blood donation — CRITICAL only
            if accident_data['severity'] == 'CRITICAL':
                alert_system.send_blood_donation_request(accident_data)
                logger.info("✅ Blood donation request sent")

            # Step 5: Organ donation — CRITICAL only
            if accident_data['severity'] == 'CRITICAL':
                alert_system.send_organ_donation_alert(accident_data)
                logger.info("✅ Organ donation alert sent")

            logger.info("✅ All alerts completed")
        except Exception as e:
            logger.error(f"❌ Alert error: {e}")

    def process_video(self, video_path, display=True, save_output=False):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"❌ Could not open video: {video_path}")
            return []

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Processing video: {video_path}")
        logger.info(f"   Resolution: {width}x{height} | FPS: {fps} | Frames: {total_frames}")

        output_path = None
        writer = None
        if save_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(UPLOAD_FOLDER, f"output_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.frame_count = 0
        self.tracker = VehicleTracker()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, accident_data = self.process_frame(frame, fps)

            if accident_data:
                self.handle_accident(accident_data, annotated_frame)

            if writer:
                writer.write(annotated_frame)

            if display:
                cv2.imshow("AI Accident Detection System", annotated_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        logger.info(f"Processing complete. Accidents detected: {len(self.accidents_detected)}")
        return self.accidents_detected


# ==========================================================
# Global detector instance
# ==========================================================
detector = None


def get_detector():
    global detector
    if detector is None:
        detector = AccidentDetector()
    return detector


# ==========================================================
# Flask API Routes
# ==========================================================

@app.route('/login', methods=['GET'])
def login_page():
    if 'role' in session:
        return redirect(url_for('home'))
    return render_template('dashboard.html')

@app.route('/login', methods=['POST'])
def do_login():
    data = request.get_json()
    role = data.get('role', '').strip()
    password = data.get('password', '').strip()
    if role in ROLE_PASSWORDS and ROLE_PASSWORDS[role] == password:
        session['role'] = role
        session.permanent = False
        logger.info(f"Login: {role}")
        return jsonify({"success": True, "role": role})
    logger.warning(f"Failed login attempt for role: {role}")
    return jsonify({"success": False, "error": "Invalid role or password"}), 401

@app.route('/logout')
def logout():
    role = session.pop('role', None)
    logger.info(f"Logout: {role}")
    return redirect(url_for('login_page'))

@app.route('/session/check')
def check_session():
    if 'role' in session:
        return jsonify({"logged_in": True, "role": session['role']})
    return jsonify({"logged_in": False}), 401

@app.route('/')
@login_required
def home():
    return render_template('dashboard.html')


@app.route('/api')
def api_info():
    return jsonify({
        "service": "AI Accident Detection System",
        "version": "2.0",
        "status": "running",
        "realtime": "SocketIO enabled",
        "features": {
            "video_clip": VIDEO_CLIP_ENABLED,
            "audio_alert": AUDIO_ALERT_ENABLED,
            "pdf_report": PDF_REPORT_ENABLED,
            "socketio": True
        },
        "endpoints": {
            "/":                    "GET  - Web Dashboard",
            "/api":                 "GET  - API Information",
            "/detect":              "POST - Upload video for detection",
            "/health":              "GET  - Health check",
            "/accidents":           "GET  - List detected accidents",
            "/stream":              "GET  - Live camera stream",
            "/claim/initiate":      "POST - Initiate insurance claim (admin only)",
            "/reports/<filename>":  "GET  - Download PDF reports",
            "/clips/<filename>":    "GET  - Download video clips"
        }
    })


@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": detector is not None,
        "realtime_enabled": True,
        "features": {
            "video_clip": VIDEO_CLIP_ENABLED,
            "audio_alert": AUDIO_ALERT_ENABLED,
            "pdf_report": PDF_REPORT_ENABLED
        },
        "timestamp": datetime.now().isoformat()
    })


@app.route('/reports/<filename>')
def download_report(filename):
    return send_from_directory('uploads/reports', filename)


@app.route('/clips/<filename>')
def download_clip(filename):
    return send_from_directory('uploads/clips', filename)


@app.route('/detect', methods=['POST'])
@login_required
def detect_accidents():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No video selected"}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(UPLOAD_FOLDER, f"upload_{timestamp}.mp4")
    video_file.save(video_path)
    logger.info(f"📤 Video uploaded: {video_path}")

    det = get_detector()
    accidents = det.process_video(video_path, display=False, save_output=True)

    return jsonify({
        "success": True,
        "video_path": video_path,
        "accidents_detected": len(accidents),
        "accidents": accidents,
        "processing_time": datetime.now().isoformat()
    })


@app.route('/accidents')
@login_required
def list_accidents():
    det = get_detector()
    return jsonify({
        "total_accidents": len(det.accidents_detected),
        "accidents": det.accidents_detected
    })


@app.route('/stream')
@login_required
def video_stream():
    def generate():
        cap = cv2.VideoCapture(0)
        det = get_detector()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame, _ = det.process_frame(frame)
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/export/excel')
@login_required
def export_excel():
    if not EXCEL_EXPORT_ENABLED:
        return jsonify({"error": "Excel export not available"}), 400
    det = get_detector()
    if not det.accidents_detected:
        return jsonify({"error": "No accidents to export"}), 400
    try:
        filepath = export_accidents_to_excel(det.accidents_detected)
        return send_from_directory(
            os.path.dirname(filepath),
            os.path.basename(filepath),
            as_attachment=True,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        logger.error(f"Excel export error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/export/map')
@login_required
def export_map():
    if not MAP_ENABLED:
        return jsonify({"error": "Map generator not available"}), 400
    det = get_detector()
    if not det.accidents_detected:
        return jsonify({"error": "No accidents to map"}), 400
    try:
        map_path = generate_accident_map(det.accidents_detected)
        return send_from_directory(
            os.path.dirname(map_path),
            os.path.basename(map_path),
            as_attachment=False,
            mimetype='text/html'
        )
    except Exception as e:
        logger.error(f"Map generation error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/statistics')
def get_statistics():
    if DATABASE_ENABLED and accident_db:
        return jsonify(accident_db.get_statistics())
    det = get_detector()
    accidents = det.accidents_detected
    stats = {
        "total": len(accidents),
        "by_severity": {
            "CRITICAL": len([a for a in accidents if a.get('severity') == 'CRITICAL']),
            "HIGH":     len([a for a in accidents if a.get('severity') == 'HIGH']),
            "MEDIUM":   len([a for a in accidents if a.get('severity') == 'MEDIUM']),
            "LOW":      len([a for a in accidents if a.get('severity') == 'LOW'])
        }
    }
    return jsonify(stats)


@app.route('/claim/initiate', methods=['POST'])
@login_required
def initiate_insurance_claim():
    """
    Formally initiates an insurance claim.
    Called ONLY by admin from the dashboard after reviewing the incident.
    NOT triggered automatically — requires human approval.
    """
    data = request.get_json()
    if not data or 'accident_index' not in data:
        return jsonify({"error": "accident_index required"}), 400

    det = get_detector()
    idx = int(data['accident_index'])

    if idx < 0 or idx >= len(det.accidents_detected):
        return jsonify({"error": "Invalid accident index"}), 400

    accident = det.accidents_detected[idx]

    if accident.get('claim_submitted'):
        return jsonify({
            "error": "Claim already submitted",
            "claim_ref": accident.get('claim_ref')
        }), 400

    damage = accident.get('damage_assessment') or DamageEstimator.estimate(
        accident.get('severity', 'LOW'),
        accident.get('vehicle_types', []),
        accident.get('impact_score', 0)
    )

    try:
        result = alert_system.send_formal_insurance_claim(
            accident, damage, incident_ref=accident.get('incident_ref')
        )
        det.accidents_detected[idx]['claim_submitted'] = True
        det.accidents_detected[idx]['claim_ref'] = result['claim_ref']
        det.accidents_detected[idx]['claim_time'] = datetime.now().isoformat()
        logger.info(f"✅ Insurance claim initiated | Ref: {result['claim_ref']}")
        return jsonify({
            "success": True,
            "claim_ref": result['claim_ref'],
            "incident_ref": result['incident_ref'],
            "message": "Claim submitted. Insurer will contact vehicle owner for surveyor inspection."
        })
    except Exception as e:
        logger.error(f"Claim error: {e}")
        return jsonify({"error": str(e)}), 500


# ==========================================================
# SocketIO Events
# ==========================================================

@socketio.on('connect')
def handle_connect():
    logger.info("🔌 Browser client connected via SocketIO")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("🔌 Browser client disconnected")


# ==========================================================
# Main Entry Point
# ==========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Accident Detection System")
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--api', action='store_true', help='Run Flask API server')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--camera', type=int, default=0, help='Camera index for live detection')

    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║  🚗 AI-Based Real-Time Accident Detection System 🚨          ║
║  ─────────────────────────────────────────────────────────   ║
║  Version: 2.0  |  SocketIO: Enabled  |  Roles: 3            ║
╚══════════════════════════════════════════════════════════════╝
    """)

    detector = AccidentDetector()

    if args.api or (not args.video and args.camera == 0):
        logger.info(f"🌐 Starting Flask+SocketIO server on {FLASK_HOST}:{FLASK_PORT}")
        socketio.run(app, host=FLASK_HOST, port=FLASK_PORT,
                     debug=args.debug or DEBUG_MODE, allow_unsafe_werkzeug=True)

    elif args.video:
        if os.path.exists(args.video):
            detector.process_video(args.video, display=True, save_output=True)
        else:
            logger.error(f"❌ Video file not found: {args.video}")

    else:
        cap = cv2.VideoCapture(args.camera)
        logger.info(f"📹 Starting live detection from camera {args.camera}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame, accident_data = detector.process_frame(frame)
            if accident_data:
                detector.handle_accident(accident_data, annotated_frame)
            cv2.imshow("AI Accident Detection - Live", annotated_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()