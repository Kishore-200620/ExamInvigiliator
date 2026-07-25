"""
config/settings.py

Centralized configuration settings for ExamGuardian AI.
"""
import torch
from config.paths import (
    DETECTION_MODEL_DIR,
    POSE_MODEL_DIR,
    DATABASE_DIR,
    LOGS_DIR,
    REPORTS_DIR,
)

# ==========================================================
# Camera Settings
# ==========================================================

CAMERA_SOURCE ="data/videos/ex3.mp4"         # Webcam (0) or video file path
FRAME_WIDTH = 1000
FRAME_HEIGHT = 600
CAMERA_FPS = 30

# ==========================================================
# Detection Settings
# ==========================================================
DETECTION_MODEL_PATH = DETECTION_MODEL_DIR / "yolo11l.pt"
DETECTION_CONFIDENCE = 0.25
DETECTION_IMAGE_SIZE = 1536
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF_PRECISION = DEVICE == "cuda"
# ==========================================================
# Pose Settings
# ==========================================================

POSE_MODEL = POSE_MODEL_DIR / "yolo11l-pose.pt"
POSE_CONFIDENCE = 0.50
MIN_POSE_CROP_HEIGHT = 80
POSE_MARGIN = 20

# ==========================================================
# Tracking Settings
# ==========================================================

TRACKER_CONFIG = "bytetrack.yaml"
TRACK_BUFFER = 30
MATCH_THRESHOLD = 0.80
TRACKING_IOU = 0.70
DETECTION_IOU = 0.60
# ==========================================================
# Seat Calibration
# ==========================================================

SEAT_LOCK_TIME = 120
CALIBRATION_TIME = 120
MAX_SEAT_DISTANCE = 80

# ==========================================================
# Behavior Analysis
# ==========================================================

BEHAVIOR_SAMPLE_INTERVAL = 3
BEHAVIOR_HISTORY_SIZE = 90

# ==========================================================
# Risk Engine
# ==========================================================

MAX_RISK_SCORE = 100
LOW_RISK_THRESHOLD = 25
MEDIUM_RISK_THRESHOLD = 50
HIGH_RISK_THRESHOLD = 75
RISK_DECAY = 1

# ==========================================================
# Database
# ==========================================================

DATABASE_FILE = DATABASE_DIR / "examguardian.db"

# ==========================================================
# Logging
# ==========================================================

LOG_FILE = LOGS_DIR / "examguardian.log"

# ==========================================================
# Reports
# ==========================================================

REPORT_OUTPUT = REPORTS_DIR

# ==========================================================
# Display
# ==========================================================

SHOW_FPS = True
SHOW_TRACK_ID = True
SHOW_SEAT_ID = True
SHOW_RISK_SCORE = True
SHOW_POSE = True
SHOW_BOUNDING_BOX = True