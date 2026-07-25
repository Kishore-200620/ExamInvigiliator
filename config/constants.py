"""
config/constants.py

Project-wide constants used across ExamGuardian AI.
"""

# ==========================================================
# Object Detection Classes (COCO Dataset)
# ==========================================================

PERSON_CLASS_ID = 0
BICYCLE_CLASS_ID = 1
CAR_CLASS_ID = 2
MOTORCYCLE_CLASS_ID = 3
BUS_CLASS_ID = 5
TRAIN_CLASS_ID = 6
TRUCK_CLASS_ID = 7
CELL_PHONE_CLASS_ID = 67
BOOK_CLASS_ID = 73

# ==========================================================
# Seat Status
# ==========================================================

SEAT_EMPTY = "EMPTY"
SEAT_OCCUPIED = "OCCUPIED"
SEAT_LOCKED = "LOCKED"

# ==========================================================
# Zone Names
# ==========================================================

SAFE_ZONE = "SAFE"
WATCH_ZONE = "WATCH"
ALERT_ZONE = "ALERT"

# ==========================================================
# Risk Levels
# ==========================================================

RISK_LOW = "LOW"
RISK_MEDIUM = "MEDIUM"
RISK_HIGH = "HIGH"
RISK_CRITICAL = "CRITICAL"

# ==========================================================
# Tracking Status
# ==========================================================

TRACK_ACTIVE = "ACTIVE"
TRACK_LOST = "LOST"
TRACK_REMOVED = "REMOVED"

# ==========================================================
# Colors (BGR for OpenCV)
# ==========================================================

COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

# ==========================================================
# Supported File Extensions
# ==========================================================

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
)

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
)

# ==========================================================
# Application Information
# ==========================================================

APP_NAME = "ExamGuardian AI"
APP_VERSION = "1.0.0"

# ==========================================================
# Default Labels
# ==========================================================

UNKNOWN_STUDENT = "Unknown"
UNKNOWN_SEAT = "Unknown Seat"