"""
config/paths.py

Centralized project paths for ExamGuardian AI.

Every module imports paths from here instead of creating
its own file or folder paths.
"""

from pathlib import Path

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================================
# Main Directories
# ==========================================================

APP_DIR = PROJECT_ROOT / "app"
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TESTS_DIR = PROJECT_ROOT / "tests"

# ==========================================================
# Model Directories
# ==========================================================

DETECTION_MODEL_DIR = MODELS_DIR / "detection"
POSE_MODEL_DIR = MODELS_DIR / "pose"
TRACKING_MODEL_DIR = MODELS_DIR / "tracking"

# ==========================================================
# Data Directories
# ==========================================================

VIDEOS_DIR = DATA_DIR / "videos"
IMAGES_DIR = DATA_DIR / "images"
SEATS_DIR = DATA_DIR / "seats"
CALIBRATION_DIR = DATA_DIR / "calibration"
DATABASE_DIR = DATA_DIR / "database"
REPORTS_DIR = DATA_DIR / "reports"

# ==========================================================
# Output Directories
# ==========================================================

SCREENSHOTS_DIR = OUTPUTS_DIR / "screenshots"
CLIPS_DIR = OUTPUTS_DIR / "clips"
LOGS_DIR = OUTPUTS_DIR / "logs"
EXPORTS_DIR = OUTPUTS_DIR / "exports"

# ==========================================================
# Create Required Directories Automatically
# ==========================================================

_REQUIRED_DIRECTORIES = [
    MODELS_DIR,
    DATA_DIR,
    OUTPUTS_DIR,
    DETECTION_MODEL_DIR,
    POSE_MODEL_DIR,
    TRACKING_MODEL_DIR,
    VIDEOS_DIR,
    IMAGES_DIR,
    SEATS_DIR,
    CALIBRATION_DIR,
    DATABASE_DIR,
    REPORTS_DIR,
    SCREENSHOTS_DIR,
    CLIPS_DIR,
    LOGS_DIR,
    EXPORTS_DIR,
]

for directory in _REQUIRED_DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)