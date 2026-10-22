
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "ExamInvigilatorAI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    DATABASE_PATH: str = "backend/data/invigilation.sqlite3"

    YOLO_MODEL: str = "models/yolo26n-pose.pt"
    DEVICE: str = "cuda"
    CONFIDENCE_THRESHOLD: float = 0.20
    YOLO_IMAGE_SIZE: int = 1536
    YOLO_CONFIDENCE: float = 0.20
    YOLO_IOU: float = 0.65
    YOLO_HALF: bool = True
    YOLO_VERBOSE: bool = False
    YOLO_POSE: bool = True

    DIAGNOSTIC_MODE: bool = False
    DIAGNOSTIC_START_FRAME: int = 1
    DIAGNOSTIC_END_FRAME: int = 500
    DIAGNOSTIC_LOG_PATH: str = "backend/logs/tracking_diagnostic.log"

    EVIDENCE_PRE_SECONDS: float = 5.0
    EVIDENCE_POST_SECONDS: float = 5.0

    # Retain the latest 20 seconds of frames in the rolling buffer.
    EVIDENCE_BUFFER_SECONDS: float = 20.0

    HEAD_YAW_THRESHOLD: float = 20.0
    HEAD_OFFSET_THRESHOLD: float = 0.20
    HEAD_TURN_DURATION_SECONDS: float = 2.0
    HEAD_TURN_UNKNOWN_GRACE_SECONDS: float = 0.5

    WRIST_OUTSIDE_DURATION_SECONDS: float = 2.0
    WRIST_CONFIDENCE_THRESHOLD: float = 0.30

    DESK_CALIBRATION_WIDTH: int = 640
    DESK_CALIBRATION_HEIGHT: int = 480
    WRIST_OUTSIDE_WARNING_SECONDS: float = 5.0
    WRIST_OUTSIDE_CONFIRM_SECONDS: float = 10.0

    MEDIAPIPE_HEAD_POSE_ENABLED: bool = True
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = 0.5
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = 0.5
    MEDIAPIPE_FACE_CROP_PADDING: int = 20

    ORIENTATION_ENABLED: bool = True
    ORIENTATION_HISTORY_SIZE: int = 7
    ORIENTATION_LEFT_RIGHT_THRESHOLD: float = 0.28
    ORIENTATION_DOWN_THRESHOLD: float = 0.22
    ORIENTATION_BODY_WEIGHT: float = 0.35
    ORIENTATION_EYE_WEIGHT: float = 0.40
    ORIENTATION_EAR_WEIGHT: float = 0.25

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
