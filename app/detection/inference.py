"""
app/detection/inference.py

YOLO inference engine for ExamGuardian AI.
"""

from ultralytics import YOLO
from config.settings import (
    DETECTION_MODEL_PATH,
    DETECTION_CONFIDENCE,
    DETECTION_IMAGE_SIZE,
    DETECTION_IOU,
    DEVICE,
)
from app.utils.logger import logger


class InferenceEngine:
    """
    Runs YOLO inference.
    """

    def __init__(self):
        logger.info("Loading YOLO detection model...")

        self.model = YOLO(DETECTION_MODEL_PATH)

        logger.info("YOLO detection model loaded successfully.")

    def infer(self, frame):
        """
        Run inference on a frame.
        """
        return self.model(
    frame,
    imgsz=DETECTION_IMAGE_SIZE,
    conf=DETECTION_CONFIDENCE,
    iou=DETECTION_IOU,
    device=DEVICE,
    augment=True,
    verbose=False,
)