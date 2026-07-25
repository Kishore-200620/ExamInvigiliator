"""
app/camera/camera.py

Camera manager for ExamGuardian AI.
"""

import cv2

from config.settings import (
    CAMERA_SOURCE,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    CAMERA_FPS,
)

from app.utils.logger import logger

from .exceptions import (
    CameraOpenError,
    FrameReadError,
)
from .frame_reader import FrameReader


class Camera:
    """
    Camera manager.

    Responsibilities:
    - Open the camera
    - Configure camera properties
    - Read frames
    - Release resources
    """

    def __init__(self):
        logger.info("Initializing camera...")

        self.capture = cv2.VideoCapture(CAMERA_SOURCE)

        if not self.capture.isOpened():
            raise CameraOpenError(
                f"Unable to open camera source: {CAMERA_SOURCE}"
            )

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        self.reader = FrameReader(self.capture)

        logger.info("Camera initialized successfully.")

    def read(self):
        """
        Read a single frame.

        Returns:
            tuple[bool, ndarray]:
                (True, frame) if successful.
        """

        try:
            frame = self.reader.read()
            return True, frame

        except FrameReadError:
            return False, None

    def release(self):
        """
        Release camera resources.
        """

        if self.capture is not None and self.capture.isOpened():
            self.capture.release()

        logger.info("Camera released.")