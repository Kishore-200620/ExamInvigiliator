"""
app/camera/frame_reader.py

Frame reader for ExamGuardian AI.
"""

from app.utils.logger import logger

from .exceptions import FrameReadError


class FrameReader:
    """
    Reads frames from an opened camera.
    """

    def __init__(self, capture):
        self.capture = capture

    def read(self):
        """
        Read a single frame.
        """
        if self.capture is None or not self.capture.isOpened():
            raise FrameReadError("Camera is not initialized or is closed.")
        success, frame = self.capture.read()

        if not success:
            logger.warning("Unable to read frame.")
            raise FrameReadError("Frame could not be read.")

        return frame