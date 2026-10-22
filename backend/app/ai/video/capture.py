import cv2
import numpy as np

from app.ai.video.base import VideoSource


class OpenCVCapture(VideoSource):
    """
    OpenCV implementation of the VideoSource interface.

    Responsibilities:
    - Open webcam/video/RTSP
    - Read raw frames
    - Release resources

    Does NOT:
    - Calculate FPS
    - Create Frame objects
    - Manage threads
    - Buffer frames
    """

    def __init__(self, source: int | str = 0):
        self.source = source
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> bool:
        """Open the video source."""
        self.capture = cv2.VideoCapture(self.source)

        if self.capture is None:
            return False

        return self.capture.isOpened()

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Read a raw frame from OpenCV.

        Returns:
            (success, image)
        """

        if self.capture is None:
            return False, None

        return self.capture.read()

    def is_opened(self) -> bool:
        """Check whether the video source is open."""

        return (
            self.capture is not None
            and self.capture.isOpened()
        )

    def release(self) -> None:
        """Release the video source."""

        if self.capture is not None:
            self.capture.release()
            self.capture = None