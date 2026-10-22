from abc import ABC, abstractmethod

from app.ai.detector.models import Detection
from app.ai.video.frame import Frame


class Detector(ABC):
    """
    Abstract base class for all object detectors.
    """

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the AI model.
        """
        pass

    @abstractmethod
    def detect(self, frame: Frame) -> list[Detection]:
        """
        Detect objects in a frame.

        Args:
            frame: Input Frame

        Returns:
            List of Detection objects.
        """
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """
        Release model resources.
        """
        pass