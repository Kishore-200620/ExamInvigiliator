from abc import ABC, abstractmethod

from app.ai.detector.models import BoundingBox


class MotionModel(ABC):
    """
    Base interface for motion prediction models.
    """

    @abstractmethod
    def initialize(self, bbox: BoundingBox) -> None:
        """
        Initialize the motion model.
        """
        pass

    @abstractmethod
    def predict(self) -> BoundingBox:
        """
        Predict the next bounding box.
        """
        pass

    @abstractmethod
    def update(self, bbox: BoundingBox) -> None:
        """
        Update the motion model with a new detection.
        """
        pass