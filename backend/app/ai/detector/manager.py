from app.ai.detector.base import Detector
from app.ai.detector.models import Detection
from app.ai.video.frame import Frame


class DetectionManager:
    """
    High-level interface for object detection.
    """

    def __init__(self, detector: Detector):
        self.detector = detector

    def start(self):
        self.detector.load_model()

    def detect(
        self,
        frame: Frame,
    ) -> list[Detection]:

        return self.detector.detect(frame)

    def detect_batch(self, frames: list[Frame]) -> list[list[Detection]]:
        """Run batched inference when the detector supports it; preserve order."""
        batch_detector = getattr(self.detector, "detect_batch", None)
        if callable(batch_detector):
            return batch_detector(frames)
        return [self.detector.detect(frame) for frame in frames]
    def get_metrics(self):
        return self.detector.get_metrics()
    def stop(self):
        self.detector.unload_model()
