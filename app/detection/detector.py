"""
app/detection/detector.py

Detection pipeline.
"""

from .inference import InferenceEngine
from .object_filter import ObjectFilter


class Detector:
    """
    Detection pipeline.
    """

    def __init__(self):
        self.inference = InferenceEngine()

    def detect(self, frame):
        """
        Detect objects in a frame.
        """

        raw_results = self.inference.infer(frame)

        return ObjectFilter.filter(raw_results)