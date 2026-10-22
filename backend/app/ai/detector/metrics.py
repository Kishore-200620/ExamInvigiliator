from dataclasses import dataclass


@dataclass(slots=True)
class DetectorMetrics:
    """
    Runtime performance metrics for the detector.
    """

    inference_time_ms: float = 0.0

    processing_fps: float = 0.0

    detections: int = 0

    total_frames: int = 0

    average_inference_ms: float = 0.0

    average_fps: float = 0.0