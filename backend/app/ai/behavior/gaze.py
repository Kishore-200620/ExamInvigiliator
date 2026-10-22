"""
Gaze estimation - Phase 5 Foundation
IMPORTANT: HEAD POSE IS NOT EYE GAZE. Do NOT convert yaw/pitch into gaze.

This module is the interface for future gaze estimation.
Currently returns unavailable - real gaze requires eye/iris landmarks + calibration.
"""
from typing import Optional

import numpy as np

from app.ai.behavior.schemas import GazeObservation, EyesObservation, FaceObservation
from app.ai.tracker.roi import ROI


class GazeAnalyzer:
    """
    Eye gaze estimation interface.

    Future implementation will support:
    - horizontal gaze (-1 left, 0 center, 1 right)
    - vertical gaze (-1 up, 1 down)
    - gaze direction/target (screen, desk, neighbor)
    - gaze confidence

    Currently: NOT IMPLEMENTED - returns available=False
    Do NOT infer gaze from head yaw/pitch alone.
    """

    def __init__(self):
        self.model_name = "not_implemented (requires eye/iris + calibration)"

    def analyze(
        self,
        image,
        face_observation: FaceObservation,
        eyes_observation: EyesObservation,
        roi: ROI,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> GazeObservation:
        """
        Estimate gaze from eyes.

        Args:
            image: full frame
            face_observation: FaceObservation
            eyes_observation: EyesObservation with iris landmarks
            roi: face ROI
            frame_id, timestamp: for observation

        Returns:
            GazeObservation with available=False until real implementation
        """
        obs = GazeObservation(frame_id=frame_id, timestamp=timestamp)

        # Check if we have required signals
        if eyes_observation is None or not eyes_observation.available:
            obs.available = False
            obs.reason = "eyes/iris landmarks not available"
            return obs

        if eyes_observation.iris_landmarks is None or len(eyes_observation.iris_landmarks) == 0:
            obs.available = False
            obs.reason = "iris landmarks not available (need MediaPipe refine_landmarks)"
            return obs

        # Even with iris, we need a calibrated gaze model (e.g., MPIIGaze, ETH-XGaze)
        # Do NOT fake it from head pose
        # For now, explicitly return unavailable
        obs.available = False
        obs.reason = "gaze estimator not implemented - requires MPIIGaze/ETH-XGaze model + per-person calibration, not just head pose"
        obs.horizontal_gaze = None
        obs.vertical_gaze = None
        obs.gaze_direction = None
        obs.gaze_target = None
        obs.confidence = None

        return obs

    def get_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "available": False,
            "reason": "Not implemented - requires eye/iris model + calibration, head pose is not gaze",
        }
