"""
Hands perception - Phase 5 Foundation
Expected: left/right wrist, hand landmarks, hand bbox, hand movement
No verified model currently available - returns unavailable.
Do NOT generate fake landmarks.
"""
from typing import Optional, List

import numpy as np

from app.ai.behavior.schemas import HandsObservation, BodyPoseObservation
from app.ai.tracker.roi import ROI
from app.ai.detector.models import BoundingBox


class HandsAnalyzer:
    """
    Hand/wrist analysis interface.

    Future: Use body pose wrists + MediaPipe Hands 21 landmarks for detailed hand analysis.
    Distinguish BODY keypoints (wrist 1 point) from HAND landmarks (21 finger points).

    Currently: NOT IMPLEMENTED - returns available=False
    """

    def __init__(self):
        self.model_name = "not_implemented"
        self.model = None
        # Future: MediaPipe Hands
        # Need to decide: wrist from body pose vs hand landmarks
        # Hands require hand ROI: lower 30% of person bbox or wrist-centered crop

    def analyze(
        self,
        image: np.ndarray,
        roi: ROI,
        body_observation: BodyPoseObservation = None,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> HandsObservation:
        """
        Analyze hands for ROI.

        Args:
            image: full frame
            roi: body/person ROI
            body_observation: BodyPoseObservation with wrist positions if available
            frame_id, timestamp: for observation

        Returns:
            HandsObservation with available=False until real model integrated
        """
        obs = HandsObservation(frame_id=frame_id, timestamp=timestamp)

        if not roi.valid:
            obs.available = False
            return obs

        # Check if body pose provides wrist positions
        has_wrist = False
        if body_observation and body_observation.available:
            if body_observation.left_wrist is not None or body_observation.right_wrist is not None:
                has_wrist = True

        # No verified hand model available - explicitly unavailable
        # Do NOT fabricate landmarks
        # Even if we have wrist from body, hand landmarks still need MediaPipe Hands
        obs.available = False
        obs.left_wrist = body_observation.left_wrist if has_wrist and body_observation else None
        obs.right_wrist = body_observation.right_wrist if has_wrist and body_observation else None
        obs.left_hand_landmarks = None
        obs.right_hand_landmarks = None
        obs.left_hand_bbox = None
        obs.right_hand_bbox = None
        obs.left_hand_movement = None
        obs.right_hand_movement = None
        obs.confidence = None

        # If no body wrist either, still unavailable
        if not has_wrist:
            # No hand model and no wrist from body
            obs.left_wrist = None
            obs.right_wrist = None

        return obs

    def get_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "available": False,
            "reason": "No verified hand model - requires body pose wrist + MediaPipe Hands 21 landmarks",
            "note": "Wrist (body 1 point) vs hand landmarks (21 finger points) are different - wrist alone does not give phone interaction",
        }
