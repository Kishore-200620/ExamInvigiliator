"""
Eyes/Iris perception - Phase 5 Foundation
Uses MediaPipe face landmarks if available to extract eye/iris geometry.
Does NOT claim gaze direction - that is separate (gaze.py).
"""
from typing import Optional, List

import cv2
import numpy as np

from app.ai.behavior.schemas import EyesObservation
from app.ai.detector.models import BoundingBox
from app.ai.tracker.roi import ROI


class EyesAnalyzer:
    """
    Extracts eye/iris landmarks and eye openness (EAR) where reliable.
    Requires face landmarks from FaceAnalyzer/MediaPipe.

    MediaPipe Face Landmarker 468 includes:
    - Left eye: landmarks 33, 133, 159, 145, 153, 154 etc
    - Right eye: 362, 263, 386, 374, 380, 381
    - Iris: 468-478 (when refine_landmarks=True)
    """

    # MediaPipe eye landmark indices (from documentation)
    LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    # Iris indices when refine_landmarks=True: left iris 468-472, right iris 473-477
    LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
    RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

    def __init__(self):
        self.model_name = "mediapipe_eyes (via Face Landmarker 468+10 iris)"
        # No separate model - uses face landmarks

    def _eye_openness(self, eye_landmarks: List[tuple[float, float]]) -> Optional[float]:
        """
        Calculate Eye Aspect Ratio (EAR) as openness metric.
        EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|)
        Requires at least 6 landmarks for eye.
        Returns 0-1, where 0 is closed, ~0.3 is open.
        """
        if eye_landmarks is None or len(eye_landmarks) < 6:
            return None
        try:
            # Use 6 points: p1 (outer corner), p4 (inner), p2/p6 (top/bottom)
            # For MediaPipe 16-point eye, approximate with indices
            # Simplified: use vertical/horizontal ratio
            # For now, use first 6 points if available
            p1 = np.array(eye_landmarks[0])
            p4 = np.array(eye_landmarks[3])
            p2 = np.array(eye_landmarks[1])
            p6 = np.array(eye_landmarks[5])
            p3 = np.array(eye_landmarks[2])
            p5 = np.array(eye_landmarks[4])

            # EAR calculation
            vert1 = np.linalg.norm(p2 - p6)
            vert2 = np.linalg.norm(p3 - p5)
            horiz = np.linalg.norm(p1 - p4)
            if horiz == 0:
                return None
            ear = (vert1 + vert2) / (2.0 * horiz)
            # Clamp to 0-0.5 range typical
            ear = float(np.clip(ear, 0.0, 0.5) / 0.5)  # normalize to 0-1
            return ear
        except Exception:
            return None

    def analyze(
        self,
        image: np.ndarray,
        face_observation,
        roi: ROI,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> EyesObservation:
        """
        Analyze eyes from face landmarks.

        Args:
            image: full frame
            face_observation: FaceObservation from FaceAnalyzer (must have landmarks)
            roi: ROI for face
            frame_id, timestamp: for observation

        Returns:
            EyesObservation with explicit availability
        """
        obs = EyesObservation(frame_id=frame_id, timestamp=timestamp)

        if face_observation is None or not face_observation.landmarks_available:
            obs.available = False
            obs.confidence = None
            obs.quality = 0.0
            return obs

        if face_observation.landmarks is None or len(face_observation.landmarks) < 468:
            obs.available = False
            obs.quality = 0.1
            return obs

        try:
            landmarks = face_observation.landmarks

            # Extract left eye landmarks
            left_eye_pts = []
            for idx in self.LEFT_EYE_INDICES:
                if idx < len(landmarks):
                    left_eye_pts.append(landmarks[idx])

            # Extract right eye
            right_eye_pts = []
            for idx in self.RIGHT_EYE_INDICES:
                if idx < len(landmarks):
                    right_eye_pts.append(landmarks[idx])

            # Iris: check if we have 478 landmarks (468 + 10 iris)
            iris_available = len(landmarks) >= 478
            left_iris = []
            right_iris = []
            if iris_available:
                for idx in self.LEFT_IRIS_INDICES:
                    if idx < len(landmarks):
                        left_iris.append(landmarks[idx])
                for idx in self.RIGHT_IRIS_INDICES:
                    if idx < len(landmarks):
                        right_iris.append(landmarks[idx])

            # Only mark available if we have reasonable eye points
            if len(left_eye_pts) >= 6 and len(right_eye_pts) >= 6:
                obs.available = True
                obs.left_eye_landmarks = left_eye_pts
                obs.right_eye_landmarks = right_eye_pts
                obs.iris_landmarks = (left_iris + right_iris) if iris_available else None
                obs.left_eye_openness = self._eye_openness(left_eye_pts)
                obs.right_eye_openness = self._eye_openness(right_eye_pts)
                obs.eye_openness_available = obs.left_eye_openness is not None
                # Blink detection requires temporal history - not available in single frame
                obs.blink_detected = None
                obs.confidence = face_observation.confidence  # inherit if available, else None
                obs.quality = face_observation.quality  # geometric quality from face
            else:
                obs.available = False
                obs.quality = 0.2

            return obs

        except Exception:
            obs.available = False
            obs.quality = 0.0
            return obs

    def get_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "available": True,  # Always available as analyzer, but per-frame available flag indicates result
        }
