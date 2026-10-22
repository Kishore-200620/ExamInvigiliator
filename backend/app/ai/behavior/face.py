"""
Face perception - Phase 5 Foundation
Uses existing MediaPipe Face Landmarker infrastructure where safe.
Returns structured FaceObservation, never fabricates model confidence.
"""
from typing import Optional

import cv2
import numpy as np

from app.ai.behavior.schemas import FaceObservation
from app.ai.detector.models import BoundingBox
from app.ai.tracker.roi import ROI


class FaceAnalyzer:
    """
    Face detection/landmark observation per student.

    Uses MediaPipe Face Landmarker 468 landmarks if available.
    Distinguishes model confidence vs geometric quality.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.model_name = "uninitialized"
        self._load_model()

    def _load_model(self):
        """Reuse existing MediaPipe infrastructure from head_pose.py if available."""
        try:
            # Try to reuse HeadPoseEstimator's MediaPipe loading pattern
            from app.ai.tracker.head_pose import HeadPoseEstimator

            # Create a temporary estimator to check if MediaPipe is available
            # But don't keep it - we need face-specific logic
            # Instead, try to load MediaPipe directly as head_pose does
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import pathlib, urllib.request

            model_path = pathlib.Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"
            if not model_path.exists():
                # Download if not exists - same URL as head_pose.py
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                try:
                    urllib.request.urlretrieve(url, str(model_path))
                except Exception:
                    pass

            if model_path.exists():
                base_options = python.BaseOptions(model_asset_path=str(model_path))
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_faces=1
                )
                self.model = vision.FaceLandmarker.create_from_options(options)
                self.model_name = "mediapipe_face_landmarker_468"
                return
        except Exception as e:
            # MediaPipe not available or failed
            self.model = None
            self.model_name = f"unavailable: {e}"

        # Fallback: no model
        self.model = None
        self.model_name = "unavailable (no MediaPipe)"

    def analyze(
        self,
        image: np.ndarray,
        roi: ROI,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> FaceObservation:
        """
        Analyze face in ROI.

        Args:
            image: full frame image (BGR)
            roi: ROI object from ROIManager (should be head ROI)
            frame_id, timestamp: for observation

        Returns:
            FaceObservation with explicit availability
        """
        obs = FaceObservation(frame_id=frame_id, timestamp=timestamp)

        if not roi.valid:
            obs.detected = False
            obs.quality = 0.0
            return obs

        # Crop using ROI
        from app.ai.tracker.roi import ROIManager

        # Use ROI bbox directly (already clipped)
        bbox = roi.bbox
        h, w = image.shape[:2]
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

        # Validate crop size
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w < 20 or crop_h < 20:
            obs.detected = False
            obs.bbox = bbox
            obs.quality = 0.1  # geometric quality: too small
            return obs

        if self.model is None:
            # No model - cannot detect face, but ROI exists
            obs.detected = False
            obs.bbox = bbox
            obs.landmarks_available = False
            obs.confidence = None  # No model confidence
            obs.quality = 0.3  # heuristic: ROI valid but no detection
            return obs

        # Run MediaPipe
        try:
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                obs.detected = False
                obs.bbox = bbox
                return obs

            # MediaPipe expects RGB 128x128
            crop_resized = cv2.resize(crop, (128, 128))
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

            import mediapipe as mp

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
            result = self.model.detect(mp_image)

            if not result.face_landmarks or len(result.face_landmarks) == 0:
                obs.detected = False
                obs.bbox = bbox
                obs.landmarks_available = False
                obs.confidence = None
                obs.quality = 0.2  # no face found
                return obs

            # Face detected
            obs.detected = True
            obs.bbox = bbox
            obs.landmarks_available = True
            # Convert landmarks to list of (x,y) normalized
            landmarks = result.face_landmarks[0]
            obs.landmarks = [(lm.x, lm.y) for lm in landmarks]
            obs.landmark_count = len(landmarks)
            # MediaPipe Face Landmarker does NOT expose a single confidence for face
            # So we keep confidence as None and use quality as geometric
            obs.confidence = None
            # Geometric quality: based on face size and detection
            # Larger face = higher quality
            face_area = crop_w * crop_h
            # Normalize to 0-1 based on 40x40 min and 200x200 good
            quality = min(1.0, max(0.1, (face_area - 1600) / (40000 - 1600)))
            obs.quality = float(quality)
            # Landmark count should be 468
            if obs.landmark_count != 468:
                obs.quality = min(obs.quality, 0.5)

            return obs

        except Exception as e:
            obs.detected = False
            obs.bbox = bbox
            obs.quality = 0.0
            return obs

    def get_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "available": self.model is not None,
        }
