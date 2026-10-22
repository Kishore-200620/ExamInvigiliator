"""
Head Pose 3D Estimation — MediaPipe Face Mesh + solvePnP

Ported from exam_invigiliator_2/ai/head_pose.py into root project architecture.

Provides true 3D head pose estimation (yaw, pitch, roll) using:
1. MediaPipe Face Mesh (468 landmarks) for face detection
2. cv2.solvePnP with 6 reference points for 3D pose estimation
3. Euler angle extraction from rotation matrix

This replaces the approximate YOLO-keypoint geometry approach
(backend/app/ai/tracker/head_pose.py) with accurate 3D angles.

The keypoint geometry module is kept as a fast fallback when
MediaPipe fails to detect a face (occlusion, extreme angles).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class HeadPose3DResult:
    """Result from MediaPipe + solvePnP head pose estimation."""

    yaw: float = 0.0       # degrees, left-right (-90..90)
    pitch: float = 0.0     # degrees, up-down (-90..90)
    roll: float = 0.0      # degrees, tilt (-90..90)
    nose: Optional[Tuple[int, int]] = None  # nose point in original frame coords
    rotation_vector: Optional[np.ndarray] = None
    translation_vector: Optional[np.ndarray] = None
    direction: str = "UNKNOWN"  # CENTER, LEFT, RIGHT, DOWN, UNKNOWN
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# MediaPipe Landmark Indices for solvePnP (6 points)
# ---------------------------------------------------------------------------

LANDMARK_IDS = {
    "nose": 1,
    "chin": 199,
    "left_eye": 33,
    "right_eye": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}


# ---------------------------------------------------------------------------
# Generic 3D Facial Model Points (millimeters)
# ---------------------------------------------------------------------------

MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -63.6, -12.5),      # Chin
        (-43.3, 32.7, -26.0),     # Left eye outer corner
        (43.3, 32.7, -26.0),      # Right eye outer corner
        (-28.9, -28.9, -24.1),    # Left mouth corner
        (28.9, -28.9, -24.1),     # Right mouth corner
    ],
    dtype=np.float64,
)


class HeadPose3DEstimator:
    """
    MediaPipe Face Mesh + solvePnP based head pose estimator.

    Provides true 3D yaw/pitch/roll in degrees.
    Each detection's bounding box is cropped to a face ROI,
    MediaPipe extracts 468 landmarks, and solvePnP derives
    rotation angles from 6 reference points.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        padding: int = 20,
    ) -> None:
        """
        Initialize MediaPipe Face Mesh.

        Parameters
        ----------
        min_detection_confidence
            MediaPipe face detection confidence threshold.
        min_tracking_confidence
            MediaPipe face tracking confidence threshold.
        padding
            Pixels to pad around the person bbox for face ROI crop.
        """
        try:
            import mediapipe as mp
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._available = True
            logger.info("HeadPose3DEstimator: MediaPipe Face Mesh initialized")
        except ImportError:
            self._face_mesh = None
            self._available = False
            logger.warning(
                "HeadPose3DEstimator: mediapipe not installed, "
                "3D head pose will be unavailable"
            )

        self._padding = padding

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._available

    def estimate(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[HeadPose3DResult]:
        """
        Estimate head pose for a single tracked person.

        Parameters
        ----------
        frame
            Full original frame (BGR).
        bbox
            Person bounding box as (x1, y1, x2, y2).

        Returns
        -------
        HeadPose3DResult or None if estimation fails.
        """
        if not self._available or self._face_mesh is None:
            return None

        # Crop face ROI from person bbox
        roi, offset = self._crop_face(frame, bbox)
        if roi.size == 0:
            return None

        # MediaPipe requires RGB
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        roi_h, roi_w = roi.shape[:2]

        # Extract 2D image points from landmarks
        image_points = self._extract_image_points(face_landmarks, roi_w, roi_h)

        # Camera intrinsic matrix (approximate)
        camera_matrix = self._camera_matrix(roi_w, roi_h)
        dist_coeffs = self._dist_coeffs()

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        # Convert rotation vector to matrix, then to Euler angles
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        yaw, pitch, roll = self._rotation_matrix_to_euler(rotation_matrix)

        # Nose point in original frame coordinates
        nose = (
            int(image_points[0][0] + offset[0]),
            int(image_points[0][1] + offset[1]),
        )

        # Derive direction from yaw
        direction = self._classify_direction(yaw, pitch)

        # Confidence based on how well landmarks were detected
        # MediaPipe doesn't expose a single confidence, so we use
        # a heuristic: inverse of angular extremity
        confidence = self._compute_confidence(yaw, pitch, roll)

        return HeadPose3DResult(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            nose=nose,
            rotation_vector=rotation_vector,
            translation_vector=translation_vector,
            direction=direction,
            confidence=confidence,
        )

    def estimate_for_tracks(
        self,
        frame: np.ndarray,
        tracks: list,
        match_result,
        detections: list,
    ) -> dict[int, HeadPose3DResult]:
        """
        Estimate head pose for all CONFIRMED tracks with matched detections.

        Parameters
        ----------
        frame
            Full original frame (BGR, from Frame.image).
        tracks
            List of StudentTrack objects.
        match_result
            Match result from Hungarian matching (has .matches).
        detections
            List of Detection objects.

        Returns
        -------
        Dict mapping track_id -> HeadPose3DResult for successful estimates.
        """
        if not self._available or frame is None:
            return {}

        from app.ai.tracker.models import TrackState

        results: dict[int, HeadPose3DResult] = {}

        # Build track_idx -> det_idx map
        match_map: dict[int, int] = {}
        try:
            for t_idx, d_idx in match_result.matches:
                match_map[int(t_idx)] = int(d_idx)
        except Exception:
            return results

        for t_idx, track in enumerate(tracks):
            if track.state != TrackState.CONFIRMED:
                continue

            d_idx = match_map.get(t_idx)
            if d_idx is None or d_idx < 0 or d_idx >= len(detections):
                continue

            det = detections[d_idx]
            bbox = (det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2)

            result = self.estimate(frame, bbox)
            if result is not None:
                results[track.track_id] = result

        return results

    def visualize(
        self,
        frame: np.ndarray,
        track_poses: dict[int, HeadPose3DResult],
        tracks: list,
    ) -> np.ndarray:
        """
        Draw 3D head pose axes and labels on the frame.

        Parameters
        ----------
        frame
            Frame to annotate.
        track_poses
            Dict mapping track_id -> HeadPose3DResult.
        tracks
            List of StudentTrack objects (for bbox positions).

        Returns
        -------
        Annotated frame.
        """
        from app.ai.tracker.models import TrackState

        for track in tracks:
            if track.state != TrackState.CONFIRMED:
                continue

            pose = track_poses.get(track.track_id)
            if pose is None:
                continue

            bbox = track.motion.bbox
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

            # Draw 3D axes from nose point
            if (
                pose.nose is not None
                and pose.rotation_vector is not None
                and pose.translation_vector is not None
            ):
                roi, offset = self._crop_face(frame, (x1, y1, x2, y2))
                if roi.size > 0:
                    roi_h, roi_w = roi.shape[:2]
                    camera_matrix = self._camera_matrix(roi_w, roi_h)
                    dist_coeffs = self._dist_coeffs()

                    axis = np.float32(
                        [[50, 0, 0], [0, 50, 0], [0, 0, 50]]
                    )
                    axis_points, _ = cv2.projectPoints(
                        axis,
                        pose.rotation_vector,
                        pose.translation_vector,
                        camera_matrix,
                        dist_coeffs,
                    )
                    axis_points = axis_points.reshape(-1, 2)
                    origin = np.array(pose.nose)

                    x_end = (
                        int(axis_points[0][0] + offset[0]),
                        int(axis_points[0][1] + offset[1]),
                    )
                    y_end = (
                        int(axis_points[1][0] + offset[0]),
                        int(axis_points[1][1] + offset[1]),
                    )
                    z_end = (
                        int(axis_points[2][0] + offset[0]),
                        int(axis_points[2][1] + offset[1]),
                    )

                    # Draw nose point
                    cv2.circle(frame, tuple(origin), 4, (0, 255, 255), -1)
                    # X axis (red)
                    cv2.line(frame, tuple(origin), x_end, (0, 0, 255), 2)
                    # Y axis (green)
                    cv2.line(frame, tuple(origin), y_end, (0, 255, 0), 2)
                    # Z axis (blue)
                    cv2.line(frame, tuple(origin), z_end, (255, 0, 0), 2)

            # Draw yaw/pitch/roll text
            cv2.putText(
                frame,
                f"Yaw: {pose.yaw:.1f}",
                (x1, max(25, y1 - 50)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Pitch: {pose.pitch:.1f}",
                (x1, max(40, y1 - 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Roll: {pose.roll:.1f}",
                (x1, max(55, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        return frame

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception:
                pass
            self._face_mesh = None
            self._available = False
            logger.info("HeadPose3DEstimator: MediaPipe resources released")

    def get_info(self) -> dict:
        return {
            "model_name": "mediapipe_face_mesh_solvepnp",
            "method": "MediaPipe 468 landmarks + cv2.solvePnP",
            "available": self._available,
            "landmarks": 468,
            "pnp_points": 6,
        }

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _crop_face(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crop face ROI from person bounding box with padding.

        Returns (roi_image, (offset_x, offset_y)).
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        pad = self._padding

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        roi = frame[y1:y2, x1:x2]
        return roi, (x1, y1)

    @staticmethod
    def _camera_matrix(width: int, height: int) -> np.ndarray:
        """Approximate intrinsic camera matrix."""
        focal_length = width
        center = (width / 2, height / 2)
        return np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _dist_coeffs() -> np.ndarray:
        """Assume zero lens distortion."""
        return np.zeros((4, 1), dtype=np.float64)

    @staticmethod
    def _extract_image_points(
        landmarks,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Convert MediaPipe landmarks to image coordinates for 6 PnP points."""
        image_points = []
        for idx in LANDMARK_IDS.values():
            lm = landmarks.landmark[idx]
            image_points.append((lm.x * width, lm.y * height))
        return np.asarray(image_points, dtype=np.float64)

    @staticmethod
    def _rotation_matrix_to_euler(
        rotation_matrix: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles.

        Returns (yaw, pitch, roll) in degrees.
        """
        sy = np.sqrt(
            rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2
        )

        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(
                rotation_matrix[2, 1], rotation_matrix[2, 2]
            )
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = np.arctan2(
                rotation_matrix[1, 0], rotation_matrix[0, 0]
            )
        else:
            pitch = np.arctan2(
                -rotation_matrix[1, 2], rotation_matrix[1, 1]
            )
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = 0.0

        return (
            float(np.degrees(yaw)),
            float(np.degrees(pitch)),
            float(np.degrees(roll)),
        )

    @staticmethod
    def _classify_direction(yaw: float, pitch: float) -> str:
        """
        Classify head direction from yaw/pitch angles.

        Uses HEAD_YAW_THRESHOLD from settings for LEFT/RIGHT,
        and a pitch threshold for DOWN.
        """
        try:
            from app.config.settings import settings
            yaw_threshold = getattr(settings, "HEAD_YAW_THRESHOLD", 20.0)
        except Exception:
            yaw_threshold = 20.0

        # Check for looking down first (pitch > 25 degrees)
        if pitch > 25.0:
            return "DOWN"

        if yaw < -yaw_threshold:
            return "LEFT"
        elif yaw > yaw_threshold:
            return "RIGHT"

        return "CENTER"

    @staticmethod
    def _compute_confidence(yaw: float, pitch: float, roll: float) -> float:
        """
        Heuristic confidence based on angle magnitude.

        Extreme angles (>60°) get lower confidence since
        solvePnP degrades at profile views.
        """
        max_angle = max(abs(yaw), abs(pitch), abs(roll))
        if max_angle > 80:
            return 0.3
        elif max_angle > 60:
            return 0.5
        elif max_angle > 40:
            return 0.7
        else:
            return 0.9
