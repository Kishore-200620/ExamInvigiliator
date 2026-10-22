"""
Body Orientation Estimation — Weighted Keypoint Fusion

Ported from exam_invigiliator_2/ai/orientation.py into root project architecture.

Estimates student body orientation (FORWARD / LEFT / RIGHT / DOWN / UNKNOWN)
using YOLO pose keypoints with a weighted multi-signal fusion approach:
  - Body score  (0.35): nose horizontal offset from shoulder center
  - Eye score   (0.40): asymmetry of nose-to-left-eye vs nose-to-right-eye
  - Ear score   (0.25): asymmetry of nose-to-left-ear vs nose-to-right-ear

Temporal smoothing via majority vote over a configurable history window per track.

This module does NOT use MediaPipe — it relies solely on YOLO's 17 COCO keypoints
which are already extracted in the detection pipeline (metadata["keypoints"]).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import math
import numpy as np

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# COCO Keypoint Indices
# ---------------------------------------------------------------------------

COCO_NOSE = 0
COCO_LEFT_EYE = 1
COCO_RIGHT_EYE = 2
COCO_LEFT_EAR = 3
COCO_RIGHT_EAR = 4
COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class OrientationResult:
    """Result from weighted-fusion body orientation estimation."""

    direction: str = "UNKNOWN"  # FORWARD, LEFT, RIGHT, DOWN, UNKNOWN
    confidence: float = 0.0
    score: float = 0.0  # raw weighted score (negative=LEFT, positive=RIGHT)
    body_score: float = 0.0  # individual component scores for debugging
    eye_score: float = 0.0
    ear_score: float = 0.0
    is_down: bool = False
    is_smoothed: bool = False  # True if temporal smoothing was applied


# ---------------------------------------------------------------------------
# Orientation Estimator
# ---------------------------------------------------------------------------

class OrientationEstimator:
    """
    Estimates student body orientation from YOLO pose keypoints.

    Uses weighted multi-signal fusion of body, eye, and ear components,
    with temporal majority-vote smoothing per tracked student.
    """

    def __init__(
        self,
        body_weight: float = 0.35,
        eye_weight: float = 0.40,
        ear_weight: float = 0.25,
        left_right_threshold: float = 0.28,
        down_threshold: float = 0.22,
        history_size: int = 7,
        min_confidence: float = 0.30,
    ) -> None:
        """
        Parameters
        ----------
        body_weight
            Weight for body (nose vs shoulder center) signal.
        eye_weight
            Weight for eye asymmetry signal.
        ear_weight
            Weight for ear asymmetry signal.
        left_right_threshold
            Weighted score threshold for LEFT/RIGHT classification.
        down_threshold
            Nose-below-eyes ratio for DOWN detection.
        history_size
            Number of frames for temporal smoothing (majority vote).
        min_confidence
            Minimum keypoint confidence to consider valid.
        """
        self._body_weight = body_weight
        self._eye_weight = eye_weight
        self._ear_weight = ear_weight
        self._left_right_threshold = left_right_threshold
        self._down_threshold = down_threshold
        self._history_size = history_size
        self._min_confidence = min_confidence

        # Per-track orientation history for temporal smoothing
        self._history: dict[int, list[str]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        keypoints: np.ndarray,
    ) -> OrientationResult:
        """
        Estimate orientation from YOLO keypoints (17,3) array.

        Parameters
        ----------
        keypoints
            YOLO keypoints as (17,3) numpy array with (x, y, confidence).

        Returns
        -------
        OrientationResult with direction, confidence, and debug scores.
        """
        if keypoints is None:
            return OrientationResult()

        kp = np.array(keypoints)
        if kp.ndim != 2 or kp.shape[0] < 7 or kp.shape[1] < 3:
            return OrientationResult()

        # Extract relevant keypoints with confidence check
        nose = self._get_kp(kp, COCO_NOSE)
        left_eye = self._get_kp(kp, COCO_LEFT_EYE)
        right_eye = self._get_kp(kp, COCO_RIGHT_EYE)
        left_ear = self._get_kp(kp, COCO_LEFT_EAR)
        right_ear = self._get_kp(kp, COCO_RIGHT_EAR)
        left_shoulder = self._get_kp(kp, COCO_LEFT_SHOULDER)
        right_shoulder = self._get_kp(kp, COCO_RIGHT_SHOULDER)

        if nose is None or left_shoulder is None or right_shoulder is None:
            return OrientationResult()

        # Check for DOWN orientation
        is_down = self._check_down(nose, left_eye, right_eye, left_shoulder, right_shoulder)
        if is_down:
            return OrientationResult(
                direction="DOWN",
                confidence=0.8,
                score=0.0,
                is_down=True,
            )

        # Compute component scores
        body_score = self._body_score(nose, left_shoulder, right_shoulder)
        eye_score = self._eye_score(nose, left_eye, right_eye)
        ear_score = self._ear_score(nose, left_ear, right_ear)

        # Weighted fusion
        total_weight = 0.0
        weighted_sum = 0.0

        if body_score is not None:
            weighted_sum += self._body_weight * body_score
            total_weight += self._body_weight

        if eye_score is not None:
            weighted_sum += self._eye_weight * eye_score
            total_weight += self._eye_weight

        if ear_score is not None:
            weighted_sum += self._ear_weight * ear_score
            total_weight += self._ear_weight

        if total_weight < 0.1:
            return OrientationResult()

        final_score = weighted_sum / total_weight

        # Classify direction
        if final_score < -self._left_right_threshold:
            direction = "LEFT"
        elif final_score > self._left_right_threshold:
            direction = "RIGHT"
        else:
            direction = "FORWARD"

        # Confidence from score magnitude
        confidence = min(0.95, 0.5 + abs(final_score))

        return OrientationResult(
            direction=direction,
            confidence=confidence,
            score=final_score,
            body_score=body_score or 0.0,
            eye_score=eye_score or 0.0,
            ear_score=ear_score or 0.0,
        )

    def estimate_with_smoothing(
        self,
        track_id: int,
        keypoints: np.ndarray,
    ) -> OrientationResult:
        """
        Estimate orientation with per-track temporal smoothing.

        Parameters
        ----------
        track_id
            Track identifier for history lookup.
        keypoints
            YOLO keypoints as (17,3) numpy array.

        Returns
        -------
        OrientationResult with smoothed direction via majority vote.
        """
        raw = self.estimate(keypoints)

        if raw.direction == "UNKNOWN":
            return raw

        # Append to history
        history = self._history[track_id]
        history.append(raw.direction)

        # Cap history size
        if len(history) > self._history_size:
            self._history[track_id] = history[-self._history_size:]
            history = self._history[track_id]

        # Majority vote
        if len(history) >= 3:
            counter = Counter(history)
            smoothed_dir, count = counter.most_common(1)[0]
            raw.direction = smoothed_dir
            raw.is_smoothed = True

        return raw

    def estimate_for_tracks(
        self,
        tracks: list,
        detections: list,
        match_result=None,
    ) -> dict[int, OrientationResult]:
        """
        Estimate orientation for all CONFIRMED tracks.

        Parameters
        ----------
        tracks
            List of StudentTrack objects.
        detections
            List of Detection objects (with keypoints in metadata).
        match_result
            Optional match result (has .matches) to find matched detections.
            If None, uses track's own keypoint data from metadata.

        Returns
        -------
        Dict mapping track_id -> OrientationResult.
        """
        from app.ai.tracker.models import TrackState

        results: dict[int, OrientationResult] = {}

        # Build track_idx -> det_idx mapping
        match_map: dict[int, int] = {}
        if match_result is not None:
            try:
                for t_idx, d_idx in match_result.matches:
                    match_map[int(t_idx)] = int(d_idx)
            except Exception:
                pass

        for t_idx, track in enumerate(tracks):
            if track.state != TrackState.CONFIRMED:
                continue

            keypoints = None

            # Try to get keypoints from matched detection
            d_idx = match_map.get(t_idx)
            if d_idx is not None and 0 <= d_idx < len(detections):
                det = detections[d_idx]
                kp = det.metadata.get("keypoints")
                if kp is not None:
                    keypoints = np.array(kp)

            # Fallback: check track metadata
            if keypoints is None:
                kp = track.metadata.get("last_keypoints")
                if kp is not None:
                    keypoints = np.array(kp)

            if keypoints is None:
                continue

            result = self.estimate_with_smoothing(track.track_id, keypoints)
            if result.direction != "UNKNOWN":
                results[track.track_id] = result

        return results

    def visualize(
        self,
        frame: np.ndarray,
        track_orientations: dict[int, OrientationResult],
        tracks: list,
    ) -> np.ndarray:
        """
        Draw orientation labels and debug scores on the frame.

        Parameters
        ----------
        frame
            Frame to annotate.
        track_orientations
            Dict mapping track_id -> OrientationResult.
        tracks
            List of StudentTrack objects (for bbox positions).

        Returns
        -------
        Annotated frame.
        """
        from app.ai.tracker.models import TrackState

        # Color mapping for orientation
        colors = {
            "FORWARD": (0, 255, 0),    # Green
            "LEFT": (0, 165, 255),      # Orange
            "RIGHT": (255, 165, 0),     # Blue-ish
            "DOWN": (0, 0, 255),        # Red
            "UNKNOWN": (128, 128, 128),  # Gray
        }

        for track in tracks:
            if track.state != TrackState.CONFIRMED:
                continue

            orient = track_orientations.get(track.track_id)
            if orient is None:
                continue

            bbox = track.motion.bbox
            color = colors.get(orient.direction, (255, 255, 255))

            # Direction label below the bbox
            label = f"Orient: {orient.direction}"
            cv2.putText(
                frame,
                label,
                (bbox.x1, bbox.y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            # Debug scores
            debug = f"B:{orient.body_score:.2f} E:{orient.eye_score:.2f} A:{orient.ear_score:.2f}"
            cv2.putText(
                frame,
                debug,
                (bbox.x1, bbox.y2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )

        return frame

    def clear_track(self, track_id: int) -> None:
        """Remove history for a track (e.g., on REMOVED state)."""
        self._history.pop(track_id, None)

    def clear_all(self) -> None:
        """Clear all track histories."""
        self._history.clear()

    def get_info(self) -> dict:
        return {
            "model_name": "yolo_keypoint_orientation_fusion",
            "method": "weighted body/eye/ear scores + majority vote",
            "body_weight": self._body_weight,
            "eye_weight": self._eye_weight,
            "ear_weight": self._ear_weight,
            "lr_threshold": self._left_right_threshold,
            "down_threshold": self._down_threshold,
            "history_size": self._history_size,
        }

    # ------------------------------------------------------------------
    # Private scoring functions
    # ------------------------------------------------------------------

    def _get_kp(
        self,
        kp: np.ndarray,
        idx: int,
    ) -> Optional[Tuple[float, float]]:
        """Get keypoint (x, y) if confidence >= threshold, else None."""
        if idx >= kp.shape[0]:
            return None
        x, y, conf = float(kp[idx, 0]), float(kp[idx, 1]), float(kp[idx, 2])
        if conf < self._min_confidence or (x == 0 and y == 0):
            return None
        return (x, y)

    def _body_score(
        self,
        nose: Tuple[float, float],
        left_shoulder: Tuple[float, float],
        right_shoulder: Tuple[float, float],
    ) -> Optional[float]:
        """
        Nose offset from shoulder center, normalized by shoulder width.

        Negative = LEFT, Positive = RIGHT.
        """
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])

        if shoulder_width < 5.0:
            return None

        offset = (nose[0] - shoulder_center_x) / shoulder_width
        return float(np.clip(offset, -1.0, 1.0))

    def _eye_score(
        self,
        nose: Tuple[float, float],
        left_eye: Optional[Tuple[float, float]],
        right_eye: Optional[Tuple[float, float]],
    ) -> Optional[float]:
        """
        Asymmetry of nose-to-left-eye vs nose-to-right-eye distances.

        When turning LEFT, right eye is closer to nose → score negative.
        When turning RIGHT, left eye is closer → score positive.
        """
        if left_eye is None or right_eye is None:
            return None

        dist_left = math.hypot(
            nose[0] - left_eye[0], nose[1] - left_eye[1]
        )
        dist_right = math.hypot(
            nose[0] - right_eye[0], nose[1] - right_eye[1]
        )

        total = dist_left + dist_right
        if total < 1.0:
            return None

        # Normalized asymmetry: (right - left) / total
        # LEFT turn -> right closer, dist_right < dist_left -> negative
        asym = (dist_right - dist_left) / total
        return float(np.clip(asym * 2.0, -1.0, 1.0))

    def _ear_score(
        self,
        nose: Tuple[float, float],
        left_ear: Optional[Tuple[float, float]],
        right_ear: Optional[Tuple[float, float]],
    ) -> Optional[float]:
        """
        Asymmetry of nose-to-ear distances with single-ear fallback.

        Both ears: (right_dist - left_dist) / total
        Single ear: strong indicator of that direction.
        """
        if left_ear is None and right_ear is None:
            return None

        # Single-ear fallback: if only one ear visible, strong
        # indicator that the person is turned toward that side
        if left_ear is not None and right_ear is None:
            # Only left ear visible → facing RIGHT (left ear exposed)
            return 0.5

        if right_ear is not None and left_ear is None:
            # Only right ear visible → facing LEFT (right ear exposed)
            return -0.5

        # Both ears present
        dist_left = math.hypot(
            nose[0] - left_ear[0], nose[1] - left_ear[1]
        )
        dist_right = math.hypot(
            nose[0] - right_ear[0], nose[1] - right_ear[1]
        )

        total = dist_left + dist_right
        if total < 1.0:
            return None

        asym = (dist_right - dist_left) / total
        return float(np.clip(asym * 2.0, -1.0, 1.0))

    def _check_down(
        self,
        nose: Tuple[float, float],
        left_eye: Optional[Tuple[float, float]],
        right_eye: Optional[Tuple[float, float]],
        left_shoulder: Tuple[float, float],
        right_shoulder: Tuple[float, float],
    ) -> bool:
        """
        Detect if the person is looking down.

        Checks if the nose is significantly below the eye center,
        normalized by shoulder width.
        """
        if left_eye is None or right_eye is None:
            return False

        eye_center_y = (left_eye[1] + right_eye[1]) / 2.0
        nose_below = nose[1] - eye_center_y  # positive = below

        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        if shoulder_width < 5.0:
            return False

        ratio = nose_below / shoulder_width
        return ratio > self._down_threshold
