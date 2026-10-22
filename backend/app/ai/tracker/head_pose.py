"""
Head Pose — Pure YOLO-keypoint geometry (no MediaPipe).

Replaces MediaPipe Face Landmarker + SolvePnP with scale-invariant
nose-offset geometry using existing YOLO26n-pose 17 COCO keypoints.

COCO indices: nose=0, left_eye=1, right_eye=2, left_ear=3, right_ear=4
"""
from typing import Optional, Tuple, Dict, Union, List
import math

import numpy as np


HEAD_OFFSET_THRESHOLD = 0.20  # normalized nose offset (inter-eye/ear) to trigger LEFT/RIGHT
CONF_THRESHOLD = 0.30  # matches wrist containment
YAW_PROXY_SCALE = 60.0  # degrees per 1.0 normalized offset (for display only)


def _get_point(keypoints_dict: Dict[str, Tuple[float, float, float]], name: str):
    v = keypoints_dict.get(name)
    if v is None:
        return None
    try:
        if len(v) >= 3:
            return (float(v[0]), float(v[1]), float(v[2]))
        elif len(v) == 2:
            return (float(v[0]), float(v[1]), 1.0)
    except Exception:
        return None
    return None


def _extract_from_dict(keypoints) -> Dict[str, Optional[Tuple[float, float, float]]]:
    """Extract 5 head points from dict[str -> (x,y,c)]."""
    out = {}
    for k in ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]:
        out[k] = _get_point(keypoints, k)
    return out


def _extract_from_array(arr: np.ndarray, confidences: Optional[np.ndarray] = None) -> Dict[str, Optional[Tuple[float, float, float]]]:
    """
    arr can be:
      - (17,3) full COCO with (x,y,c)
      - (17,2) with confidences separate (17,)
      - (5,3) head-only
      - (5,2) head-only with confidences
    confidences: optional (N,) when arr is (N,2)
    """
    out = {"nose": None, "left_eye": None, "right_eye": None, "left_ear": None, "right_ear": None}
    try:
        a = np.array(arr)
        if confidences is not None:
            c = np.array(confidences).flatten()
            # arr is (N,2), c is (N,)
            if a.ndim == 2 and a.shape[1] >= 2 and c.shape[0] == a.shape[0]:
                if a.shape[0] >= 17:
                    # full COCO
                    out["nose"] = (float(a[0, 0]), float(a[0, 1]), float(c[0]))
                    out["left_eye"] = (float(a[1, 0]), float(a[1, 1]), float(c[1]))
                    out["right_eye"] = (float(a[2, 0]), float(a[2, 1]), float(c[2]))
                    out["left_ear"] = (float(a[3, 0]), float(a[3, 1]), float(c[3]))
                    out["right_ear"] = (float(a[4, 0]), float(a[4, 1]), float(c[4]))
                elif a.shape[0] == 5:
                    out["nose"] = (float(a[0, 0]), float(a[0, 1]), float(c[0]))
                    out["left_eye"] = (float(a[1, 0]), float(a[1, 1]), float(c[1]))
                    out["right_eye"] = (float(a[2, 0]), float(a[2, 1]), float(c[2]))
                    out["left_ear"] = (float(a[3, 0]), float(a[3, 1]), float(c[3]))
                    out["right_ear"] = (float(a[4, 0]), float(a[4, 1]), float(c[4]))
                else:
                    return out
                return out
        # No separate confidences — expect 3 columns
        if a.ndim == 2 and a.shape[1] == 3:
            if a.shape[0] >= 17:
                out["nose"] = (float(a[0, 0]), float(a[0, 1]), float(a[0, 2]))
                out["left_eye"] = (float(a[1, 0]), float(a[1, 1]), float(a[1, 2]))
                out["right_eye"] = (float(a[2, 0]), float(a[2, 1]), float(a[2, 2]))
                out["left_ear"] = (float(a[3, 0]), float(a[3, 1]), float(a[3, 2]))
                out["right_ear"] = (float(a[4, 0]), float(a[4, 1]), float(a[4, 2]))
            elif a.shape[0] == 5:
                out["nose"] = (float(a[0, 0]), float(a[0, 1]), float(a[0, 2]))
                out["left_eye"] = (float(a[1, 0]), float(a[1, 1]), float(a[1, 2]))
                out["right_eye"] = (float(a[2, 0]), float(a[2, 1]), float(a[2, 2]))
                out["left_ear"] = (float(a[3, 0]), float(a[3, 1]), float(a[3, 2]))
                out["right_ear"] = (float(a[4, 0]), float(a[4, 1]), float(a[4, 2]))
        elif a.ndim == 2 and a.shape[1] == 2:
            # No confidence column and no separate confidences -> assume 1.0
            if a.shape[0] >= 17:
                out["nose"] = (float(a[0, 0]), float(a[0, 1]), 1.0)
                out["left_eye"] = (float(a[1, 0]), float(a[1, 1]), 1.0)
                out["right_eye"] = (float(a[2, 0]), float(a[2, 1]), 1.0)
                out["left_ear"] = (float(a[3, 0]), float(a[3, 1]), 1.0)
                out["right_ear"] = (float(a[4, 0]), float(a[4, 1]), 1.0)
            elif a.shape[0] == 5:
                out["nose"] = (float(a[0, 0]), float(a[0, 1]), 1.0)
                out["left_eye"] = (float(a[1, 0]), float(a[1, 1]), 1.0)
                out["right_eye"] = (float(a[2, 0]), float(a[2, 1]), 1.0)
                out["left_ear"] = (float(a[3, 0]), float(a[3, 1]), 1.0)
                out["right_ear"] = (float(a[4, 0]), float(a[4, 1]), 1.0)
    except Exception:
        pass
    return out


def classify_head_direction(
    keypoints: Union[Dict[str, Tuple[float, float, float]], np.ndarray, List],
    confidences: Optional[np.ndarray] = None,
    offset_threshold: float = HEAD_OFFSET_THRESHOLD,
    conf_threshold: float = CONF_THRESHOLD,
) -> Tuple[str, float]:
    """
    Pure YOLO-keypoint head direction classifier.

    Args:
        keypoints: Either
            - dict {"nose": (x,y,c), "left_eye": ..., "right_eye": ..., "left_ear": ..., "right_ear": ...}
            - np.ndarray (17,3) full COCO or (5,3) head-only with (x,y,conf)
            - np.ndarray (N,2) with separate confidences (N,)
            - list of tuples
        confidences: optional (N,) when keypoints is (N,2)
        offset_threshold: normalized offset to trigger LEFT/RIGHT (default 0.20 ~0.15-0.25)
        conf_threshold: min keypoint confidence to be considered valid (default 0.30)

    Returns:
        (direction: CENTER|LEFT|RIGHT|UNKNOWN, confidence: 0.0-0.95)

    Logic:
        a) Nose offset vs eye-mid normalized by inter-eye distance (scale-invariant).
           Falls back to ear-mid / inter-ear if eyes low-confidence.
        b) Ear-confidence asymmetry reinforces direction (far ear drops when turning).
        c) Threshold 0.20 of inter-distance decides LEFT/RIGHT vs CENTER.
        d) UNKNOWN when nose or both eye/ear pairs low-confidence.
    """
    # Resolve offset_threshold from settings if available
    try:
        from app.config.settings import settings
        # New setting takes precedence if present
        if hasattr(settings, "HEAD_OFFSET_THRESHOLD"):
            offset_threshold = float(getattr(settings, "HEAD_OFFSET_THRESHOLD", offset_threshold))
        # Backward: HEAD_YAW_THRESHOLD is degrees, not used here; keep offset_threshold independent
    except Exception:
        pass

    # Extract points
    points: Dict[str, Optional[Tuple[float, float, float]]] = {}
    if isinstance(keypoints, dict):
        points = _extract_from_dict(keypoints)
    elif isinstance(keypoints, (list, np.ndarray)):
        points = _extract_from_array(np.array(keypoints) if not isinstance(keypoints, np.ndarray) else keypoints, confidences)
    else:
        return "UNKNOWN", 0.0

    def is_valid(pt):
        return pt is not None and pt[2] is not None and float(pt[2]) >= conf_threshold

    nose = points.get("nose")
    left_eye = points.get("left_eye")
    right_eye = points.get("right_eye")
    left_ear = points.get("left_ear")
    right_ear = points.get("right_ear")

    if not is_valid(nose):
        return "UNKNOWN", 0.0

    left_eye_valid = is_valid(left_eye)
    right_eye_valid = is_valid(right_eye)
    left_ear_valid = is_valid(left_ear)
    right_ear_valid = is_valid(right_ear)

    use_eyes = left_eye_valid and right_eye_valid
    use_ears = left_ear_valid and right_ear_valid

    inter_dist = None
    mid_x = None
    base_conf = None
    ear_gap = 0.0
    # Compute ear gap regardless of which pair is used (if both ears have some confidence)
    if left_ear is not None and right_ear is not None:
        try:
            ear_gap = abs(float(left_ear[2]) - float(right_ear[2]))
        except Exception:
            ear_gap = 0.0

    if use_eyes:
        lx, ly, lc = left_eye  # type: ignore
        rx, ry, rc = right_eye  # type: ignore
        inter_dist = math.hypot(lx - rx, ly - ry)
        if inter_dist < 1e-6:
            # Degenerate inter-eye distance -> fall back to ears if possible
            if use_ears:
                lx2, ly2, lc2 = left_ear  # type: ignore
                rx2, ry2, rc2 = right_ear  # type: ignore
                inter_dist = math.hypot(lx2 - rx2, ly2 - ry2)
                mid_x = (lx2 + rx2) / 2.0
                base_conf = min(float(nose[2]), float(lc2), float(rc2))
            else:
                return "UNKNOWN", 0.0
        else:
            mid_x = (lx + rx) / 2.0
            base_conf = min(float(nose[2]), float(lc), float(rc))
    elif use_ears:
        lx, ly, lc = left_ear  # type: ignore
        rx, ry, rc = right_ear  # type: ignore
        inter_dist = math.hypot(lx - rx, ly - ry)
        if inter_dist < 1e-6:
            return "UNKNOWN", 0.0
        mid_x = (lx + rx) / 2.0
        base_conf = min(float(nose[2]), float(lc), float(rc))
    else:
        # No valid eye nor ear pair -> UNKNOWN (avoid forcing LEFT/RIGHT from noise)
        return "UNKNOWN", 0.0

    if inter_dist is None or inter_dist < 1e-6 or mid_x is None or base_conf is None:
        return "UNKNOWN", 0.0

    nose_x = float(nose[0])  # type: ignore
    offset = nose_x - float(mid_x)
    normalized = offset / float(inter_dist)

    # Classification
    if normalized < -offset_threshold:
        direction = "LEFT"
    elif normalized > offset_threshold:
        direction = "RIGHT"
    else:
        direction = "CENTER"

    abs_norm = abs(normalized)

    # Confidence computation (compatible with RuleEngine min_confidence 0.30)
    if direction == "CENTER":
        # Center: high when offset small, decays toward threshold
        # At 0.0 offset: strength 1.0, at thresh: 0.70
        strength = 1.0 - (abs_norm / offset_threshold) * 0.30
        strength = max(0.60, min(1.0, strength))
        confidence = float(base_conf) * strength
        # Symmetric ears reinforce center
        if ear_gap < 0.15:
            confidence = min(0.95, confidence * 1.05)
        # Clamp to valid range; for CENTER we allow up to 0.90
        confidence = float(max(0.0, min(0.95, confidence)))
        # If confidence still below threshold due to low base, keep as is (RuleEngine will ignore)
    else:
        # LEFT / RIGHT: margin beyond threshold increases confidence
        margin = abs_norm - offset_threshold
        # margin 0 -> 0.65, margin 0.20 -> 1.0
        strength = 0.65 + min(0.35, max(0.0, margin) * 1.75)
        confidence = float(base_conf) * strength
        # Ear asymmetry reinforcement
        if ear_gap > 0.25 and left_ear is not None and right_ear is not None:
            left_c = float(left_ear[2])
            right_c = float(right_ear[2])
            # For LEFT (nose left of mid), expect left ear less confident (occluded) -> right > left
            # For RIGHT, left > right
            if direction == "LEFT" and (right_c - left_c) > 0.20:
                confidence = min(0.95, confidence + 0.08)
            elif direction == "RIGHT" and (left_c - right_c) > 0.20:
                confidence = min(0.95, confidence + 0.08)
            else:
                # Gap exists but opposite alignment -> slight penalty (maybe noisy detection)
                confidence = max(0.30, confidence - 0.05)
        confidence = float(max(0.0, min(0.95, confidence)))

    # Ensure at least conf_threshold for valid direction, else downgrade to UNKNOWN?
    # We return direction with low confidence so RuleEngine filters via min_confidence=0.30;
    # but we don't force UNKNOWN just because confidence low — the direction is still best guess.
    # Only UNKNOWN is for missing keypoints.
    return direction, float(confidence)


def estimate_yaw_proxy(normalized_offset: float) -> float:
    """Map normalized offset to yaw proxy degrees for display (not used by RuleEngine)."""
    # Clip to [-0.6, 0.6] -> [-36, 36] deg with scale 60
    yaw = float(np.clip(normalized_offset * YAW_PROXY_SCALE, -60.0, 60.0))
    return yaw


class HeadPoseEstimator:
    """
    Lightweight YOLO-keypoint head pose estimator — no model loading.

    Provides backward-compatible API:
      - classify_direction(yaw) for yaw->direction (legacy)
      - classify_head_direction(keypoints, confidences) for new geometry
      - estimate_detailed(frame, bbox) shim returns UNKNOWN (advise to use keypoints)
    """

    def __init__(self, device: str = "cpu", input_size: Tuple[int, int] = (128, 128)):
        self.device = device
        self.input_h, self.input_w = input_size
        self.model = None
        self.model_name = "yolo_keypoint_geometry (no MediaPipe, cheap per-frame math)"

    @staticmethod
    def classify_direction(yaw: Optional[float], threshold: float = 20.0) -> str:
        """Legacy yaw -> direction (kept for backward compatibility)."""
        if yaw is None:
            return "UNKNOWN"
        try:
            from app.config.settings import settings
            thresh = getattr(settings, "HEAD_YAW_THRESHOLD", threshold)
        except Exception:
            thresh = threshold
        if yaw < -thresh:
            return "LEFT"
        elif yaw > thresh:
            return "RIGHT"
        return "CENTER"

    @staticmethod
    def classify_head_direction(
        keypoints: Union[Dict[str, Tuple[float, float, float]], np.ndarray, List],
        confidences: Optional[np.ndarray] = None,
        offset_threshold: float = HEAD_OFFSET_THRESHOLD,
        conf_threshold: float = CONF_THRESHOLD,
    ) -> Tuple[str, float]:
        """Wrapper for free function."""
        return classify_head_direction(keypoints, confidences, offset_threshold, conf_threshold)

    def estimate_from_keypoints(
        self,
        keypoints: Union[Dict[str, Tuple[float, float, float]], np.ndarray],
        confidences: Optional[np.ndarray] = None,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> Tuple[float, float, float, str, float, bool]:
        """
        Estimate from YOLO keypoints (no image needed).

        Returns (yaw_proxy, pitch, roll, direction, confidence, detected)
        pitch/roll always 0 (not estimated from keypoints).
        """
        direction, confidence = classify_head_direction(keypoints, confidences)
        detected = direction != "UNKNOWN" and confidence >= CONF_THRESHOLD
        if not detected:
            return (0.0, 0.0, 0.0, "UNKNOWN", 0.0, False)
        # Compute yaw proxy for display from normalized offset
        # Re-compute normalized to derive yaw
        try:
            # Re-extract to compute normalized
            if isinstance(keypoints, dict):
                pts = _extract_from_dict(keypoints)  # type: ignore
            else:
                pts = _extract_from_array(np.array(keypoints) if not isinstance(keypoints, np.ndarray) else keypoints, confidences)
            nose = pts.get("nose")
            left_eye = pts.get("left_eye")
            right_eye = pts.get("right_eye")
            left_ear = pts.get("left_ear")
            right_ear = pts.get("right_ear")
            use_eyes = nose and left_eye and right_eye and left_eye[2] >= CONF_THRESHOLD and right_eye[2] >= CONF_THRESHOLD
            if use_eyes:
                inter = math.hypot(left_eye[0] - right_eye[0], left_eye[1] - right_eye[1]) or 1.0
                mid = (left_eye[0] + right_eye[0]) / 2.0
                norm = (nose[0] - mid) / inter
            elif left_ear and right_ear and left_ear[2] >= CONF_THRESHOLD and right_ear[2] >= CONF_THRESHOLD:
                inter = math.hypot(left_ear[0] - right_ear[0], left_ear[1] - right_ear[1]) or 1.0
                mid = (left_ear[0] + right_ear[0]) / 2.0
                norm = (nose[0] - mid) / inter
            else:
                norm = 0.0
            yaw = estimate_yaw_proxy(norm)
        except Exception:
            yaw = 0.0
        return (float(yaw), 0.0, 0.0, direction, float(confidence), True)

    def estimate_detailed(self, frame=None, person_bbox=None) -> Tuple[float, float, float, str, float, bool]:
        """
        Shim: Previously MediaPipe+SolvePnP. Now deprecated — requires YOLO keypoints.
        Returns UNKNOWN. Callers should use estimate_from_keypoints with detection keypoints.
        """
        return (0.0, 0.0, 0.0, "UNKNOWN", 0.0, False)

    def estimate(self, frame=None, person_bbox=None) -> Tuple[float, float, float]:
        yaw, pitch, roll, _, _, _ = self.estimate_detailed(frame, person_bbox)
        return (yaw, pitch, roll)

    def latency_ms(self, *args, **kwargs) -> float:
        # Geometry is ~0.02ms, not measurable as model
        return 0.02

    def get_info(self):
        return {"model_name": self.model_name, "method": "yolo_keypoint_geometry", "offset_threshold": HEAD_OFFSET_THRESHOLD, "conf_threshold": CONF_THRESHOLD}
