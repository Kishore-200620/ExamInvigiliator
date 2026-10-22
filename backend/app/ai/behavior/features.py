"""
Derived movement and body behavior features - Phase 6 Step 4-6
Calculates from real observations, never fabricates.
"""
from typing import Optional, Tuple
import math
import time

from app.ai.behavior.schemas import BodyPoseObservation
from app.ai.tracker.models import Track
from app.ai.detector.models import BoundingBox


def _get_keypoint(body_obs: BodyPoseObservation, name: str) -> Optional[tuple[float, float, float]]:
    """Get keypoint by name, return None if unavailable or low confidence."""
    if not body_obs or not body_obs.available or not body_obs.keypoints:
        return None
    kp = body_obs.keypoints.get(name)
    if kp is None:
        return None
    x, y, conf = kp
    # Check confidence threshold
    if conf is not None and conf < 0.3:
        return None
    return kp


def calculate_velocity(
    prev_point: Optional[tuple[float, float]],
    curr_point: Optional[tuple[float, float]],
    dt: float,
) -> Optional[float]:
    """Calculate velocity = distance / dt, return None if invalid."""
    if prev_point is None or curr_point is None:
        return None
    if dt is None or dt <= 0 or dt != dt:  # NaN
        return None
    try:
        dx = curr_point[0] - prev_point[0]
        dy = curr_point[1] - prev_point[1]
        dist = math.sqrt(dx * dx + dy * dy)
        return dist / dt
    except Exception:
        return None


def person_velocity(track: Track, prev_track: Track = None, dt: float = None) -> Optional[float]:
    """Person movement from StudentTrack motion (Kalman) or bbox center."""
    if track is None or track.motion is None or track.motion.bbox is None:
        return None
    if prev_track is None or prev_track.motion.bbox is None:
        return None
    if dt is None or dt <= 0:
        return None
    # Use bbox centers
    curr_center = track.motion.bbox.center
    prev_center = prev_track.motion.bbox.center
    return calculate_velocity(prev_center, curr_center, dt)


def wrist_velocity(
    curr_body: BodyPoseObservation,
    prev_body: BodyPoseObservation,
    dt: float,
    side: str = "left",
) -> Optional[float]:
    """Wrist velocity from body pose."""
    if side == "left":
        curr = _get_keypoint(curr_body, "left_wrist")
        prev = _get_keypoint(prev_body, "left_wrist")
    else:
        curr = _get_keypoint(curr_body, "right_wrist")
        prev = _get_keypoint(prev_body, "right_wrist")
    if curr is None or prev is None:
        return None
    # curr and prev are (x,y,conf)
    return calculate_velocity((prev[0], prev[1]), (curr[0], curr[1]), dt)


def body_movement_magnitude(
    curr_body: BodyPoseObservation,
    prev_body: BodyPoseObservation,
    dt: float,
) -> Optional[float]:
    """Overall body movement as average of shoulder and hip movement."""
    if not curr_body or not prev_body or not curr_body.available or not prev_body.available:
        return None
    # Use shoulders and hips
    points = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    velocities = []
    for name in points:
        curr = _get_keypoint(curr_body, name)
        prev = _get_keypoint(prev_body, name)
        v = calculate_velocity(
            (prev[0], prev[1]) if prev else None,
            (curr[0], curr[1]) if curr else None,
            dt,
        ) if curr and prev else None
        if v is not None:
            velocities.append(v)
    if not velocities:
        return None
    return sum(velocities) / len(velocities)


def wrist_vertical_movement(
    curr_body: BodyPoseObservation, prev_body: BodyPoseObservation, dt: float, side: str = "left"
) -> Optional[float]:
    """Vertical wrist movement (y difference) - for hand below desk detection."""
    curr = _get_keypoint(curr_body, f"{side}_wrist")
    prev = _get_keypoint(prev_body, f"{side}_wrist")
    if curr is None or prev is None or dt is None or dt <= 0:
        return None
    # Vertical movement: positive is down
    return (curr[1] - prev[1]) / dt


def wrist_horizontal_movement(
    curr_body: BodyPoseObservation, prev_body: BodyPoseObservation, dt: float, side: str = "left"
) -> Optional[float]:
    """Horizontal wrist movement."""
    curr = _get_keypoint(curr_body, f"{side}_wrist")
    prev = _get_keypoint(prev_body, f"{side}_wrist")
    if curr is None or prev is None or dt is None or dt <= 0:
        return None
    return (curr[0] - prev[0]) / dt


def wrist_to_body_distance(
    wrist: Optional[tuple[float, float, float]], body_bbox: BoundingBox
) -> Optional[float]:
    """Distance from wrist to body bbox center - for hand-to-body relationship."""
    if wrist is None or body_bbox is None:
        return None
    wx, wy = wrist[0], wrist[1]
    cx, cy = body_bbox.center
    return math.sqrt((wx - cx) ** 2 + (wy - cy) ** 2)


def student_to_student_distance(
    bbox1: BoundingBox, bbox2: BoundingBox
) -> Optional[float]:
    """Distance between two students' bbox centers - for proximity."""
    if bbox1 is None or bbox2 is None:
        return None
    c1 = bbox1.center
    c2 = bbox2.center
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def shoulder_orientation(body_obs: BodyPoseObservation) -> Optional[float]:
    """Shoulder orientation angle (degrees) from left to right shoulder."""
    left = _get_keypoint(body_obs, "left_shoulder")
    right = _get_keypoint(body_obs, "right_shoulder")
    if left is None or right is None:
        return None
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle


def arm_extension(body_obs: BodyPoseObservation, side: str = "left") -> Optional[float]:
    """Arm extension: distance from shoulder to wrist vs arm length."""
    shoulder = _get_keypoint(body_obs, f"{side}_shoulder")
    elbow = _get_keypoint(body_obs, f"{side}_elbow")
    wrist = _get_keypoint(body_obs, f"{side}_wrist")
    if shoulder is None or elbow is None or wrist is None:
        return None
    # Compute arm length as shoulder->elbow + elbow->wrist, vs shoulder->wrist
    se_dist = math.sqrt((shoulder[0]-elbow[0])**2 + (shoulder[1]-elbow[1])**2)
    ew_dist = math.sqrt((elbow[0]-wrist[0])**2 + (elbow[1]-wrist[1])**2)
    sw_dist = math.sqrt((shoulder[0]-wrist[0])**2 + (shoulder[1]-wrist[1])**2)
    total = se_dist + ew_dist
    if total == 0:
        return None
    # Extension ratio: 1.0 means fully extended, <1 means bent
    return sw_dist / total
