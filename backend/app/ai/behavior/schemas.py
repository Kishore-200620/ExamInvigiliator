"""
Typed behavioral observation schemas - Phase 5 Foundation
Follows existing project pattern: dataclasses (like tracker/models.py), not arbitrary dicts.
Each sub-observation explicitly represents availability; unknown/unavailable is None, not fake.
Distinguishes model confidence vs geometric/quality score.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Any

from app.ai.detector.models import BoundingBox


@dataclass(slots=True)
class FaceObservation:
    """Face detection/landmark observation per student."""

    detected: bool = False
    bbox: Optional[BoundingBox] = None  # face bbox if available (upper 40% of person)
    landmarks_available: bool = False
    landmarks: Optional[List[tuple[float, float]]] = None  # normalized x,y
    landmark_count: int = 0
    # Distinguish model confidence vs heuristic quality
    confidence: Optional[float] = None  # actual MediaPipe/model confidence if exposed, else None
    quality: Optional[float] = None  # geometric quality (e.g., face size, sharpness) 0-1
    frame_id: int = 0
    timestamp: float = 0.0


@dataclass(slots=True)
class EyesObservation:
    """Eye/iris observation per student."""

    available: bool = False
    left_eye_landmarks: Optional[List[tuple[float, float]]] = None
    right_eye_landmarks: Optional[List[tuple[float, float]]] = None
    iris_landmarks: Optional[List[tuple[float, float]]] = None
    left_eye_openness: Optional[float] = None  # EAR 0-1 if reliable
    right_eye_openness: Optional[float] = None
    eye_openness_available: bool = False
    blink_detected: Optional[bool] = None  # only if temporal history available
    frame_id: int = 0
    timestamp: float = 0.0
    # Quality vs confidence
    confidence: Optional[float] = None
    quality: Optional[float] = None


@dataclass(slots=True)
class GazeObservation:
    """Eye gaze estimation per student - HEAD POSE IS NOT EYE GAZE."""

    available: bool = False  # False until real gaze estimator validated
    horizontal_gaze: Optional[float] = None  # -1 left, 0 center, 1 right
    vertical_gaze: Optional[float] = None  # -1 up, 1 down
    gaze_direction: Optional[str] = None  # e.g., "left", "center", "right"
    gaze_target: Optional[str] = None  # e.g., "screen", "desk", "neighbor"
    confidence: Optional[float] = None
    frame_id: int = 0
    timestamp: float = 0.0
    reason: Optional[str] = None  # why unavailable, e.g., "iris landmarks not available"


@dataclass(slots=True)
class HeadPoseObservation:
    """Head pose per student - yaw/pitch/roll, NOT gaze."""

    available: bool = False
    yaw: Optional[float] = None  # -90 to 90, left-right
    pitch: Optional[float] = None  # -90 to 90, up-down
    roll: Optional[float] = None  # -90 to 90
    direction: Optional[str] = "UNKNOWN"  # CENTER, LEFT, RIGHT, UNKNOWN
    confidence: Optional[float] = None  # model confidence if available, else None
    quality: Optional[float] = None  # geometric quality
    frame_id: int = 0
    timestamp: float = 0.0
    # For debugging
    head_bbox: Optional[BoundingBox] = None


def check_wrist_containment(
    wrist_kpt: Optional[tuple[float, float, float]],
    person_bbox: Optional[BoundingBox],
    conf_threshold: float = 0.30,
) -> str:
    """
    Determine if wrist is INSIDE, OUTSIDE, or UNKNOWN relative to person bounding box.
    Returns UNKNOWN if wrist is missing or confidence is below threshold.
    """
    if wrist_kpt is None or person_bbox is None:
        return "UNKNOWN"
    x, y, conf = wrist_kpt
    if conf is None or conf < conf_threshold:
        return "UNKNOWN"
    if hasattr(person_bbox, "x1"):
        x1, y1, x2, y2 = float(person_bbox.x1), float(person_bbox.y1), float(person_bbox.x2), float(person_bbox.y2)
    elif isinstance(person_bbox, (list, tuple)) and len(person_bbox) == 4:
        x1, y1, x2, y2 = float(person_bbox[0]), float(person_bbox[1]), float(person_bbox[2]), float(person_bbox[3])
    else:
        return "UNKNOWN"
    if x1 <= x <= x2 and y1 <= y <= y2:
        return "INSIDE"
    return "OUTSIDE"



@dataclass(slots=True)
class BodyPoseObservation:
    """Body pose per student - 17 COCO or 33 BlazePose keypoints."""

    available: bool = False
    # Keypoints: dict name -> (x, y, confidence) in image coordinates
    keypoints: Optional[dict[str, tuple[float, float, float]]] = None
    # Specific joints for exam behavior (COCO 17)
    nose: Optional[tuple[float, float, float]] = None
    left_eye: Optional[tuple[float, float, float]] = None
    right_eye: Optional[tuple[float, float, float]] = None
    left_ear: Optional[tuple[float, float, float]] = None
    right_ear: Optional[tuple[float, float, float]] = None
    left_shoulder: Optional[tuple[float, float, float]] = None
    right_shoulder: Optional[tuple[float, float, float]] = None
    left_elbow: Optional[tuple[float, float, float]] = None
    right_elbow: Optional[tuple[float, float, float]] = None
    left_wrist: Optional[tuple[float, float, float]] = None
    right_wrist: Optional[tuple[float, float, float]] = None
    left_hip: Optional[tuple[float, float, float]] = None
    right_hip: Optional[tuple[float, float, float]] = None
    left_knee: Optional[tuple[float, float, float]] = None
    right_knee: Optional[tuple[float, float, float]] = None
    left_ankle: Optional[tuple[float, float, float]] = None
    right_ankle: Optional[tuple[float, float, float]] = None
    frame_id: int = 0
    timestamp: float = 0.0
    confidence: Optional[float] = None
    quality: Optional[float] = None


@dataclass(slots=True)
class HandsObservation:
    """Hand/wrist per student."""

    available: bool = False
    left_wrist: Optional[tuple[float, float, float]] = None  # from body pose if available
    right_wrist: Optional[tuple[float, float, float]] = None
    left_hand_landmarks: Optional[List[tuple[float, float]]] = None  # 21 MediaPipe Hands
    right_hand_landmarks: Optional[List[tuple[float, float]]] = None
    left_hand_bbox: Optional[BoundingBox] = None
    right_hand_bbox: Optional[BoundingBox] = None
    left_hand_movement: Optional[float] = None
    right_hand_movement: Optional[float] = None
    frame_id: int = 0
    timestamp: float = 0.0
    confidence: Optional[float] = None


@dataclass(slots=True)
class ObjectsObservation:
    """Prohibited objects per student ROI."""

    available: bool = False
    # List of detected objects near this student
    objects: List[dict[str, Any]] = field(default_factory=list)  # each {type, bbox, confidence}
    phone_detected: bool = False
    laptop_detected: bool = False
    tablet_detected: bool = False
    book_detected: bool = False
    paper_detected: bool = False
    calculator_detected: bool = False
    earphones_detected: bool = False
    frame_id: int = 0
    timestamp: float = 0.0


@dataclass(slots=True)
class MovementObservation:
    """Movement per student - derived from tracker, not new model."""

    available: bool = False
    head_movement: Optional[float] = None
    shoulder_movement: Optional[float] = None
    wrist_movement: Optional[float] = None
    body_movement: Optional[float] = None  # center displacement
    velocity_x: Optional[float] = None
    velocity_y: Optional[float] = None
    direction: Optional[float] = None  # atan2
    stationary_duration: Optional[float] = None
    repeated_movement: Optional[bool] = None
    frame_id: int = 0
    timestamp: float = 0.0


@dataclass(slots=True)
class BehaviorObservation:
    """Per-student aggregated observation - data aggregation only, NOT malpractice decision."""

    student_id: int  # track_id
    frame_id: int
    timestamp: float

    face: FaceObservation = field(default_factory=FaceObservation)
    eyes: EyesObservation = field(default_factory=EyesObservation)
    gaze: GazeObservation = field(default_factory=GazeObservation)
    head_pose: HeadPoseObservation = field(default_factory=HeadPoseObservation)
    body_pose: BodyPoseObservation = field(default_factory=BodyPoseObservation)
    hands: HandsObservation = field(default_factory=HandsObservation)
    objects: ObjectsObservation = field(default_factory=ObjectsObservation)
    movement: MovementObservation = field(default_factory=MovementObservation)

    # Overall
    confidence: Optional[float] = None
    # Evidence that this observation is not a fabrication
    is_fabricated: bool = False
