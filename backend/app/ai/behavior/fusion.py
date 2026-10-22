"""
Behavior fusion - Phase 5 Foundation
Data aggregation only, NOT malpractice decision.
Rule Engine (Phase 6) will decide cheating.
"""
from typing import List

from dataclasses import dataclass, field
from typing import Optional

from app.ai.behavior.schemas import (
    BehaviorObservation,
    FaceObservation,
    EyesObservation,
    GazeObservation,
    HeadPoseObservation,
    BodyPoseObservation,
    HandsObservation,
    ObjectsObservation,
    MovementObservation,
)
from app.ai.detector.models import BoundingBox


@dataclass(slots=True)
class StudentBehaviorState:
    """Normalized per-student state - INPUT to Rule Engine, NOT cheating classification."""

    student_id: int
    timestamp: float
    frame_id: int

    face_present: bool = False
    eye_state: str = "unknown"  # open, closed, unknown
    gaze_state: str = "unknown"
    body_state: str = "unknown"
    movement_state: str = "unknown"
    hands_state: str = "unknown"
    objects_state: str = "unknown"

    head_direction: str = "UNKNOWN"  # CENTER, LEFT, RIGHT, UNKNOWN (from MediaPipe 3D yaw when available, else keypoint geometry)
    left_wrist_status: str = "UNKNOWN"  # INSIDE, OUTSIDE, UNKNOWN
    right_wrist_status: str = "UNKNOWN"  # INSIDE, OUTSIDE, UNKNOWN
    track_bbox: Optional[BoundingBox] = None

    # Body orientation (weighted keypoint fusion: FORWARD/LEFT/RIGHT/DOWN/UNKNOWN)
    orientation: str = "UNKNOWN"
    orientation_confidence: Optional[float] = None
    orientation_score: Optional[float] = None
    orientation_body_score: Optional[float] = None
    orientation_eye_score: Optional[float] = None
    orientation_ear_score: Optional[float] = None
    orientation_is_down: Optional[bool] = None

    # Derived features (from movement.py, body.py, eyes.py)
    wrist_velocity: Optional[float] = None
    elbow_velocity: Optional[float] = None
    shoulder_velocity: Optional[float] = None
    body_movement: Optional[float] = None
    wrist_movement: Optional[float] = None
    proximity_to_other_student: Optional[float] = None
    prolonged_eye_closure: Optional[bool] = None
    eye_openness_trend: Optional[str] = None

    # Raw observations for debugging
    face: FaceObservation = None
    eyes: EyesObservation = None
    gaze: GazeObservation = None
    head_pose: HeadPoseObservation = None
    body_pose: BodyPoseObservation = None
    hands: HandsObservation = None
    objects: ObjectsObservation = None
    movement: MovementObservation = None


class BehaviorFusion:
    """
    Combine behavioral observations per student - data aggregation only.

    Input: face, eyes, gaze, body, hands, objects, movement, head_pose
    Output: BehaviorObservation (aggregated)

    At this stage, fusion = aggregation, NOT "malpractice = true".
    """

    def __init__(self):
        self.model_name = "aggregation_only"

    def fuse(
        self,
        student_id: int,
        frame_id: int,
        timestamp: float,
        face: FaceObservation = None,
        eyes: EyesObservation = None,
        gaze: GazeObservation = None,
        head_pose: HeadPoseObservation = None,
        body_pose: BodyPoseObservation = None,
        hands: HandsObservation = None,
        objects: ObjectsObservation = None,
        movement: MovementObservation = None,
        track_bbox: BoundingBox = None,
        confidence: float = None,
    ) -> BehaviorObservation:
        """
        Aggregate per-student observations into single BehaviorObservation.

        No malpractice logic here - just data collection.
        """
        # Create with defaults if None
        obs = BehaviorObservation(
            student_id=student_id,
            frame_id=frame_id,
            timestamp=timestamp,
            face=face if face is not None else FaceObservation(frame_id=frame_id, timestamp=timestamp),
            eyes=eyes if eyes is not None else EyesObservation(frame_id=frame_id, timestamp=timestamp),
            gaze=gaze if gaze is not None else GazeObservation(frame_id=frame_id, timestamp=timestamp),
            head_pose=head_pose if head_pose is not None else HeadPoseObservation(frame_id=frame_id, timestamp=timestamp),
            body_pose=body_pose if body_pose is not None else BodyPoseObservation(frame_id=frame_id, timestamp=timestamp),
            hands=hands if hands is not None else HandsObservation(frame_id=frame_id, timestamp=timestamp),
            objects=objects if objects is not None else ObjectsObservation(frame_id=frame_id, timestamp=timestamp),
            movement=movement if movement is not None else MovementObservation(frame_id=frame_id, timestamp=timestamp),
            confidence=confidence,
            is_fabricated=False,
        )

        # Validate: ensure no fabricated malpractice field
        # BehaviorObservation has no "is_cheating" or "malpractice" field by design
        assert not hasattr(obs, "is_cheating"), "Fusion must not produce cheating decision"
        assert not hasattr(obs, "malpractice"), "Fusion must not produce malpractice"

        return obs

    def create_behavior_state(
        self,
        student_id: int,
        frame_id: int,
        timestamp: float,
        behavior_obs: BehaviorObservation,
        movement_features: dict = None,
        eye_features: dict = None,
        track_bbox: BoundingBox = None,
        orientation_result: dict | None = None,
    ) -> StudentBehaviorState:
        """
        Create normalized per-student state for Rule Engine.
        This is INPUT to Rule Engine, NOT cheating classification.

        orientation_result: optional dict from OrientationEstimator (direction, confidence, score, ...)
        """
        movement_features = movement_features or {}
        eye_features = eye_features or {}
        orientation_result = orientation_result or {}
        # Also allow orientation via movement_features for backward compat
        if not orientation_result and movement_features.get("orientation"):
            orientation_result = movement_features.get("orientation") if isinstance(movement_features.get("orientation"), dict) else {}

        # Derive per-student states from observations
        face_present = behavior_obs.face.detected if behavior_obs.face else False
        eye_state = "unknown"
        if behavior_obs.eyes and behavior_obs.eyes.available:
            if behavior_obs.eyes.left_eye_openness is not None:
                # Simple threshold for open/closed
                avg_openness = (behavior_obs.eyes.left_eye_openness + (behavior_obs.eyes.right_eye_openness or 0)) / 2
                eye_state = "open" if avg_openness > 0.2 else "closed"
            else:
                eye_state = "unknown"

        # Check for prolonged eye closure from temporal features
        prolonged = eye_features.get("prolonged_eye_closure")
        trend = eye_features.get("eye_openness_trend")

        gaze_state = "unknown"  # gaze is currently unavailable
        if behavior_obs.gaze and behavior_obs.gaze.available:
            gaze_state = behavior_obs.gaze.gaze_direction or "unknown"

        body_state = "unknown"
        if behavior_obs.body_pose and behavior_obs.body_pose.available:
            body_state = "visible"

        # Movement state from features
        movement_state = "unknown"
        body_movement = movement_features.get("body_movement")
        if body_movement is not None:
            if body_movement > 50:  # threshold for excessive movement, configurable later
                movement_state = "moving"
            elif body_movement < 5:
                movement_state = "stationary"
            else:
                movement_state = "normal"

        head_direction = "UNKNOWN"
        if behavior_obs.head_pose and behavior_obs.head_pose.available:
            # Prefer explicit direction if present
            if behavior_obs.head_pose.direction and behavior_obs.head_pose.direction != "UNKNOWN":
                head_direction = behavior_obs.head_pose.direction
            # If direction is UNKNOWN but yaw is available (e.g., MediaPipe 3D), derive via yaw threshold
            elif behavior_obs.head_pose.yaw is not None:
                try:
                    from app.ai.tracker.head_pose import HeadPoseEstimator as _HPE

                    head_direction = _HPE.classify_direction(behavior_obs.head_pose.yaw)
                except Exception:
                    pass
                # Also try MediaPipe 3D classifier if yaw is large but direction still UNKNOWN
                if head_direction == "UNKNOWN" and behavior_obs.head_pose.yaw is not None:
                    try:
                        from app.config.settings import settings as _s2

                        thr = float(getattr(_s2, "HEAD_YAW_THRESHOLD", 20.0))
                        y = float(behavior_obs.head_pose.yaw)
                        if y < -thr:
                            head_direction = "LEFT"
                        elif y > thr:
                            head_direction = "RIGHT"
                        else:
                            head_direction = "CENTER"
                    except Exception:
                        pass

        # Orientation handling (weighted fusion) - may be passed via orientation_result or movement_features
        orientation = "UNKNOWN"
        orientation_confidence = None
        orientation_score = None
        orientation_body_score = None
        orientation_eye_score = None
        orientation_ear_score = None
        orientation_is_down = None
        if orientation_result and isinstance(orientation_result, dict):
            orientation = orientation_result.get("direction", orientation_result.get("orientation", "UNKNOWN"))
            orientation_confidence = orientation_result.get("confidence")
            orientation_score = orientation_result.get("score")
            orientation_body_score = orientation_result.get("body_score")
            orientation_eye_score = orientation_result.get("eye_score")
            orientation_ear_score = orientation_result.get("ear_score")
            orientation_is_down = orientation_result.get("is_down")
        # Fallback: check behavior_obs for orientation-like fields (future)
        if orientation == "UNKNOWN" and hasattr(behavior_obs, "orientation"):
            try:
                orientation = getattr(behavior_obs, "orientation", "UNKNOWN")
            except Exception:
                pass

        from app.ai.behavior.schemas import check_wrist_containment
        from app.config.settings import settings
        conf_thresh = getattr(settings, "WRIST_CONFIDENCE_THRESHOLD", 0.30)
        lw_kpt = behavior_obs.body_pose.left_wrist if behavior_obs.body_pose else None
        rw_kpt = behavior_obs.body_pose.right_wrist if behavior_obs.body_pose else None
        left_wrist_status = check_wrist_containment(lw_kpt, track_bbox, conf_thresh)
        right_wrist_status = check_wrist_containment(rw_kpt, track_bbox, conf_thresh)

        return StudentBehaviorState(
            student_id=student_id,
            timestamp=timestamp,
            frame_id=frame_id,
            face_present=face_present,
            eye_state=eye_state,
            gaze_state=gaze_state,
            body_state=body_state,
            movement_state=movement_state,
            hands_state="unknown" if not behavior_obs.hands or not behavior_obs.hands.available else "visible",
            objects_state="unknown" if not behavior_obs.objects or not behavior_obs.objects.available else "visible",
            head_direction=head_direction,
            left_wrist_status=left_wrist_status,
            right_wrist_status=right_wrist_status,
            track_bbox=track_bbox,
            orientation=orientation,
            orientation_confidence=orientation_confidence,
            orientation_score=orientation_score,
            orientation_body_score=orientation_body_score,
            orientation_eye_score=orientation_eye_score,
            orientation_ear_score=orientation_ear_score,
            orientation_is_down=orientation_is_down,
            wrist_velocity=movement_features.get("wrist_velocity"),
            elbow_velocity=movement_features.get("elbow_velocity"),
            shoulder_velocity=movement_features.get("shoulder_velocity"),
            body_movement=body_movement,
            wrist_movement=movement_features.get("wrist_movement"),
            proximity_to_other_student=movement_features.get("proximity"),
            prolonged_eye_closure=prolonged,
            eye_openness_trend=trend,
            face=behavior_obs.face,
            eyes=behavior_obs.eyes,
            gaze=behavior_obs.gaze,
            head_pose=behavior_obs.head_pose,
            body_pose=behavior_obs.body_pose,
            hands=behavior_obs.hands,
            objects=behavior_obs.objects,
            movement=behavior_obs.movement,
        )


    def fuse_batch(
        self, observations: List[BehaviorObservation]
    ) -> List[BehaviorObservation]:
        """Batch fuse (for future optimization) - currently just passes through."""
        return observations

    def get_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "available": True,
            "note": "Data aggregation only, no malpractice decision - Rule Engine Phase 6 will consume BehaviorObservation",
        }
