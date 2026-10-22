"""
Behavioral Perception Layer - Phase 5 Foundation
Each module produces observations, NEVER decides malpractice.
Rule Engine (Phase 6+) consumes BehaviorObservation.
"""
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

# New estimators ported from exam_invigiliator_2 (optional, lazy-loaded)
try:
    from app.ai.behavior.head_pose_3d import HeadPose3DEstimator, HeadPose3DResult
except Exception:
    HeadPose3DEstimator = None  # type: ignore
    HeadPose3DResult = None  # type: ignore

try:
    from app.ai.behavior.orientation import OrientationEstimator, OrientationResult
except Exception:
    OrientationEstimator = None  # type: ignore
    OrientationResult = None  # type: ignore

__all__ = [
    "BehaviorObservation",
    "FaceObservation",
    "EyesObservation",
    "GazeObservation",
    "HeadPoseObservation",
    "BodyPoseObservation",
    "HandsObservation",
    "ObjectsObservation",
    "MovementObservation",
    "HeadPose3DEstimator",
    "HeadPose3DResult",
    "OrientationEstimator",
    "OrientationResult",
]
