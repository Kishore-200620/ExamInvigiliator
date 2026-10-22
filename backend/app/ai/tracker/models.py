from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from app.ai.detector.models import BoundingBox
from app.ai.tracker.motion import MotionState


class TrackState(Enum):
    """
    Current lifecycle state of a track.
    """

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"
    TENTATIVE_LOST = "tentative_lost"


@dataclass(slots=True)
class HeadPose:
    """
    Head pose signal for Phase 5 - per-track, not Rule Engine decision.
    Yaw: -90 (left) .. 90 (right), Pitch: -90 (down) .. 90 (up), Roll: -90..90
    """

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    confidence: float = 0.0
    frame_id: int = 0
    timestamp: float = 0.0


@dataclass(slots=True)
class TrackStatistics:
    """
    Statistics maintained for every track.
    """

    age: int = 0

    hits: int = 0

    misses: int = 0

    total_visible_frames: int = 0


@dataclass(slots=True)
class Track:
    """
    Represents one tracked student.
    """

    track_id: int

    motion: MotionState

    confidence: float = 0.0

    state: TrackState = TrackState.TENTATIVE

    statistics: TrackStatistics = field(
        default_factory=TrackStatistics
    )

    created_at: datetime = field(
        default_factory=datetime.now
    )

    last_seen: datetime = field(
        default_factory=datetime.now
    )

    feature_history: list[Any] = field(
        default_factory=list
    )

    # Phase 5: head pose / behavior analysis (3D MediaPipe yaw/pitch/roll + keypoint fallback)
    pose_history: list[HeadPose] = field(default_factory=list)
    gaze_history: list[Any] = field(default_factory=list)
    head_bbox_history: list[BoundingBox] = field(default_factory=list)
    last_pose_update_age: int = 0

    # Body orientation estimation (weighted keypoint fusion)
    orientation_history: list[Any] = field(default_factory=list)

    metadata: dict[str, Any] = field(
        default_factory=dict
    )