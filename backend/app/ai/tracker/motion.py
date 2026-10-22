from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.ai.detector.models import BoundingBox
from app.ai.tracker.base import MotionModel


@dataclass(slots=True)
class MotionState:
    """
    Stores all motion-related information for a track.

    This class is independent of the underlying motion model
    (Kalman, Optical Flow, etc.).
    """

    # Current observed bounding box
    bbox: BoundingBox

    # Previous observed bounding box
    previous_bbox: BoundingBox | None = None

    # Predicted bounding box
    predicted_bbox: BoundingBox | None = None

    # Velocity (pixels/frame)
    velocity_x: float = 0.0
    velocity_y: float = 0.0

    # Width / height velocity
    velocity_w: float = 0.0
    velocity_h: float = 0.0

    # Motion model (Kalman Filter)
    motion_model: MotionModel | None = None

    # Last update timestamp
    last_update: datetime = field(
        default_factory=datetime.now
    )