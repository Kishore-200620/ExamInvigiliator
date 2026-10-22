from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BoundingBox:
    """
    Bounding box in XYXY format.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        return (
            (self.x1 + self.x2) // 2,
            (self.y1 + self.y2) // 2,
        )


@dataclass(slots=True)
class Detection:
    """
    Standard detection object produced by all detectors.
    """

    bbox: BoundingBox

    confidence: float

    class_id: int

    class_name: str

    frame_id: int

    metadata: dict[str, Any] = field(default_factory=dict)