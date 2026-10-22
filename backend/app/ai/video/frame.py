from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class Frame:
    """
    Standard frame object passed through the AI pipeline.
    """

    image: np.ndarray

    frame_id: int

    timestamp: float

    fps: float

    source_id: str

    metadata: dict[str, Any] = field(default_factory=dict)