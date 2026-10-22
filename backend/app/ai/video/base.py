from abc import ABC, abstractmethod

import numpy as np


class VideoSource(ABC):
    """
    Abstract interface for all video capture implementations.
    """

    @abstractmethod
    def open(self) -> bool:
        """Open the video source."""
        pass

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read a raw frame."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check whether the source is open."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Release the video source."""
        pass