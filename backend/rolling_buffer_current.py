"""
Rolling Original Frame Buffer - Phase 7
Thread-safe, time-based rolling buffer for original unmodified frames.
Stores ORIGINAL frames at ingestion (before processing queue), not annotated.
"""
import threading
import copy
import time
from collections import deque
from typing import List, Optional
from dataclasses import dataclass

from app.ai.video.frame import Frame


class RollingOriginalBuffer:
    """
    Thread-safe rolling buffer for original evidence frames.

    - Time-based eviction (retention_seconds), not just count
    - Stores deep copy of image to avoid mutation
    - Never stores annotated/resized frames - caller must pass original
    - Supports concurrent add/get
    """

    def __init__(self, retention_seconds: float = 3.0):
        self.retention_seconds = float(retention_seconds)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._frames: deque[Frame] = deque()
        self._closed = False
        # For quick stats
        self._total_added = 0

    def add(self, frame: Frame) -> None:
        """Add original frame (thread-safe). Copies image to prevent mutation."""
        if frame is None or frame.image is None:
            return
        # Deep copy image to preserve original
        try:
            image_copy = frame.image.copy()
        except Exception:
            image_copy = frame.image
        # Create Frame copy (preserve original fields, but with copied image)
        # Do NOT mutate incoming frame; store independent copy
        stored = Frame(
            image=image_copy,
            frame_id=frame.frame_id,
            timestamp=float(frame.timestamp),
            fps=float(frame.fps),
            source_id=frame.source_id,
            metadata=dict(frame.metadata) if frame.metadata else {},
        )
        with self._lock:
            self._frames.append(stored)
            self._total_added += 1
            # Evict old frames time-based: keep frames where newest.timestamp - oldest.timestamp <= retention
            # Use stored's timestamp as newest
            newest_ts = stored.timestamp
            while len(self._frames) > 1:
                oldest_ts = self._frames[0].timestamp
                if (newest_ts - oldest_ts) > self.retention_seconds:
                    self._frames.popleft()
                else:
                    break
            self._condition.notify_all()

    def get_range(self, start_time: float, end_time: float) -> List[Frame]:
        """
        Get frames where timestamp in [start_time, end_time] inclusive.
        Returns copies list (references to stored frames, but caller should not mutate image).
        Thread-safe.
        """
        with self._lock:
            result = [f for f in self._frames if start_time <= f.timestamp <= end_time]
            return list(result)

    def update_frame_metadata(self, frame_id: int, values: dict) -> bool:
        """Attach analysis results to an already buffered source frame."""
        with self._lock:
            for frame in self._frames:
                if frame.frame_id == frame_id:
                    if frame.metadata is None:
                        frame.metadata = {}
                    frame.metadata.update(values)
                    return True
        return False

    def get_all(self) -> List[Frame]:
        with self._lock:
            return list(self._frames)

    def get_latest_timestamp(self) -> Optional[float]:
        with self._lock:
            if self._frames:
                return self._frames[-1].timestamp
            return None

    def wait_until(self, timestamp: float, timeout: Optional[float] = None) -> bool:
        """Wait for a target media timestamp, or return when capture is complete."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while True:
                latest = self._frames[-1].timestamp if self._frames else None
                if latest is not None and latest >= timestamp:
                    return True
                if self._closed:
                    return False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)

    def mark_complete(self) -> None:
        """Wake evidence writers when no more source frames will arrive."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def get_earliest_timestamp(self) -> Optional[float]:
        with self._lock:
            if self._frames:
                return self._frames[0].timestamp
            return None

    def size(self) -> int:
        with self._lock:
            return len(self._frames)

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def total_added(self) -> int:
        with self._lock:
            return self._total_added

    def contains(self, frame_id: int) -> bool:
        with self._lock:
            return any(f.frame_id == frame_id for f in self._frames)

    def get_info(self) -> dict:
        with self._lock:
            return {
                "retention_seconds": self.retention_seconds,
                "size": len(self._frames),
                "total_added": self._total_added,
                "earliest": self._frames[0].timestamp if self._frames else None,
                "latest": self._frames[-1].timestamp if self._frames else None,
            }
