"""
Temporal History - Phase 6
Bounded per-student history for face, eyes, gaze, head pose, body pose, hands, objects, movement.
"""
from collections import deque, defaultdict
from typing import Dict, List, Optional, Any
import time


class TemporalHistory:
    """
    Maintain bounded recent observations per student.

    History is per student_id, per signal type, capped by max_size and max_duration.
    Old observations automatically expire.
    """

    def __init__(
        self,
        max_size: int = 10,
        max_duration_seconds: float = 10.0,
        history_configs: dict = None,
    ):
        self.max_size = max_size
        self.max_duration = max_duration_seconds
        # Per-signal configs: {signal: {"max_size": 10, "max_duration": 10.0}}
        self.history_configs = history_configs or {}
        # Structure: {student_id: {signal: deque}}
        self._histories: Dict[int, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        # For generic history
        self._generic_histories: Dict[int, deque] = defaultdict(deque)

    def _get_config(self, signal: str):
        if signal in self.history_configs:
            return self.history_configs[signal]
        return {"max_size": self.max_size, "max_duration": self.max_duration}

    def append(self, student_id: int, signal: str, observation: Any, timestamp: float = None):
        """Append observation for student and signal."""
        # Handle invalid timestamps
        if timestamp is None:
            timestamp = time.time()
        try:
            timestamp = float(timestamp)
            if timestamp < 0 or timestamp != timestamp:  # NaN check
                timestamp = time.time()
        except (TypeError, ValueError):
            timestamp = time.time()
        # Handle duplicate frame IDs: if observation has frame_id and last entry has same frame_id, skip or replace
        try:
            if observation is not None and hasattr(observation, "frame_id"):
                if len(self._histories[student_id][signal]) > 0:
                    last_obs = self._histories[student_id][signal][-1][1]
                    if hasattr(last_obs, "frame_id") and last_obs.frame_id == observation.frame_id:
                        # Duplicate frame_id, skip to avoid duplicates
                        return
            # Also handle out-of-order timestamps: if new timestamp < last timestamp, still append but log
            # For now, allow out-of-order but keep sorted by timestamp for get_since to work
            # We will append and then sort if needed, or just append and let duration check handle
            pass
        except Exception:
            pass
        # Ensure observation has timestamp/frame_id if possible
        config = self._get_config(signal)
        max_size = config["max_size"]
        max_duration = config["max_duration"]

        hist = self._histories[student_id][signal]
        # Attach timestamp if observation is dict-like and has timestamp
        # For dataclass, we can't easily attach, so we store tuple (timestamp, observation)
        # If out-of-order, insert in correct position to keep sorted
        if len(hist) > 0 and timestamp < hist[-1][0]:
            # Out-of-order timestamp, insert sorted
            # Find insertion point
            inserted = False
            for i in range(len(hist)):
                if timestamp < hist[i][0]:
                    hist.insert(i, (timestamp, observation))
                    inserted = True
                    break
            if not inserted:
                hist.append((timestamp, observation))
        else:
            hist.append((timestamp, observation))
        # Enforce size bound
        while len(hist) > max_size:
            hist.popleft()
        # Enforce duration bound
        while len(hist) > 1 and (timestamp - hist[0][0]) > max_duration:
            hist.popleft()

    def get_recent(self, student_id: int, signal: str, n: int = 5) -> List[Any]:
        """Get n most recent observations for student and signal."""
        hist = self._histories[student_id][signal]
        recent = list(hist)[-n:]
        return [obs for _, obs in recent]

    def get_since(self, student_id: int, signal: str, since_timestamp: float) -> List[Any]:
        """Get observations since timestamp."""
        hist = self._histories[student_id][signal]
        return [obs for ts, obs in hist if ts >= since_timestamp]

    def latest(self, student_id: int, signal: str) -> Optional[Any]:
        """Get latest observation."""
        hist = self._histories[student_id][signal]
        if hist:
            return hist[-1][1]
        return None

    def clear_student(self, student_id: int):
        """Clear all history for a student (when track terminated)."""
        if student_id in self._histories:
            del self._histories[student_id]
        if student_id in self._generic_histories:
            del self._generic_histories[student_id]

    def clear_all(self):
        """Clear all history."""
        self._histories.clear()
        self._generic_histories.clear()

    def get_all_students(self) -> List[int]:
        return list(self._histories.keys())

    def get_history_size(self, student_id: int, signal: str) -> int:
        return len(self._histories[student_id][signal])

    # Generic helpers for movement etc. that may not fit signal categories
    def append_generic(self, student_id: int, observation: Any, timestamp: float = None):
        if timestamp is None:
            timestamp = time.time()
        hist = self._generic_histories[student_id]
        hist.append((timestamp, observation))
        while len(hist) > self.max_size:
            hist.popleft()
        while len(hist) > 1 and (timestamp - hist[0][0]) > self.max_duration:
            hist.popleft()

    def get_generic_recent(self, student_id: int, n: int = 5):
        hist = self._generic_histories[student_id]
        return [obs for _, obs in list(hist)[-n:]]
