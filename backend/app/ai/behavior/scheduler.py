"""
Pose Scheduler - Phase 6 Step 2
Configurable scheduler for YOLO26n-Pose full-frame inference.
Baseline: every 60 frames (~2.4s at 25 FPS), not every frame.
Supports event-triggered early inference.
"""
from typing import Dict, Set


class PoseScheduler:
    """
    Scheduler for YOLO26n-Pose full-frame inference.

    YOLO pose is full-frame (not per-student crop), so cost is per-frame not per-student.
    Running every frame would be 40ms + 52ms = 92ms >66ms FAIL 15FPS.
    Running every 60 frames: 40/60=0.66ms avg +52=52.6ms PASS.

    Configurable via POSE_INTERVAL_FRAMES (default 60).
    """

    def __init__(
        self,
        interval_frames: int = 60,
        enabled: bool = True,
    ):
        self.interval_frames = interval_frames
        self.enabled = enabled
        self.last_pose_frame: int = -100000  # far in past
        self.forced_students: Set[int] = set()

    def should_run(self, current_frame: int, num_tracks: int = 0) -> bool:
        """Check if pose should run this frame (periodic)."""
        if not self.enabled:
            return False
        if current_frame - self.last_pose_frame >= self.interval_frames:
            return True
        return False

    def force_update(self, student_id: int):
        """Force pose update for a specific student (event-triggered)."""
        self.forced_students.add(student_id)

    def should_run_for_student(self, student_id: int, current_frame: int) -> bool:
        """Check if this specific student needs pose update (forced)."""
        if student_id in self.forced_students:
            return True
        return False

    def mark_run(self, current_frame: int, forced_ids: Set[int] = None):
        """Mark that pose was run this frame."""
        self.last_pose_frame = current_frame
        if forced_ids:
            for sid in forced_ids:
                self.forced_students.discard(sid)
        else:
            # If it was a forced run, clear all forced
            self.forced_students.clear()

    def get_info(self) -> dict:
        return {
            "interval_frames": self.interval_frames,
            "enabled": self.enabled,
            "last_pose_frame": self.last_pose_frame,
            "pending_forced": len(self.forced_students),
        }
