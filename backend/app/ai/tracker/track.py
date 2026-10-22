from datetime import datetime

from app.ai.detector.models import Detection
from app.ai.tracker.models import Track, TrackState
from app.ai.tracker.motion import MotionState
from app.ai.tracker.kalman import KalmanMotionModel


class StudentTrack(Track):
    """
    Represents one tracked student.
    """

    def __init__(
        self,
        track_id: int,
        detection: Detection,
        min_hits: int = 3,
        lost_threshold: int = 5,
        tentative_max_misses: int = 3,
        tentative_lost_grace: int = 15,
    ):

        motion = MotionState(
            bbox=detection.bbox,
        )

        kalman = KalmanMotionModel()
        kalman.initialize(detection.bbox)

        motion.motion_model = kalman

        super().__init__(
            track_id=track_id,
            motion=motion,
            confidence=detection.confidence,
            state=TrackState.TENTATIVE,
        )

        # Configurable thresholds passed from TrackingManager (avoid hardcoding 3 in two places)
        self.min_hits = min_hits
        self.lost_threshold = lost_threshold
        self.tentative_max_misses = tentative_max_misses
        self.tentative_lost_grace = tentative_lost_grace

    def update(self, detection: Detection):

        self.motion.previous_bbox = self.motion.bbox

        self.motion.bbox = detection.bbox

        self.motion.motion_model.update(
            detection.bbox
        )

        self.confidence = detection.confidence
        self.statistics.hits += 1
        self.statistics.age += 1
        self.statistics.total_visible_frames += 1
        self.statistics.misses = 0

        self.last_seen = datetime.now()

        prev_state_before = self.state
        # State transitions on successful match
        if self.state == TrackState.TENTATIVE and self.statistics.hits >= self.min_hits:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.TENTATIVE_LOST:
            self.state = TrackState.CONFIRMED
        # Phase 5: event-triggered head pose / behavior (not every frame, not every track)
        # On CONFIRMED transition + periodic refresh; uses upper 40% bbox crop
        # Real 6DRepNet weights needed; current stub uses MobileNet feature proxy
        # Stores yaw/pitch/roll + confidence in pose_history (capped 10)
        if self.state == TrackState.CONFIRMED and (prev_state_before == TrackState.TENTATIVE or prev_state_before == TrackState.LOST or prev_state_before == TrackState.TENTATIVE_LOST):
            # Transition to CONFIRMED -> extract pose (stub) for identity tracking
            pass  # manager._update_pose_features handles extraction; keep track logic clean

    def predict(self):

        self.motion.predicted_bbox = (
            self.motion.motion_model.predict()
        )

        self.statistics.age += 1

    def miss(self):

        self.statistics.age += 1

        self.statistics.misses += 1

        # CONFIRMED -> LOST after lost_threshold consecutive misses (distinct from max_misses)
        # e.g. lost_threshold=5 (~0.16s @30fps) gives brief occlusion grace before final removal at max_misses=30
        if self.state == TrackState.CONFIRMED and self.statistics.misses >= self.lost_threshold:
            self.state = TrackState.LOST
        # TENTATIVE -> REMOVED or TENTATIVE_LOST: pure noise (hits==0) removed immediately at 3 misses,
        # but tracks with >=1 hit get grace window (15 frames ~500ms) before REMOVED, allowing rejoin
        elif self.state == TrackState.TENTATIVE and self.statistics.misses >= self.tentative_max_misses:
            if self.statistics.hits == 0:
                self.state = TrackState.REMOVED
            else:
                self.state = TrackState.TENTATIVE_LOST
        elif self.state == TrackState.TENTATIVE_LOST and self.statistics.misses >= self.tentative_max_misses + self.tentative_lost_grace:
            self.state = TrackState.REMOVED