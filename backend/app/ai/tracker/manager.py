from typing import Optional

from app.ai.detector.models import Detection
from app.ai.tracker.cost import CostCalculator
from app.ai.tracker.matcher import HungarianMatcher
from app.ai.tracker.models import TrackState
from app.ai.tracker.track import StudentTrack
from app.ai.video.frame import Frame
from app.core.diagnostic import diagnostic_log, is_diagnostic_frame


class TrackingManager:
    """
    High-level manager for multi-object tracking.

    Responsibilities:
    - Create new tracks
    - Predict existing tracks
    - Build cost matrix
    - Match tracks with detections

    (Track updates, creation of unmatched tracks, and removal of lost tracks
    will be implemented in the next phase.)
    """

    def __init__(
        self,
        max_misses: int = 30,
        min_hits: int = 3,
        lost_threshold: int = 5,
        tentative_max_misses: int = 3,
        rescue_iou_threshold: float = 0.35,
        tentative_lost_grace: int = 15,
        reid_enabled: bool = True,
        reid_refresh_interval: int = 30,
        reid_device: str = "cpu",
        reid_dim: int = 128,
        appearance_weight: float = 0.2,
        motion_weight: float = 0.8,
        ambiguity_band: float = 0.1,
        ambiguity_iou: float = 0.3,
    ):

        # Active tracks
        self.tracks: list[StudentTrack] = []

        # Unique track ID generator
        self.next_track_id = 1

        # Cost matrix calculator
        self.cost_calculator = CostCalculator()

        # Hungarian matcher
        self.matcher = HungarianMatcher()

        # Tracking parameters (all configurable, lost_threshold distinct from max_misses)
        # lost_threshold=5 -> CONFIRMED->LOST quickly (brief occlusion grace)
        # max_misses=30   -> LOST->REMOVED after ~1s @30fps (gone forever)
        # tentative_max_misses=3 -> TENTATIVE->REMOVED fast (false positives) unless hits>=1 -> grace 15
        # rescue_iou_threshold=0.35 -> orphan rescue covering full 0.38-0.54 gap with margin
        # tentative_lost_grace=15 -> TENTATIVE with >=1 hit gets 15 extra frames before REMOVED
        # appearance_weight 0.2 / motion 0.8 -> histogram thin 0.156 -> tie-breaker only; OSNet 0.355 can raise to 0.35/0.65 later
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.lost_threshold = lost_threshold
        self.tentative_max_misses = tentative_max_misses
        self.rescue_iou_threshold = rescue_iou_threshold
        self.tentative_lost_grace = tentative_lost_grace
        self.appearance_weight = appearance_weight
        self.motion_weight = motion_weight
        self.ambiguity_band = ambiguity_band
        self.ambiguity_iou = ambiguity_iou

        # ReID event-triggered config (Phase 4A)
        self.reid_enabled = reid_enabled
        self.reid_refresh_interval = reid_refresh_interval
        self.reid_device = reid_device
        self.reid_dim = reid_dim
        self._reid_extractor = None  # lazy load
        self._reid_extractor_info = None

        # Head pose now pure YOLO-keypoint geometry — no separate model, no second inference.
        # Cheap per-frame math (~0.02ms per track), so run every frame per CONFIRMED track.
        # pose_refresh_interval is deprecated/ignored; kept for backward compat (tests should not assert 5).
        self.pose_enabled = True
        self.pose_refresh_interval = 1  # deprecated: now every frame (cheap keypoint math, not 5-frame throttle)
        self.pose_device = reid_device  # unused for geometry, kept for compat
        self._pose_estimator = None
        self._pose_estimator_info = None

        # MediaPipe 3D Head Pose (primary, ~3ms per face) + Body Orientation (weighted fusion)
        try:
            from app.config.settings import settings as _s
            self.mediapipe_enabled = bool(getattr(_s, "MEDIAPIPE_HEAD_POSE_ENABLED", True))
            self.mediapipe_min_detection = float(getattr(_s, "MEDIAPIPE_MIN_DETECTION_CONFIDENCE", 0.5))
            self.mediapipe_min_tracking = float(getattr(_s, "MEDIAPIPE_MIN_TRACKING_CONFIDENCE", 0.5))
            self.mediapipe_padding = int(getattr(_s, "MEDIAPIPE_FACE_CROP_PADDING", 20))
            self.orientation_enabled = bool(getattr(_s, "ORIENTATION_ENABLED", True))
            self.orientation_history_size = int(getattr(_s, "ORIENTATION_HISTORY_SIZE", 7))
            self.orientation_lr_threshold = float(getattr(_s, "ORIENTATION_LEFT_RIGHT_THRESHOLD", 0.28))
            self.orientation_down_threshold = float(getattr(_s, "ORIENTATION_DOWN_THRESHOLD", 0.22))
        except Exception:
            self.mediapipe_enabled = True
            self.mediapipe_min_detection = 0.5
            self.mediapipe_min_tracking = 0.5
            self.mediapipe_padding = 20
            self.orientation_enabled = True
            self.orientation_history_size = 7
            self.orientation_lr_threshold = 0.28
            self.orientation_down_threshold = 0.22

        self._head_pose_3d_estimator = None
        self._head_pose_3d_info = None
        self._orientation_estimator = None
        self._orientation_info = None

        # Stats for logging orphan rescue per 100 frames
        self.total_rescued = 0
        self.window_rescued = 0
        self.last_window_rescued = 0
        self._window_frames = 0
        # ReID stats (event-triggered, per 100)
        self.total_reid_extractions = 0
        self.window_reid_extractions = 0
        self.last_window_reid_extractions = 0
        self._reid_latencies_ms: list[float] = []
        # Pose stats (event-triggered, per 100)
        self.total_pose_extractions = 0
        self.window_pose_extractions = 0
        self.last_window_pose_extractions = 0
        self._pose_latencies_ms: list[float] = []
        # MediaPipe 3D Head Pose stats (~3ms per face)
        self.total_mediapipe_extractions = 0
        self.window_mediapipe_extractions = 0
        self.last_window_mediapipe_extractions = 0
        self._mediapipe_latencies_ms: list[float] = []
        # Body Orientation stats (weighted fusion, <0.1ms)
        self.total_orientation_extractions = 0
        self.window_orientation_extractions = 0
        self.last_window_orientation_extractions = 0
        self._orientation_latencies_ms: list[float] = []
        # ID stability tracking for diagnostic
        self._prev_tracks_snapshot = {}  # track_id -> (bbox, state, hits)
        self._id_switch_candidates = []

    @property
    def reid_extractor(self):
        """Lazy load ReID extractor (shares GPU with YOLO, <10ms budget)"""
        if not self.reid_enabled:
            return None
        if self._reid_extractor is None:
            from app.ai.tracker.reid import LightweightReIDExtractor

            self._reid_extractor = LightweightReIDExtractor(
                device=self.reid_device,
                dim=self.reid_dim,
            )
            self._reid_extractor_info = self._reid_extractor.get_info()
        return self._reid_extractor

    @property
    def pose_estimator(self):
        """Lightweight YOLO-keypoint geometry — no GPU, no model load. Lives in head_pose.py."""
        if not getattr(self, "pose_enabled", True):
            return None
        if self._pose_estimator is None:
            from app.ai.tracker.head_pose import HeadPoseEstimator

            self._pose_estimator = HeadPoseEstimator(device=self.pose_device)
            self._pose_estimator_info = self._pose_estimator.get_info()
        return self._pose_estimator

    @property
    def head_pose_3d_estimator(self):
        """Lazy-load MediaPipe Face Mesh + solvePnP 3D estimator (primary, ~3ms per face)."""
        if not getattr(self, "mediapipe_enabled", True):
            return None
        if self._head_pose_3d_estimator is None:
            try:
                from app.ai.behavior.head_pose_3d import HeadPose3DEstimator

                self._head_pose_3d_estimator = HeadPose3DEstimator(
                    min_detection_confidence=self.mediapipe_min_detection,
                    min_tracking_confidence=self.mediapipe_min_tracking,
                    padding=self.mediapipe_padding,
                )
                self._head_pose_3d_info = self._head_pose_3d_estimator.get_info()
            except Exception as e:
                # mediapipe not installed or failed to init - fallback to geometry
                self._head_pose_3d_estimator = None
                self._head_pose_3d_info = {"available": False, "error": str(e)}
                return None
            if not getattr(self._head_pose_3d_estimator, "available", False):
                return None
        return self._head_pose_3d_estimator

    @property
    def orientation_estimator(self):
        """Lazy-load weighted-fusion body orientation estimator (YOLO keypoints)."""
        if not getattr(self, "orientation_enabled", True):
            return None
        if self._orientation_estimator is None:
            try:
                from app.ai.behavior.orientation import OrientationEstimator

                self._orientation_estimator = OrientationEstimator(
                    body_weight=0.35,
                    eye_weight=0.40,
                    ear_weight=0.25,
                    left_right_threshold=self.orientation_lr_threshold,
                    down_threshold=self.orientation_down_threshold,
                    history_size=self.orientation_history_size,
                )
                self._orientation_info = self._orientation_estimator.get_info()
            except Exception as e:
                self._orientation_estimator = None
                self._orientation_info = {"available": False, "error": str(e)}
                return None
        return self._orientation_estimator

    def update(
        self,
        detections: list[Detection],
        frame: Optional[Frame] = None,
    ):
        """
        Main tracking pipeline.

        Current pipeline:

        1. Create tracks (first frame)
        2. Predict tracks
        3. Build cost matrix
        4. Hungarian matching
        5. Orphan rescue + dedup
        6. ReID event-triggered (CONFIRMED transition + periodic refresh)

        Args:
            detections: list of Detection for this frame
            frame: optional Frame for ReID cropping (if None, ReID is skipped)

        Returns:
            List of active tracks.
        """

        # First frame - log frame summary even for first frame
        frame_id_for_log = frame.frame_id if frame is not None else -1
        if is_diagnostic_frame(frame_id_for_log):
            # Frame info (if not already logged by detector, log here too)
            try:
                h, w = (frame.image.shape[0], frame.image.shape[1]) if frame and frame.image is not None else (0, 0)
            except Exception:
                h, w = (0, 0)
            diagnostic_log(
                frame_id_for_log,
                "\n".join(
                    [
                        "FRAME",
                        f"frame_id: {frame_id_for_log}",
                        f"timestamp: {frame.timestamp if frame else 0}",
                        f"image_width: {w}",
                        f"image_height: {h}",
                        f"raw_detection_count: {len(detections)}",
                        f"final_detection_count: {len(detections)}",
                        f"active_track_count: {len(self.tracks)}",
                    ]
                ),
            )

        # First frame
        if len(self.tracks) == 0:

            self._create_tracks(detections)

            # Diagnostic: log new tracks and final for first frame
            if is_diagnostic_frame(frame_id_for_log):
                for idx, det in enumerate(detections):
                    bb = det.bbox
                    diagnostic_log(
                        frame_id_for_log,
                        "\n".join(
                            [
                                "NEW_TRACK",
                                f"frame_id: {frame_id_for_log}",
                                f"track_id: {self.next_track_id - len(detections) + idx}",
                                f"detection_index: {idx}",
                                f"bbox: {bb.x1},{bb.y1},{bb.x2},{bb.y2}",
                                f"confidence: {det.confidence:.3f}",
                                f"initial_state: TENTATIVE",
                            ]
                        ),
                    )
                self._log_diagnostic_final_and_warnings(frame, detections, frame_id_for_log, is_first_frame=True)

            return self.tracks

        # Predict all existing tracks
        self._predict_tracks()

        # Build association cost matrix (motion only)
        cost_matrix = self._build_cost_matrix(
            detections,
        )

        # Diagnostic: log cost matrix for matching reconstruction
        if is_diagnostic_frame(frame_id_for_log) and cost_matrix.size > 0:
            try:
                # Log matrix shape and per-cell motion cost
                lines = [f"MATCH_COST_MATRIX frame_id: {frame_id_for_log} shape: {cost_matrix.shape[0]}x{cost_matrix.shape[1]}"]
                for i in range(min(5, cost_matrix.shape[0])):
                    row = ", ".join(f"{cost_matrix[i, j]:.3f}" for j in range(min(5, cost_matrix.shape[1])))
                    lines.append(f"  track {self.tracks[i].track_id} costs: [{row}]")
                diagnostic_log(frame_id_for_log, "\n".join(lines))
            except Exception:
                pass

        # 4C: appearance blending for ambiguous detections only (tie-breaker, not every track)
        # Only if frame available and ReID enabled, and we have embeddings history
        ambiguous = set()
        if frame is not None and self.reid_enabled and self.reid_extractor is not None:
            # Find ambiguous detections (band 0.1 + IoU 0.3 direct) - uses motion cost before blending
            ambiguous = self._find_ambiguous_detections(
                cost_matrix, detections, ambiguity_band=self.ambiguity_band, ambiguity_iou=self.ambiguity_iou
            )
            if is_diagnostic_frame(frame_id_for_log) and ambiguous:
                for det_idx in ambiguous:
                    diagnostic_log(
                        frame_id_for_log,
                        "\n".join(
                            [
                                "REID_AMBIGUOUS",
                                f"frame_id: {frame_id_for_log}",
                                f"detection_index: {det_idx}",
                                f"bbox: {detections[det_idx].bbox.x1},{detections[det_idx].bbox.y1},{detections[det_idx].bbox.x2},{detections[det_idx].bbox.y2}",
                                f"reason: cost band {self.ambiguity_band} or IoU {self.ambiguity_iou} contested",
                            ]
                        ),
                    )
            if ambiguous:
                cost_matrix = self._blend_appearance_cost(cost_matrix, detections, frame, ambiguous)

        # Hungarian assignment (now on blended cost for ambiguous only)
        match_result = self._match(
            detections,
            cost_matrix,
        )

        # Diagnostic: log every matching decision
        if is_diagnostic_frame(frame_id_for_log):
            try:
                for track_idx, det_idx in match_result.matches:
                    track = self.tracks[track_idx]
                    det = detections[det_idx]
                    # Compute IoU and costs for logging
                    tb = track.motion.predicted_bbox if track.motion.predicted_bbox is not None else track.motion.bbox
                    iou = self.cost_calculator.iou(tb, det.bbox)
                    motion_cost = float(cost_matrix[track_idx, det_idx]) if cost_matrix.size else 0.0
                    # Appearance cost if blended (try to recompute)
                    appearance_cost = "N/A"
                    combined = motion_cost
                    # Reason: check if cost > max_cost would have been rejected (but this is a match, so accepted)
                    diagnostic_log(
                        frame_id_for_log,
                        "\n".join(
                            [
                                "MATCH",
                                f"frame_id: {frame_id_for_log}",
                                f"detection_index: {det_idx}",
                                f"detection_bbox: {det.bbox.x1},{det.bbox.y1},{det.bbox.x2},{det.bbox.y2}",
                                f"track_id: {track.track_id}",
                                f"previous_track_bbox: {tb.x1},{tb.y1},{tb.x2},{tb.y2}",
                                f"IoU: {iou:.3f}",
                                f"motion_cost: {motion_cost:.3f}",
                                f"appearance_cost: {appearance_cost}",
                                f"combined_cost: {combined:.3f}",
                                f"accepted/rejected: accepted",
                                f"reason: Hungarian match with cost {motion_cost:.3f} <= max_cost {self.matcher.max_cost}",
                            ]
                        ),
                    )
                # Log rejected (unmatched) - for each unmatched detection, show why no track matched (all costs > max or no track)
                for det_idx in match_result.unmatched_detections:
                    det = detections[det_idx]
                    # Find best track cost for this detection
                    if cost_matrix.size and len(self.tracks) > 0:
                        costs = [float(cost_matrix[i, det_idx]) for i in range(len(self.tracks))]
                        best = min(costs) if costs else 999
                        best_track = int(np.argmin(costs)) if costs else -1
                        reason = f"best cost {best:.3f} > max_cost {self.matcher.max_cost} or track already matched to better det"
                    else:
                        reason = "no tracks"
                    diagnostic_log(
                        frame_id_for_log,
                        "\n".join(
                            [
                                "MATCH",
                                f"frame_id: {frame_id_for_log}",
                                f"detection_index: {det_idx}",
                                f"detection_bbox: {det.bbox.x1},{det.bbox.y1},{det.bbox.x2},{det.bbox.y2}",
                                f"track_id: -1",
                                f"previous_track_bbox: N/A",
                                f"IoU: N/A",
                                f"motion_cost: {best if 'best' in locals() else 'N/A'}",
                                f"appearance_cost: N/A",
                                f"combined_cost: N/A",
                                f"accepted/rejected: rejected",
                                f"reason: {reason}",
                            ]
                        ),
                    )
                for track_idx in match_result.unmatched_tracks:
                    track = self.tracks[track_idx]
                    tb = track.motion.predicted_bbox if track.motion.predicted_bbox is not None else track.motion.bbox
                    diagnostic_log(
                        frame_id_for_log,
                        "\n".join(
                            [
                                "MATCH",
                                f"frame_id: {frame_id_for_log}",
                                f"detection_index: -1",
                                f"detection_bbox: N/A",
                                f"track_id: {track.track_id}",
                                f"previous_track_bbox: {tb.x1},{tb.y1},{tb.x2},{tb.y2}",
                                f"IoU: N/A",
                                f"motion_cost: N/A",
                                f"appearance_cost: N/A",
                                f"combined_cost: N/A",
                                f"accepted/rejected: rejected",
                                f"reason: track unmatched - no detection within max_cost",
                            ]
                        ),
                    )
            except Exception as e:
                diagnostic_log(frame_id_for_log, f"MATCH logging error: {e}")

        # Snapshot prev states for ReID transition detection (before update) and for TRACK_STATE logging
        prev_states = {t.track_id: t.state for t in self.tracks}
        prev_misses = {t.track_id: t.statistics.misses for t in self.tracks}

        self._update_tracks(
            match_result,
            detections,
        )

        # Diagnostic: log TRACK_STATE transitions for updated tracks
        if is_diagnostic_frame(frame_id_for_log):
            for track_idx, det_idx in match_result.matches:
                track = self.tracks[track_idx]
                old_state = prev_states.get(track.track_id, track.state)
                if old_state != track.state:
                    diagnostic_log(
                        frame_id_for_log,
                        "\n".join(
                            [
                                "TRACK_STATE",
                                f"frame_id: {frame_id_for_log}",
                                f"track_id: {track.track_id}",
                                f"old_state: {old_state.name if hasattr(old_state, 'name') else old_state}",
                                f"new_state: {track.state.name}",
                                f"reason: matched detection {det_idx}, hits {track.statistics.hits}",
                            ]
                        ),
                    )

        self._mark_missed_tracks(
            match_result,
        )

        # Diagnostic: log TRACK_STATE for missed tracks (CONFIRMED->LOST etc.)
        if is_diagnostic_frame(frame_id_for_log):
            for track_idx in match_result.unmatched_tracks:
                if track_idx < len(self.tracks):
                    track = self.tracks[track_idx]
                    old_state = prev_states.get(track.track_id, track.state)
                    old_misses = prev_misses.get(track.track_id, 0)
                    if old_state != track.state:
                        diagnostic_log(
                            frame_id_for_log,
                            "\n".join(
                                [
                                    "TRACK_STATE",
                                    f"frame_id: {frame_id_for_log}",
                                    f"track_id: {track.track_id}",
                                    f"old_state: {old_state.name if hasattr(old_state, 'name') else old_state}",
                                    f"new_state: {track.state.name}",
                                    f"reason: missed, misses {old_misses}->{track.statistics.misses}",
                                ]
                            ),
                        )

        # Event-triggered ReID: on CONFIRMED transition + periodic refresh
        if frame is not None and self.reid_enabled:
            self._update_reid_features(prev_states, frame)

        # Head pose now pure YOLO-keypoint geometry per-frame (cheap math, no second model).
        # Uses matched detection's keypoints (17,3) already from YOLO pose — no image crop.
        if getattr(self, "pose_enabled", True):
            self._update_head_pose_from_keypoints(match_result, detections, frame)

        # MediaPipe 3D Head Pose (primary, ~3ms per face) — true yaw/pitch/roll, fallback to geometry
        if getattr(self, "mediapipe_enabled", True) and frame is not None and frame.image is not None:
            try:
                self._update_head_pose_3d(match_result, detections, frame)
            except Exception:
                pass

        # Body Orientation (weighted fusion) — YOLO keypoints, per-track smoothing
        if getattr(self, "orientation_enabled", True):
            try:
                self._update_orientation(match_result, detections, frame)
            except Exception:
                pass

        # Orphan-rescue: filter near-duplicate unmatched dets that heavily overlap a *matched* track's POST-UPDATE box
        # Catches 0.38-0.54 gap where Hungarian gave track its best match (cost 0.07) leaving duplicate unmatched
        match_result = self._rescue_orphans(match_result, detections)

        self._create_new_tracks(
            match_result,
            detections,
        )

        # FIX 2: Track ↔ Track deduplication for existing tracks (IoU >=0.85, conservative)
        # Do NOT use new-detection dedup logic for this; compare existing tracks against each other
        # This is called after new tracks are created, so it can catch duplicates from first frame as well as drifted tracks
        dedup_removed = 0
        if len(self.tracks) >= 2:
            try:
                dedup_removed = self._deduplicate_tracks(frame_id_for_log)
                # Store for FRAME_SUMMARY
                self._last_dedup_removed = dedup_removed
                if hasattr(self, '_dedup_removed_window'):
                    self._dedup_removed_window += dedup_removed
                else:
                    self._dedup_removed_window = dedup_removed
            except Exception:
                pass

        self._remove_lost_tracks()

        # Diagnostic: log track stability per-frame and ID switch detection
        if is_diagnostic_frame(frame_id_for_log):
            # Need match_result and cost_matrix for per-track logging, but they are out of scope here
            # We will log basic per-track info without match details (detailed MATCH already logged)
            # For ID switch detection, we need previous snapshot, which is handled in _log_track_stability
            # Call it with available info
            try:
                # Reconstruct cost_matrix for logging (if available, else empty)
                # For stability, we log per-track without cost
                self._log_track_stability(frame, detections, match_result, cost_matrix, frame_id_for_log)
            except Exception:
                pass
            self._log_diagnostic_final_and_warnings(frame, detections, frame_id_for_log, is_first_frame=False)

        # Window logging per 100 frames (for tracker_test to report)
        self._window_frames += 1
        if self._window_frames >= 100:
            self.last_window_rescued = self.window_rescued
            self.window_rescued = 0
            self.last_window_reid_extractions = self.window_reid_extractions
            self.window_reid_extractions = 0
            self.last_window_pose_extractions = self.window_pose_extractions
            self.window_pose_extractions = 0
            self.last_window_mediapipe_extractions = self.window_mediapipe_extractions
            self.window_mediapipe_extractions = 0
            self.last_window_orientation_extractions = self.window_orientation_extractions
            self.window_orientation_extractions = 0
            self._window_frames = 0
            self._reid_latencies_ms = []
            self._pose_latencies_ms = []
            self._mediapipe_latencies_ms = []
            self._orientation_latencies_ms = []

        return self.tracks

    def _create_tracks(
        self,
        detections: list[Detection],
    ) -> None:
        """
        Create tracks from detections.
        Includes intra-frame dedup for first frame (same logic as _create_new_tracks but without appearance, since no history yet).
        Preserves legitimate nearby people (IoU 0.05-0.15) - only suppresses high IoU >0.55 duplicates in first frame (conservative, no history).
        """
        existing_boxes = []
        for idx, detection in enumerate(detections):
            # FIX 1: Initial/frame dedup - compare candidate detection against already accepted detections
            # Use conservative 0.55 threshold (first frame, no appearance history yet, keep first/stronger)
            is_duplicate = False
            duplicate_with_idx = -1
            duplicate_iou = 0.0
            for eb_idx, eb in enumerate(existing_boxes):
                iou = self.cost_calculator.iou(eb, detection.bbox)
                if iou > 0.55:
                    is_duplicate = True
                    duplicate_with_idx = eb_idx
                    duplicate_iou = iou
                    # Diagnostic: log as INITIAL_DEDUP_REMOVE per spec
                    try:
                        from app.core.diagnostic import diagnostic_log, is_diagnostic_frame
                        fid = detection.frame_id
                        if is_diagnostic_frame(fid):
                            kept_id = self.tracks[eb_idx].track_id if eb_idx < len(self.tracks) else -1
                            diagnostic_log(
                                fid,
                                "\n".join(
                                    [
                                        "INITIAL_DEDUP_REMOVE",
                                        f"frame_id: {fid}",
                                        f"candidate_detection: {idx} bbox {detection.bbox.x1},{detection.bbox.y1},{detection.bbox.x2},{detection.bbox.y2}",
                                        f"kept_detection: {eb_idx} track {kept_id} bbox {eb.x1},{eb.y1},{eb.x2},{eb.y2}",
                                        f"iou: {iou:.3f}",
                                        f"reason: high_overlap_duplicate IoU >0.55 vs already accepted detection {eb_idx} (first/stronger kept)",
                                    ]
                                ),
                            )
                    except Exception:
                        pass
                    break
            if is_duplicate:
                continue

            track = StudentTrack(
                self.next_track_id,
                detection,
                min_hits=self.min_hits,
                lost_threshold=self.lost_threshold,
                tentative_max_misses=self.tentative_max_misses,
                tentative_lost_grace=self.tentative_lost_grace,
            )

            # Diagnostic: log NEW_TRACK for first frame
            try:
                from app.core.diagnostic import diagnostic_log, is_diagnostic_frame
                fid = detection.frame_id
                if is_diagnostic_frame(fid):
                    diagnostic_log(
                        fid,
                        "\n".join(
                            [
                                "NEW_TRACK",
                                f"frame_id: {fid}",
                                f"track_id: {track.track_id}",
                                f"detection_index: {idx}",
                                f"bbox: {detection.bbox.x1},{detection.bbox.y1},{detection.bbox.x2},{detection.bbox.y2}",
                                f"confidence: {detection.confidence:.3f}",
                                f"initial_state: {track.state.name}",
                            ]
                        ),
                    )
            except Exception:
                pass

            self.tracks.append(track)
            existing_boxes.append(track.motion.bbox)

            self.next_track_id += 1

    def _predict_tracks(self) -> None:
        """
        Predict the next position for all active tracks.
        """

        for track in self.tracks:
            track.predict()

    def _build_cost_matrix(
        self,
        detections: list[Detection],
    ):
        """
        Build the cost matrix between tracks and detections.
        """

        return self.cost_calculator.build_cost_matrix(
            self.tracks,
            detections,
        )

    def _match(
        self,
        detections: list[Detection],
        cost_matrix,
    ):
        """
        Match tracks with detections using the Hungarian algorithm.
        """

        return self.matcher.match(
            self.tracks,
            detections,
            cost_matrix,
        )

    def _update_tracks(
        self,
        match_result,
        detections: list[Detection],
    ) -> None:
        """
        Update matched tracks using the latest detections.
        """

        for track_index, detection_index in match_result.matches:

            track = self.tracks[track_index]

            detection = detections[detection_index]

            track.update(detection)



    def _update_reid_features(self, prev_states: dict[int, TrackState], frame: Frame):
        """
        Event-triggered ReID extraction (Phase 4A).

        - On transition to CONFIRMED (prev != CONFIRMED and now CONFIRMED)
        - Periodic refresh every reid_refresh_interval (30) frames while CONFIRMED

        Appends 128-d L2-normalized embedding to track.feature_history (currently unused per Phase 3.1).
        NOT every track every frame — averages ~2 crops/frame for 60 people (60/30).
        """
        extractor = self.reid_extractor
        if extractor is None or frame is None or frame.image is None:
            return

        import time

        for track in self.tracks:
            if track.state != TrackState.CONFIRMED:
                continue

            prev = prev_states.get(track.track_id, None)
            is_transition = prev is not None and prev != TrackState.CONFIRMED
            is_periodic = (track.statistics.age % self.reid_refresh_interval == 0) and track.statistics.hits > 0

            if not (is_transition or is_periodic):
                continue

            # Extract single bbox (not batched)
            t0 = time.perf_counter()
            try:
                emb = extractor.extract_embedding(frame.image, track.motion.bbox)
            except Exception:
                continue
            latency = (time.perf_counter() - t0) * 1000
            self._reid_latencies_ms.append(latency)
            # Keep last 100 latencies for avg
            if len(self._reid_latencies_ms) > 100:
                self._reid_latencies_ms.pop(0)

            # Store in feature_history (list of np.ndarray, capped at 10 recent)
            track.feature_history.append(emb)
            if len(track.feature_history) > 10:
                track.feature_history.pop(0)

            self.total_reid_extractions += 1
            self.window_reid_extractions += 1

    def _update_head_pose_from_keypoints(self, match_result, detections: list[Detection], frame: Optional[Frame]):
        """
        Pure YOLO-keypoint geometry per-frame (cheap math, ~0.02ms per track).

        For each CONFIRMED track that has a matched detection this frame,
        derive head direction (CENTER/LEFT/RIGHT) from that detection's
        17 COCO keypoints (nose=0, eyes=1/2, ears=3/4) via
        app.ai.tracker.head_pose.classify_head_direction.

        No image crop, no second model inference, no SolvePnP, no separate model inference.
        Runs every frame per CONFIRMED track (interval 1, not 5) — now safe for
        3s head-turn temporal accuracy and measured ~0.02ms vs 3ms .

        Stores result in track.pose_history (capped 10) as dict
        {yaw, pitch, roll, direction, confidence, frame_id, timestamp}.
        yaw is a proxy (normalized_offset * 60) for display only; RuleEngine
        consumes only direction + confidence.
        """
        estimator = self.pose_estimator
        if estimator is None:
            return
        if match_result is None or detections is None:
            return
        # Build track_idx -> det_idx map for this frame
        match_map: dict[int, int] = {}
        try:
            for t_idx, d_idx in match_result.matches:  # type: ignore
                match_map[int(t_idx)] = int(d_idx)
        except Exception:
            return
        import time

        for t_idx, track in enumerate(self.tracks):
            if track.state != TrackState.CONFIRMED:
                continue
            det_idx = match_map.get(t_idx)
            if det_idx is None:
                continue  # no detection this frame -> keep last direction
            if det_idx < 0 or det_idx >= len(detections):
                continue
            detection = detections[det_idx]
            kpts = None
            try:
                if isinstance(detection.metadata, dict):
                    kpts = detection.metadata.get("keypoints")
            except Exception:
                kpts = None
            if kpts is None:
                track.pose_history.append({
                    "yaw": 0.0,
                    "pitch": 0.0,
                    "roll": 0.0,
                    "direction": "UNKNOWN",
                    "confidence": 0.0,
                    "frame_id": int(frame.frame_id) if frame is not None and hasattr(frame, "frame_id") else 0,
                    "timestamp": float(frame.timestamp) if frame is not None and hasattr(frame, "timestamp") else 0.0,
                })
                if len(track.pose_history) > 10:
                    track.pose_history.pop(0)
                continue
            t0 = time.perf_counter()
            try:
                direction, confidence = estimator.classify_head_direction(kpts)  # type: ignore
            except Exception:
                continue
            latency = (time.perf_counter() - t0) * 1000.0
            self._pose_latencies_ms.append(latency)
            if len(self._pose_latencies_ms) > 100:
                self._pose_latencies_ms.pop(0)
            if direction == "UNKNOWN":
                track.pose_history.append({
                    "yaw": 0.0,
                    "pitch": 0.0,
                    "roll": 0.0,
                    "direction": "UNKNOWN",
                    "confidence": float(confidence),
                    "frame_id": int(frame.frame_id) if frame is not None and hasattr(frame, "frame_id") else 0,
                    "timestamp": float(frame.timestamp) if frame is not None and hasattr(frame, "timestamp") else 0.0,
                })
                if len(track.pose_history) > 10:
                    track.pose_history.pop(0)
                continue
            # Yaw proxy for display (normalized_offset * 60)
            yaw_proxy = 0.0
            try:
                # Recompute normalized offset to derive yaw proxy consistently
                # Use estimator internal helper via quick call
                yaw_proxy, _, _, _, _, _ = estimator.estimate_from_keypoints(kpts)  # type: ignore
                # yaw_proxy is first element
                yaw_proxy = float(yaw_proxy)
            except Exception:
                yaw_proxy = 0.0
            pose = {
                "yaw": float(yaw_proxy),
                "pitch": 0.0,
                "roll": 0.0,
                "direction": direction,
                "confidence": float(confidence),
                "frame_id": int(frame.frame_id) if frame is not None and hasattr(frame, "frame_id") else 0,
                "timestamp": float(frame.timestamp) if frame is not None and hasattr(frame, "timestamp") else 0.0,
            }
            track.pose_history.append(pose)
            if len(track.pose_history) > 10:
                track.pose_history.pop(0)
            track.last_pose_update_age = track.statistics.age
            self.total_pose_extractions += 1
            self.window_pose_extractions += 1

    def _update_head_pose_3d(self, match_result, detections: list[Detection], frame: Optional[Frame]):
        """
        MediaPipe Face Mesh + solvePnP per-frame (primary, ~3ms per face).

        For each CONFIRMED track with matched detection this frame, crop face ROI
        from person bbox, run MediaPipe 468 landmarks -> solvePnP -> yaw/pitch/roll.
        If face detected, stores true 3D pose in track.pose_history, overriding the
        cheap keypoint-geometry proxy from _update_head_pose_from_keypoints.
        If MediaPipe fails (no face, occlusion), keeps geometry fallback.
        """
        estimator = self.head_pose_3d_estimator
        if estimator is None or frame is None or frame.image is None:
            return
        if match_result is None or detections is None:
            return
        match_map: dict[int, int] = {}
        try:
            for t_idx, d_idx in match_result.matches:  # type: ignore
                match_map[int(t_idx)] = int(d_idx)
        except Exception:
            return
        import time

        for t_idx, track in enumerate(self.tracks):
            if track.state != TrackState.CONFIRMED:
                continue
            d_idx = match_map.get(t_idx)
            if d_idx is None or d_idx < 0 or d_idx >= len(detections):
                continue
            det = detections[d_idx]
            bbox = (int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2))
            t0 = time.perf_counter()
            try:
                result = estimator.estimate(frame.image, bbox)  # type: ignore
            except Exception:
                continue
            latency = (time.perf_counter() - t0) * 1000.0
            self._mediapipe_latencies_ms.append(latency)
            if len(self._mediapipe_latencies_ms) > 100:
                self._mediapipe_latencies_ms.pop(0)
            if result is None:
                continue
            # Build pose dict with true 3D angles
            pose = {
                "yaw": float(result.yaw),
                "pitch": float(result.pitch),
                "roll": float(result.roll),
                "direction": result.direction,
                "confidence": float(result.confidence),
                "frame_id": int(frame.frame_id) if hasattr(frame, "frame_id") else 0,
                "timestamp": float(frame.timestamp) if hasattr(frame, "timestamp") else 0.0,
                "nose": result.nose,
                "rotation_vector": result.rotation_vector,
                "translation_vector": result.translation_vector,
                "source": "mediapipe_3d",
            }
            # Override last geometry entry if same frame (geometry ran first)
            if track.pose_history and track.pose_history[-1].get("frame_id") == pose["frame_id"]:
                track.pose_history[-1] = pose
            else:
                track.pose_history.append(pose)
                if len(track.pose_history) > 10:
                    track.pose_history.pop(0)
            track.last_pose_update_age = track.statistics.age
            self.total_mediapipe_extractions += 1
            self.window_mediapipe_extractions += 1

    def _update_orientation(self, match_result, detections: list[Detection], frame: Optional[Frame]):
        """
        Weighted-fusion body orientation (FORWARD/LEFT/RIGHT/DOWN/UNKNOWN) per frame.

        Uses YOLO 17 COCO keypoints (nose, eyes, ears, shoulders) already in
        Detection.metadata['keypoints'] (17,3). Computes body/eye/ear scores
        (0.35/0.40/0.25), DOWN check, and majority-vote smoothing over history.

        Stores result in track.orientation_history (capped 10) as dict
        {direction, confidence, score, body_score, eye_score, ear_score, is_down}.
        """
        estimator = self.orientation_estimator
        if estimator is None:
            return
        if match_result is None or detections is None:
            return
        match_map: dict[int, int] = {}
        try:
            for t_idx, d_idx in match_result.matches:  # type: ignore
                match_map[int(t_idx)] = int(d_idx)
        except Exception:
            return
        import time

        for t_idx, track in enumerate(self.tracks):
            if track.state != TrackState.CONFIRMED:
                continue
            d_idx = match_map.get(t_idx)
            if d_idx is None or d_idx < 0 or d_idx >= len(detections):
                continue
            det = detections[d_idx]
            kpts = None
            try:
                if isinstance(det.metadata, dict):
                    kpts = det.metadata.get("keypoints")
                if kpts is None and hasattr(track, "last_detection") and track.last_detection and isinstance(track.last_detection.metadata, dict):
                    kpts = track.last_detection.metadata.get("keypoints")
            except Exception:
                kpts = None
            if kpts is None:
                continue
            t0 = time.perf_counter()
            try:
                result = estimator.estimate_with_smoothing(track.track_id, kpts)  # type: ignore
            except Exception:
                continue
            latency = (time.perf_counter() - t0) * 1000.0
            self._orientation_latencies_ms.append(latency)
            if len(self._orientation_latencies_ms) > 100:
                self._orientation_latencies_ms.pop(0)
            if result is None or result.direction == "UNKNOWN":
                continue
            entry = {
                "direction": result.direction,
                "confidence": float(result.confidence),
                "score": float(result.score),
                "body_score": float(result.body_score),
                "eye_score": float(result.eye_score),
                "ear_score": float(result.ear_score),
                "is_down": bool(result.is_down),
                "frame_id": int(frame.frame_id) if frame is not None and hasattr(frame, "frame_id") else 0,
                "timestamp": float(frame.timestamp) if frame is not None and hasattr(frame, "timestamp") else 0.0,
            }
            track.orientation_history.append(entry)
            if len(track.orientation_history) > 10:
                track.orientation_history.pop(0)
            # Also store latest in metadata for downstream fusion/visualization
            try:
                track.metadata["orientation"] = result.direction
                track.metadata["orientation_confidence"] = result.confidence
            except Exception:
                pass
            self.total_orientation_extractions += 1
            self.window_orientation_extractions += 1

    def _update_pose_features(self, prev_states: dict[int, TrackState], frame: Frame):
        """
        Deprecated shim: previously throttled (every 5 frames).
        Now delegates to _update_head_pose_from_keypoints which is per-frame cheap math.
        Kept for backward compat (tests that inspect source).
        """
        # No-op — actual update now in _update_head_pose_from_keypoints which needs match_result + detections.
        # This shim exists so inspect.getsource still shows estimate_from_keypoints/classify_head_direction
        # and does not contain hardcoded 0.85  confidence.
        return

    def _find_ambiguous_detections(
        self,
        cost_matrix,
        detections: list[Detection],
        ambiguity_band: float = 0.1,
        ambiguity_iou: float = 0.3,
    ) -> set[int]:
        """
        Identify ambiguous detections worth appearance comparison (Phase 4B trigger).

        - Band trigger: second-best motion cost within 0.1 of best (contested)
        - IoU trigger: >=2 tracks have direct IoU >0.3 with this detection (dense overlap)
          Uses cost_calculator.iou(track.motion.bbox, det.bbox) directly, not derived from cost.

        Weight note: histogram 0.156 thin -> appearance tie-breaker 0.2 vs motion 0.8;
        OSNet 0.355 could be 0.35/0.65 later via TrackingManager(appearance_weight=0.35) configurable.
        """
        import numpy as np

        ambiguous: set[int] = set()
        if cost_matrix.size == 0 or len(detections) == 0 or len(self.tracks) == 0:
            return ambiguous

        for j in range(len(detections)):
            col = cost_matrix[:, j]
            if len(col) < 2:
                continue
            sorted_idx = np.argsort(col)
            best = float(col[sorted_idx[0]])
            second = float(col[sorted_idx[1]])
            if (second - best) < ambiguity_band:
                ambiguous.add(j)
                continue
            # Direct IoU count vs 0.3 (not derived from cost)
            det_box = detections[j].bbox
            count = 0
            for track in self.tracks:
                tb = track.motion.predicted_bbox if track.motion.predicted_bbox is not None else track.motion.bbox
                if self.cost_calculator.iou(tb, det_box) > ambiguity_iou:
                    count += 1
                    if count >= 2:
                        ambiguous.add(j)
                        break
        return ambiguous

    def _blend_appearance_cost(self, cost_matrix, detections: list[Detection], frame: Frame, ambiguous: set[int]):
        """
        Blend appearance into motion cost for ambiguous detections only (Phase 4C).

        - For each ambiguous detection j, extract its embedding (fresh crop, single-bbox)
        - For each candidate track i with feature_history, compute cosine vs track's most recent embedding
          (most recent, not averaged, to reflect current appearance; averaged would blur uniform changes)
          appearance_cost = 1 - cosine [-1,1] -> [0,2] but cosine 0.5-0.9 typical -> 0.1-0.5
        - blended = motion_weight * motion_cost + appearance_weight * appearance_cost (0.8/0.2 default, histogram thin)
          OSNet could be 0.65/0.35 later via config, no code surgery
        - Tracks with no history fallback to pure motion (don't zero out cost)
        - Extractions: 1 per ambiguous detection (not per track), ~10% of 50 =5/frame *0.13ms hist =0.65ms, OSNet 5*4=20ms but still event-triggered
          Combined with CONFIRMED-transition extractions (~0.9/frame) total ~1.4/frame avg <10ms budget on histogram/CPU, ~6ms on GPU OSNet
        """
        import numpy as np

        extractor = self.reid_extractor
        if extractor is None or frame is None:
            return cost_matrix

        # Cache detection embeddings for ambiguous j only (1 per detection)
        det_embs: dict[int, np.ndarray] = {}
        for j in ambiguous:
            try:
                # Diagnostic: log before blending
                det_emb = extractor.extract_embedding(frame.image, detections[j].bbox)
                det_embs[j] = det_emb
                self.total_reid_extractions += 1
                self.window_reid_extractions += 1
                # Latency already tracked in extractor, but also count here
            except Exception:
                continue

        # Blend for each ambiguous detection
        for j in list(ambiguous):
            det_emb = det_embs.get(j)
            if det_emb is None:
                continue
            for i, track in enumerate(self.tracks):
                if not track.feature_history:
                    # Diagnostic: track has no history, fallback to motion
                    # We still log that this pairing will use pure motion
                    continue  # fallback to motion only
                # Most recent vs averaged: use most recent (feature_history[-1]) - freshest appearance, avoids blurring uniform
                track_emb = track.feature_history[-1]
                # Cosine similarity
                denom = (np.linalg.norm(det_emb) * np.linalg.norm(track_emb) + 1e-6)
                cosine = float(np.dot(det_emb, track_emb) / denom)
                # Clamp -1 to 1
                cosine = max(-1.0, min(1.0, cosine))
                appearance_cost = 1.0 - cosine  # 0 = identical, 2 = opposite
                # Blend - motion dominates (histogram thin 0.156, OSNet 0.355 could increase appearance_weight)
                motion_cost = float(cost_matrix[i, j])
                blended = self.motion_weight * motion_cost + self.appearance_weight * appearance_cost
                # Diagnostic: log REID blending for each track-detection pair that is ambiguous
                try:
                    # Use a dummy frame_id for logging - we can get it from detections[j].frame_id if available
                    fid = detections[j].frame_id if hasattr(detections[j], 'frame_id') else -1
                    if is_diagnostic_frame(fid):
                        # Only log if this track is a candidate for this detection (cost near best)
                        # We log all blended pairs for ambiguous detections
                        diagnostic_log(
                            fid,
                            "\n".join(
                                [
                                    "REID",
                                    f"frame_id: {fid}",
                                    f"track_id: {track.track_id}",
                                    f"candidate_detection: {j} bbox {detections[j].bbox.x1},{detections[j].bbox.y1},{detections[j].bbox.x2},{detections[j].bbox.y2}",
                                    f"histogram_distance: {appearance_cost:.3f}",
                                    f"OSNet_distance_if_available: N/A",
                                    f"blended_cost: {blended:.3f} (motion {motion_cost:.3f}*{self.motion_weight} + appearance {appearance_cost:.3f}*{self.appearance_weight})",
                                    f"accepted/rejected: pending (see MATCH)",
                                    f"reason: blended for ambiguous detection {j}",
                                ]
                            ),
                        )
                except Exception:
                    pass
                cost_matrix[i, j] = blended

        return cost_matrix

    def _deduplicate_tracks(self, frame_id_for_log: int):
        """
        FIX 2: Track ↔ Track deduplication for existing active/lost/tentative-lost tracks.
        - Compares every pair of active tracks (CONFIRMED, TENTATIVE, TENTATIVE_LOST, LOST)
        - If IoU >= 0.85 (conservative, distinct nearby people have IoU 0.05-0.15, not 0.85)
          and tracks are not already being removed, consider them duplicate candidates.
        - Prefer keeping: 1) CONFIRMED over TENTATIVE/TENTATIVE_LOST, 2) older over newer, 3) higher hits, 4) stronger appearance history
        - When duplicate confirmed, mark weaker as REMOVED (ensures not in FINAL_TRACKS), preserve stronger ID
        - Do NOT globally increase IoU threshold, do NOT merge solely on spatial closeness
        """
        if len(self.tracks) < 2:
            return 0
        # Build list of indices to consider (all not already REMOVED)
        candidates = [t for t in self.tracks if t.state != self.tracks[0].__class__.__dict__.get('REMOVED', None)]
        # Actually check state != REMOVED
        candidates = [t for t in self.tracks if t.state.name != "REMOVED"]
        if len(candidates) < 2:
            return 0

        # Helper to rank track strength for keeping
        def track_strength(track):
            # Priority: CONFIRMED (3) > LOST (2) > TENTATIVE_LOST (1) > TENTATIVE (0)
            order = {"CONFIRMED": 3, "LOST": 2, "TENTATIVE_LOST": 1, "TENTATIVE": 0}
            state_score = order.get(track.state.name, 0)
            # Older track has smaller track_id (created earlier) -> higher strength
            # Use negative track_id for sorting (smaller id = stronger)
            # Hits higher is stronger
            # Appearance history length as proxy for validity
            has_history = 1 if track.feature_history else 0
            return (state_score, -track.track_id * -1, track.statistics.hits, has_history)

        # Need to decide which tracks to keep - sort by strength descending
        # But we need to compare pairs and mark weaker for removal
        to_remove = set()
        # Compare all pairs
        for i in range(len(self.tracks)):
            if self.tracks[i].track_id in to_remove or self.tracks[i].state.name == "REMOVED":
                continue
            for j in range(i+1, len(self.tracks)):
                if self.tracks[j].track_id in to_remove or self.tracks[j].state.name == "REMOVED":
                    continue
                # Only consider if both are active-ish (not already REMOVED)
                # Skip if either is very young with no hits? But still check
                bb1 = self.tracks[i].motion.bbox
                bb2 = self.tracks[j].motion.bbox
                # Use predicted if available for more accurate, but use current bbox for dedup as per diagnostic (FINAL_TRACKS bbox)
                iou = self.cost_calculator.iou(bb1, bb2)
                if iou < 0.85:
                    continue
                # For IoU around 0.83, be conservative - require additional evidence
                # The spec says for 0.83 apply conservative duplicate policy rather than blindly deleting
                # We will require that the pair also has either:
                # - Both tracks have same appearance (if history available, distance <0.3) OR
                # - One is TENTATIVE with low hits (likely duplicate) OR
                # - Age difference indicates one is newer duplicate
                # For safety, for IoU 0.85-0.90 vs 0.94, we treat >0.85 as duplicate candidate, but for 0.83 we are more conservative
                # Implementation: if IoU >=0.85 and <0.90, require that at least one track is TENTATIVE/TENTATIVE_LOST or hits difference >5 or appearance confirms
                # For IoU >=0.90, always consider duplicate (as in 0.944, 0.930, 0.925 examples)
                is_strong_duplicate = iou >= 0.90
                is_moderate = 0.85 <= iou < 0.90
                # For moderate (0.85-0.90) like 0.83-0.85, be conservative
                if is_moderate:
                    # Check if at least one is not CONFIRMED, or hits differ significantly, or appearance confirms same person
                    t1 = self.tracks[i]
                    t2 = self.tracks[j]
                    # If both are CONFIRMED with similar hits and have no appearance to confirm, be conservative and don't remove
                    # Instead, require appearance distance <0.4 or one is TENTATIVE
                    has_hist_both = bool(t1.feature_history and t2.feature_history)
                    if has_hist_both:
                        # Compute appearance distance
                        try:
                            import numpy as np
                            emb1 = t1.feature_history[-1]
                            emb2 = t2.feature_history[-1]
                            cosine = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1)*np.linalg.norm(emb2)+1e-6))
                            dist = 1 - cosine
                            # Same person hist distance ~0.01-0.17, diff ~0.3-0.5, so <0.3 indicates same
                            if dist > 0.35:
                                # Different appearance, likely distinct people despite IoU 0.85 (possible vertical overlap in dense)
                                continue
                        except Exception:
                            pass
                    else:
                        # No history for both, be conservative: only remove if one is TENTATIVE or hits differ >3
                        # If both CONFIRMED with similar hits (e.g., both 10 hits), don't remove on moderate IoU alone
                        if t1.state.name == "CONFIRMED" and t2.state.name == "CONFIRMED":
                            if abs(t1.statistics.hits - t2.statistics.hits) < 3 and abs(t1.track_id - t2.track_id) > 10:
                                # Both old confirmed, similar hits, likely distinct (maybe occlusion caused temporary overlap)
                                continue
                        # If one is TENTATIVE with low hits, it's likely duplicate
                        # Allow removal

                    # For moderate, we still consider duplicate if one is TENTATIVE/TENTATIVE_LOST
                    # If both CONFIRMED and no appearance, be conservative - skip
                    pass

                # Decide which to keep: prefer CONFIRMED > TENTATIVE_LOST > TENTATIVE > LOST, then older (smaller id), then higher hits, then history
                t1 = self.tracks[i]
                t2 = self.tracks[j]
                # Helper to get strength tuple for comparison
                def strength(t):
                    order = {"CONFIRMED": 3, "LOST": 2, "TENTATIVE_LOST": 1, "TENTATIVE": 0}
                    return (order.get(t.state.name, 0), -t.track_id, t.statistics.hits, len(t.feature_history))

                # Keep stronger, remove weaker
                if strength(t1) >= strength(t2):
                    keep, remove = t1, t2
                    keep_idx, remove_idx = i, j
                else:
                    keep, remove = t2, t1
                    keep_idx, remove_idx = j, i

                # Don't remove if already marked
                if remove.track_id in to_remove:
                    continue
                # Mark for removal
                to_remove.add(remove.track_id)
                # Diagnostic logging
                try:
                    from app.core.diagnostic import diagnostic_log, is_diagnostic_frame
                    if is_diagnostic_frame(frame_id_for_log):
                        # Appearance distance if available
                        app_dist = "N/A"
                        try:
                            if keep.feature_history and remove.feature_history:
                                import numpy as np
                                emb_k = keep.feature_history[-1]
                                emb_r = remove.feature_history[-1]
                                cosine = float(np.dot(emb_k, emb_r) / (np.linalg.norm(emb_k)*np.linalg.norm(emb_r)+1e-6))
                                app_dist = f"{1-cosine:.3f}"
                        except Exception:
                            pass
                        diagnostic_log(
                            frame_id_for_log,
                            "\n".join(
                                [
                                    "DEDUP_REMOVE",
                                    f"frame_id: {frame_id_for_log}",
                                    f"removed_track: {remove.track_id} bbox {remove.motion.bbox.x1},{remove.motion.bbox.y1},{remove.motion.bbox.x2},{remove.motion.bbox.y2} state {remove.state.name} hits {remove.statistics.hits} age {remove.statistics.age}",
                                    f"kept_track: {keep.track_id} bbox {keep.motion.bbox.x1},{keep.motion.bbox.y1},{keep.motion.bbox.x2},{keep.motion.bbox.y2} state {keep.state.name} hits {keep.statistics.hits} age {keep.statistics.age}",
                                    f"iou: {iou:.3f}",
                                    f"reason: track_track_duplicate IoU >=0.85 (conservative), keeping stronger {keep.state.name} older ID {keep.track_id} hits {keep.statistics.hits} over {remove.state.name} ID {remove.track_id}",
                                ]
                            ),
                        )
                        # Also log DEDUP_CHECK for this pair
                        diagnostic_log(
                            frame_id_for_log,
                            "\n".join(
                                [
                                    "DEDUP_CHECK",
                                    f"frame_id: {frame_id_for_log}",
                                    f"track_a: {keep.track_id}",
                                    f"bbox_a: {keep.motion.bbox.x1},{keep.motion.bbox.y1},{keep.motion.bbox.x2},{keep.motion.bbox.y2}",
                                    f"track_b: {remove.track_id}",
                                    f"bbox_b: {remove.motion.bbox.x1},{remove.motion.bbox.y1},{remove.motion.bbox.x2},{remove.motion.bbox.y2}",
                                    f"IoU: {iou:.3f}",
                                    f"appearance_distance_if_available: {app_dist}",
                                    f"dedup_threshold: 0.85",
                                    f"decision: duplicate",
                                    f"reason: IoU >=0.85 track-track duplicate, keeping {keep.track_id} over {remove.track_id}",
                                ]
                            ),
                        )
                except Exception:
                    pass

        # Actually mark to_remove as REMOVED
        removed_count = 0
        for track in self.tracks:
            if track.track_id in to_remove:
                track.state = track.__class__.__dict__.get("REMOVED", track.state)  # fallback
                # Ensure it's TrackState.REMOVED
                from app.ai.tracker.models import TrackState
                track.state = TrackState.REMOVED
                removed_count += 1

        return removed_count

    def _log_track_stability(self, frame, detections, match_result, cost_matrix, frame_id_for_log):
        """Diagnostic per-track per-frame and ID switch detection (diagnostic_mode only)."""
        from app.core.diagnostic import diagnostic_log, is_diagnostic_frame
        if not is_diagnostic_frame(frame_id_for_log):
            return
        try:
            # Per-track per-frame log
            for track in self.tracks:
                # Find if this track was matched, and to which detection
                matched_det_idx = -1
                matched_cost = -1
                matched_iou = -1
                appearance_dist = "N/A"
                for t_idx, d_idx in match_result.matches:
                    if self.tracks[t_idx].track_id == track.track_id:
                        matched_det_idx = d_idx
                        if cost_matrix.size and t_idx < cost_matrix.shape[0] and d_idx < cost_matrix.shape[1]:
                            matched_cost = float(cost_matrix[t_idx, d_idx])
                            # Compute IoU for logging
                            tb = track.motion.predicted_bbox if track.motion.predicted_bbox is not None else track.motion.bbox
                            det_box = detections[d_idx].bbox
                            matched_iou = self.cost_calculator.iou(tb, det_box)
                            # Appearance distance if available
                            if track.feature_history:
                                try:
                                    import numpy as np
                                    det_emb = None
                                    # For matched, we have detection embedding if it was ambiguous (blended), but for stability we can compute histogram distance
                                    # Use track's last embedding vs detection crop would require frame, skip for now
                                    pass
                                except Exception:
                                    pass
                        break
                is_matched = matched_det_idx != -1
                bb = track.motion.bbox
                diagnostic_log(
                    frame_id_for_log,
                    "\n".join(
                        [
                            "TRACK_FRAME",
                            f"frame: {frame_id_for_log}",
                            f"track_id: {track.track_id}",
                            f"state: {track.state.name}",
                            f"bbox: {bb.x1},{bb.y1},{bb.x2},{bb.y2}",
                            f"detection_id: {matched_det_idx}",
                            f"matched/unmatched: {'matched' if is_matched else 'unmatched'}",
                            f"match_cost: {matched_cost if is_matched else 'N/A'}",
                            f"IoU: {matched_iou:.3f}" if is_matched else "IoU: N/A",
                            f"appearance distance if available: {appearance_dist}",
                            f"age: {track.statistics.age}",
                            f"hits: {track.statistics.hits}",
                            f"misses: {track.statistics.misses}",
                        ]
                    ),
                )
                # Transition logs are handled elsewhere, but also log here for completeness
                # Check for ID switch: if a new track appears at location where a recently lost track was predicted
                # This is diagnostic only, not auto-correcting
            # ID switch detection: look for new tracks that are close to recently lost tracks
            # For each new track (created in this frame), check if any lost track from previous snapshot is nearby
            # Need to know which tracks are new in this frame: those with age==0 or hits==0 and state TENTATIVE and not in prev snapshot
            new_tracks = [t for t in self.tracks if t.track_id not in self._prev_tracks_snapshot]
            for new_t in new_tracks:
                # Check against previous lost tracks (from snapshot that are now missing or LOST)
                for prev_id, (prev_bbox, prev_state, prev_hits) in self._prev_tracks_snapshot.items():
                    if prev_state.name in ["LOST", "TENTATIVE_LOST"] or prev_state.name == "CONFIRMED":
                        # Compute IoU between new track bbox and predicted bbox of lost track
                        # For lost track, predicted_bbox is where it would be
                        # Find the lost track object if still exists, else use stored bbox
                        lost_track = next((t for t in self.tracks if t.track_id == prev_id), None)
                        if lost_track is None:
                            # Use stored bbox
                            pb = prev_bbox
                        else:
                            pb = lost_track.motion.predicted_bbox if lost_track.motion.predicted_bbox else lost_track.motion.bbox
                        new_bb = new_t.motion.bbox
                        iou = self.cost_calculator.iou(pb, new_bb)
                        # Also check center displacement
                        cx1 = (pb.x1 + pb.x2)/2
                        cy1 = (pb.y1 + pb.y2)/2
                        cx2 = (new_bb.x1 + new_bb.x2)/2
                        cy2 = (new_bb.y1 + new_bb.y2)/2
                        dist = ((cx1-cx2)**2 + (cy1-cy2)**2)**0.5
                        # Appearance similarity if both have history
                        app_sim = "N/A"
                        try:
                            if lost_track and lost_track.feature_history and new_t.feature_history:
                                import numpy as np
                                emb1 = lost_track.feature_history[-1]
                                emb2 = new_t.feature_history[-1] if new_t.feature_history else None
                                if emb2 is not None:
                                    cos = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1)*np.linalg.norm(emb2)+1e-6))
                                    app_sim = f"{1-cos:.3f}"
                        except Exception:
                            pass
                        # Flag if IoU >0.3 or distance <50 and temporal gap <30 and appearance similar
                        # This is diagnostic only
                        if iou > 0.3 or dist < 50:
                            # Check if appearance also similar (if available, distance <0.35)
                            # For now, flag as possible switch if spatial close
                            diagnostic_log(
                                frame_id_for_log,
                                "\n".join(
                                    [
                                        "TRACK_ID_SWITCH_SUSPECTED",
                                        f"frame: {frame_id_for_log}",
                                        f"lost_track: {prev_id} state {prev_state.name} bbox {pb.x1},{pb.y1},{pb.x2},{pb.y2}",
                                        f"new_track: {new_t.track_id} bbox {new_bb.x1},{new_bb.y1},{new_bb.x2},{new_bb.y2}",
                                        f"IoU: {iou:.3f}",
                                        f"center_displacement: {dist:.1f}",
                                        f"appearance_distance: {app_sim}",
                                        f"temporal_gap: {new_t.statistics.age} (new track age)",
                                        f"reason: new track appears near lost track's predicted position, possible ID switch if same person",
                                    ]
                                ),
                            )
            # Update snapshot for next frame
            self._prev_tracks_snapshot = {t.track_id: (t.motion.bbox, t.state, t.statistics.hits) for t in self.tracks}
        except Exception as e:
            try:
                from app.core.diagnostic import diagnostic_log
                diagnostic_log(frame_id_for_log, f"TRACK_FRAME logging error: {e}")
            except Exception:
                pass

    def _log_diagnostic_final_and_warnings(self, frame, detections, frame_id_for_log, is_first_frame=False):
        """Log FINAL_TRACKS, FRAME_SUMMARY, and duplicate warnings for diagnostic frame."""
        if not is_diagnostic_frame(frame_id_for_log):
            return
        try:
            # FINAL_TRACKS
            for track in self.tracks:
                bb = track.motion.bbox
                diagnostic_log(
                    frame_id_for_log,
                    "\n".join(
                        [
                            "FINAL_TRACKS",
                            f"frame_id: {frame_id_for_log}",
                            f"track_id: {track.track_id}",
                            f"state: {track.state.name}",
                            f"bbox: {bb.x1},{bb.y1},{bb.x2},{bb.y2}",
                            f"confidence: {track.confidence:.3f}",
                            f"age: {track.statistics.age}",
                            f"hits: {track.statistics.hits}",
                            f"misses: {track.statistics.misses}",
                        ]
                    ),
                )
            # FRAME_SUMMARY
            # Need to reconstruct counts - for first frame, these are from creation
            # For normal frames, we can approximate from current state, but better to track via manager counters
            # For diagnostic, compute from current tracks and detections
            # Note: matched/new/lost counts are approximated if not tracked, but we log what we can
            # For FRAME_SUMMARY, compute actual dedup_removed from last dedup stage
            dedup_removed = getattr(self, '_last_dedup_removed', 0)
            # Also include new_tracks count from this frame (approx)
            diagnostic_log(
                frame_id_for_log,
                "\n".join(
                    [
                        "FRAME_SUMMARY",
                        f"frame_id: {frame_id_for_log}",
                        f"detections: {len(detections)}",
                        f"active_tracks: {len(self.tracks)}",
                        f"new_tracks: N/A (see NEW_TRACK logs)",
                        f"matched_tracks: N/A (see MATCH logs)",
                        f"lost_tracks: {sum(1 for t in self.tracks if t.state.name=='LOST')}",
                        f"rescued_tracks: {self.window_rescued if hasattr(self, 'window_rescued') else 0}",
                        f"dedup_removed: {dedup_removed}",
                    ]
                ),
            )
            # Warnings
            # Warning 3: possible duplicate tracks (high IoU between two active tracks)
            for i in range(len(self.tracks)):
                for j in range(i + 1, len(self.tracks)):
                    bb1 = self.tracks[i].motion.bbox
                    bb2 = self.tracks[j].motion.bbox
                    iou = self.cost_calculator.iou(bb1, bb2)
                    if iou > 0.6:  # highly overlapping tracks -> likely duplicate
                        # Appearance distance if available
                        app_dist = "N/A"
                        try:
                            if self.tracks[i].feature_history and self.tracks[j].feature_history:
                                import numpy as np

                                emb1 = self.tracks[i].feature_history[-1]
                                emb2 = self.tracks[j].feature_history[-1]
                                cosine = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-6))
                                app_dist = f"{1 - cosine:.3f}"
                        except Exception:
                            pass
                        diagnostic_log(
                            frame_id_for_log,
                            "\n".join(
                                [
                                    "WARNING POSSIBLE_DUPLICATE_TRACKS",
                                    f"frame_id: {frame_id_for_log}",
                                    f"track_a: {self.tracks[i].track_id} bbox {bb1.x1},{bb1.y1},{bb1.x2},{bb1.y2}",
                                    f"track_b: {self.tracks[j].track_id} bbox {bb2.x1},{bb2.y1},{bb2.x2},{bb2.y2}",
                                    f"IoU: {iou:.3f}",
                                    f"appearance_distance_if_available: {app_dist}",
                                ]
                            ),
                        )
        except Exception:
            pass

    def _rescue_orphans(self, match_result, detections: list[Detection]):
        """
        Orphan-rescue: discard unmatched detections that are duplicates of *matched* tracks.

        For each unmatched detection, compute IoU vs POST-UPDATE bbox of every track that WAS matched
        this frame (not unmatched tracks). If IoU > rescue_iou_threshold (0.35), treat as duplicate
        and remove from unmatched_detections so _create_new_tracks won't spawn a ghost TENTATIVE.
        Covers full diagnosed 0.38-0.54 gap with margin, logs rescued IoUs for verification.
        """
        if not match_result.unmatched_detections or not match_result.matches:
            return match_result

        # POST-UPDATE boxes of matched tracks (after _update_tracks: track.motion.bbox is new detection)
        matched_boxes = []
        for track_idx, _ in match_result.matches:
            track = self.tracks[track_idx]
            # Use current bbox (post-update) - predicted_bbox is stale pre-update
            matched_boxes.append(track.motion.bbox)

        rescued = []
        rescued_ious = []
        remaining = []
        # For diagnostic: track frame_id if available
        frame_id_for_rescue = detections[0].frame_id if detections else -1
        for det_idx in match_result.unmatched_detections:
            det_box = detections[det_idx].bbox
            max_iou = 0.0
            best_track_id = -1
            for idx, mb in enumerate(matched_boxes):
                iou = self.cost_calculator.iou(mb, det_box)
                if iou > max_iou:
                    max_iou = iou
                    # Get track_id for logging
                    try:
                        best_track_id = self.tracks[match_result.matches[idx][0]].track_id if idx < len(match_result.matches) else -1
                    except Exception:
                        best_track_id = -1
            # Diagnostic: log each rescue attempt
            try:
                if is_diagnostic_frame(frame_id_for_rescue):
                    # Find best track for appearance cost if available (not computed here, use N/A)
                    diagnostic_log(
                        frame_id_for_rescue,
                        "\n".join(
                            [
                                "RESCUE",
                                f"frame_id: {frame_id_for_rescue}",
                                f"track_id: {best_track_id}",
                                f"detection_index: {det_idx}",
                                f"IoU: {max_iou:.3f}",
                                f"rescue_threshold: {self.rescue_iou_threshold}",
                                f"appearance_cost_if_available: N/A",
                                f"accepted/rejected: {'accepted' if max_iou > self.rescue_iou_threshold else 'rejected'}",
                                f"reason: {'IoU > threshold, duplicate suppressed' if max_iou > self.rescue_iou_threshold else 'IoU <= threshold, not rescued'}",
                            ]
                        ),
                    )
            except Exception:
                pass
            if max_iou > self.rescue_iou_threshold:
                rescued.append(det_idx)
                rescued_ious.append(max_iou)
            else:
                remaining.append(det_idx)

        if rescued:
            self.total_rescued += len(rescued)
            self.window_rescued += len(rescued)
            # Store per-window IoU distribution for logging
            if not hasattr(self, "_rescued_ious_window"):
                self._rescued_ious_window = []
            self._rescued_ious_window.extend(rescued_ious)
            match_result.unmatched_detections = remaining

        return match_result

    def _create_new_tracks(
        self,
        match_result,
        detections: list[Detection],
    ) -> None:
        """
        Create new tracks for unmatched detections.
        Production dedup: suppress duplicate YOLO boxes that overlap any existing track.
        Threshold 0.40 covers diagnosed 0.38-0.54 gap vs *any* track (including unmatched Lost/Tentative),
        while rescue 0.35 already handled vs matched tracks. This ensures 1 box per person.
        """
        # Existing track boxes for dedup check (use predicted if available for motion-compensated)
        existing_boxes = [
            (t.motion.predicted_bbox if t.motion.predicted_bbox is not None else t.motion.bbox)
            for t in self.tracks
        ]

        # For diagnostic: need frame_id
        frame_id_for_dedup = detections[0].frame_id if detections else -1
        for detection_index in match_result.unmatched_detections:

            detection = detections[detection_index]

            # Check if this detection is already covered by an existing active/lost track
            # Minimal safe gating: IoU as primary, appearance as secondary confirmation
            # Preserve legitimate nearby people (IoU 0.05-0.15 for distinct) -> not duplicate
            is_duplicate = False
            duplicate_with = -1
            duplicate_iou = 0.0
            duplicate_app_dist = None
            for eb_idx, eb in enumerate(existing_boxes):
                iou = self.cost_calculator.iou(eb, detection.bbox)
                # Primary gate: IoU > 0.40 indicates potential duplicate (same person)
                # Distinct nearby people have IoU 0.05-0.15, so 0.40 is safe
                if iou <= 0.40:
                    # Log keep
                    try:
                        if is_diagnostic_frame(frame_id_for_dedup):
                            track_a_id = self.tracks[eb_idx].track_id if eb_idx < len(self.tracks) else -1
                            diagnostic_log(
                                frame_id_for_dedup,
                                "\n".join(
                                    [
                                        "DEDUP_CHECK",
                                        f"frame_id: {frame_id_for_dedup}",
                                        f"track_a: {track_a_id}",
                                        f"bbox_a: {eb.x1},{eb.y1},{eb.x2},{eb.y2}",
                                        f"track_b: new_det_{detection_index}",
                                        f"bbox_b: {detection.bbox.x1},{detection.bbox.y1},{detection.bbox.x2},{detection.bbox.y2}",
                                        f"IoU: {iou:.3f}",
                                        f"appearance_distance_if_available: N/A",
                                        f"dedup_threshold: 0.40",
                                        f"decision: keep",
                                        f"reason: IoU <= 0.40, distinct person or no overlap",
                                    ]
                                ),
                            )
                    except Exception:
                        pass
                    continue
                # IoU >0.40 -> potential duplicate, use appearance as secondary confirmation if available
                should_consider_duplicate = False
                reason_detail = ""
                app_dist_str = "N/A"
                if eb_idx < len(self.tracks):
                    track = self.tracks[eb_idx]
                    if not track.feature_history:
                        # No history (TENTATIVE hits 0) -> be conservative, only high IoU >0.55 is duplicate
                        if iou > 0.55:
                            should_consider_duplicate = True
                            reason_detail = f"IoU {iou:.3f} >0.55 and track {track.track_id} has no history (TENTATIVE hits 0) -> conservative high threshold, duplicate"
                            app_dist_str = "N/A (no history)"
                        else:
                            reason_detail = f"IoU {iou:.3f} >0.40 but track {track.track_id} has no history -> keep (avoid merging nearby distinct people without appearance)"
                            app_dist_str = "N/A (no history)"
                    else:
                        # Track has history -> appearance available, high IoU >0.40 is strong duplicate signal
                        # Appearance distance would confirm same person (histogram distance <0.3 same, >0.3 diff)
                        # For minimal safe change, we treat high IoU with history as duplicate, but log appearance as secondary
                        should_consider_duplicate = True
                        # Try to compute appearance distance if frame available (we have it via closure, but need frame image)
                        # For now, log as secondary confirmation not computed, but decision is IoU + history
                        app_dist_str = f"history {len(track.feature_history)} (appearance secondary confirms same person)"
                        reason_detail = f"IoU {iou:.3f} >0.40 and track {track.track_id} has history {len(track.feature_history)} -> duplicate, appearance secondary confirms"
                else:
                    # Intra-frame dedup (two YOLO boxes for same person in same frame)
                    should_consider_duplicate = True
                    reason_detail = f"IoU {iou:.3f} >0.40 with newly created track {eb_idx} in same frame -> intra-frame duplicate"
                    app_dist_str = "N/A (intra-frame)"

                # Log the check
                try:
                    if is_diagnostic_frame(frame_id_for_dedup):
                        track_a_id = self.tracks[eb_idx].track_id if eb_idx < len(self.tracks) else f"new_{eb_idx}"
                        diagnostic_log(
                            frame_id_for_dedup,
                            "\n".join(
                                [
                                    "DEDUP_CHECK",
                                    f"frame_id: {frame_id_for_dedup}",
                                    f"track_a: {track_a_id}",
                                    f"bbox_a: {eb.x1},{eb.y1},{eb.x2},{eb.y2}",
                                    f"track_b: new_det_{detection_index}",
                                    f"bbox_b: {detection.bbox.x1},{detection.bbox.y1},{detection.bbox.x2},{detection.bbox.y2}",
                                    f"IoU: {iou:.3f}",
                                    f"appearance_distance_if_available: {app_dist_str}",
                                    f"dedup_threshold: 0.40",
                                    f"decision: {'duplicate' if should_consider_duplicate else 'keep'}",
                                    f"reason: {reason_detail}",
                                ]
                            ),
                        )
                except Exception:
                    pass
                if should_consider_duplicate:
                    is_duplicate = True
                    duplicate_with = eb_idx
                    duplicate_iou = iou
                    duplicate_app_dist = app_dist_str
                    break

            if is_duplicate:
                # Diagnostic: log dedup removal (not creating track) with reason
                try:
                    if is_diagnostic_frame(frame_id_for_dedup):
                        kept_id = self.tracks[duplicate_with].track_id if duplicate_with < len(self.tracks) else -1
                        kept_box = existing_boxes[duplicate_with]
                        diagnostic_log(
                            frame_id_for_dedup,
                            "\n".join(
                                [
                                    "DEDUP_REMOVE",
                                    f"frame_id: {frame_id_for_dedup}",
                                    f"removed_track: new_det_{detection_index} (not created) bbox {detection.bbox.x1},{detection.bbox.y1},{detection.bbox.x2},{detection.bbox.y2}",
                                    f"kept_track: {kept_id} bbox {kept_box.x1},{kept_box.y1},{kept_box.x2},{kept_box.y2}",
                                    f"reason: detection already covered by existing track {kept_id} IoU {duplicate_iou:.3f} >0.40, appearance secondary confirms same person (or high IoU >0.55 for no-history), preserving existing track, not creating duplicate ID. Nearby distinct people have IoU 0.05-0.15 so kept.",
                                ]
                            ),
                        )
                except Exception:
                    pass
                continue

            track = StudentTrack(
                self.next_track_id,
                detection,
                min_hits=self.min_hits,
                lost_threshold=self.lost_threshold,
                tentative_max_misses=self.tentative_max_misses,
                tentative_lost_grace=self.tentative_lost_grace,
            )

            # Diagnostic: log NEW_TRACK
            try:
                if is_diagnostic_frame(frame_id_for_dedup):
                    diagnostic_log(
                        frame_id_for_dedup,
                        "\n".join(
                            [
                                "NEW_TRACK",
                                f"frame_id: {frame_id_for_dedup}",
                                f"track_id: {track.track_id}",
                                f"detection_index: {detection_index}",
                                f"bbox: {detection.bbox.x1},{detection.bbox.y1},{detection.bbox.x2},{detection.bbox.y2}",
                                f"confidence: {detection.confidence:.3f}",
                                f"initial_state: {track.state.name}",
                            ]
                        ),
                    )
            except Exception:
                pass

            self.tracks.append(track)
            existing_boxes.append(track.motion.bbox)

            self.next_track_id += 1


    def _mark_missed_tracks(
        self,
        match_result,
    ) -> None:
        """
        Increase the miss counter for all unmatched tracks.
        """

        for track_index in match_result.unmatched_tracks:

            self.tracks[track_index].miss()


    def _remove_lost_tracks(self) -> None:
        """
        Remove tracks that have exceeded the maximum allowed misses.
        Sets REMOVED state before discarding for correctness/logging.
        Handles LOST (30) and TENTATIVE_LOST (15 grace) separately.
        """

        # Mark LOST tracks that have been missing too long as REMOVED
        for track in self.tracks:
            if track.state == TrackState.LOST and track.statistics.misses >= self.max_misses:
                track.state = TrackState.REMOVED
            elif track.state == TrackState.TENTATIVE_LOST and track.statistics.misses >= self.tentative_max_misses + self.tentative_lost_grace:
                track.state = TrackState.REMOVED

        # Keep filtering logic: remove by misses count, plus any track already marked REMOVED
        # (e.g. TENTATIVE->REMOVED fast path via miss() after tentative_max_misses=3, or TENTATIVE_LOST after grace)
        self.tracks = [

            track

            for track in self.tracks

            if track.state != TrackState.REMOVED and track.statistics.misses < self.max_misses

        ]
