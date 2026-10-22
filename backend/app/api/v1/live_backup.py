"""

Live Exam API - Phase 9

Real-time AI state from the REAL frozen pipeline (no synthetic events).

Pipeline per frame (frozen thresholds, no AI changes):

  VideoManager(OpenCVCapture) -> YOLODetector(0.20/0.65) -> TrackingManager

  -> BehaviorFusion (movement from tracker + pose_history when available,

     gaze/hands/objects explicitly unavailable) -> RuleEngine.evaluate

  -> ViolationService.register_event + EvidenceManager (background thread)

Frontend receives lightweight JSON (boxes/IDs/confidence/events);

video is delivered separately via /api/v1/video/samples/{id}.

"""

import asyncio

import math

import pathlib

import queue

import threading

import time

import uuid

from typing import Optional

import cv2

from fastapi import APIRouter, HTTPException

from fastapi.responses import FileResponse, StreamingResponse

from pydantic import BaseModel

from app.ai.behavior.fusion import BehaviorFusion

from app.ai.behavior.schemas import BodyPoseObservation, HeadPoseObservation, MovementObservation

from app.ai.detector.manager import DetectionManager

from app.ai.detector.yolo import YOLODetector

from app.ai.tracker.manager import TrackingManager

from app.ai.video.capture import OpenCVCapture

from app.ai.video.manager import VideoManager

from app.ai.video.rolling_buffer import RollingOriginalBuffer

from app.api.v1.live_overlay import count_active_violations

from app.core.logger import app_logger

from app.services.evidence_manager import EvidenceManager

from app.config.settings import settings

from app.services.rule_engine import EventStatus, RuleEngine

from app.services.temporal_history import TemporalHistory

from app.services.violation_service import violation_service

router = APIRouter(prefix="/live", tags=["Live Exam"])

try:

    _proj_root = pathlib.Path(__file__).resolve().parents[4]

    _cand = _proj_root / "sample_clips"

    SAMPLE_DIR = _cand if _cand.exists() else pathlib.Path("sample_clips")

except Exception:

    SAMPLE_DIR = pathlib.Path("sample_clips")

live_sessions: dict[str, dict] = {}

live_lock = threading.Lock()

def _get_live_session(session_id: str):

    with live_lock:

        return live_sessions.get(session_id)

class LiveSessionStartRequest(BaseModel):

    video_id: str

    session_name: Optional[str] = None

    max_frames: Optional[int] = None

class LiveSessionStartResponse(BaseModel):

    session_id: str

    stream_url: str

    state_url: str

    video_url: str

    processed_video_url: str

    total_frames: Optional[int] = None

    message: str

class StudentTrackState(BaseModel):

    track_id: int

    bbox: list[float]

    confidence: float

    state: str

    hits: int

    misses: int

    age: int

    pose: Optional[dict] = None

    reid_embedding: bool = False

    velocity: Optional[dict] = None

    face_present: bool = False

    eye_state: str = "unknown"

    body_movement: Optional[float] = None

    movement_state: str = "unknown"

    head_direction: str = "UNKNOWN"

    head_yaw: Optional[float] = None

    head_pitch: Optional[float] = None

    head_roll: Optional[float] = None

    head_confidence: Optional[float] = None

    head_duration: float = 0.0

    left_wrist_status: str = "UNKNOWN"

    left_wrist_pos: Optional[list[float]] = None

    right_wrist_status: str = "UNKNOWN"

    right_wrist_pos: Optional[list[float]] = None

    desk_id: Optional[int] = None

    desk_rule_status: str = "UNKNOWN"

    desk_wrist_duration: float = 0.0

    violation_active: bool = False

    active_violations: list[str] = []

class ViolationEvent(BaseModel):

    event_id: str

    student_id: int

    rule_id: str

    event_type: str

    severity: str

    status: str

    confidence: float

    start_time: float

    end_time: Optional[float] = None

    timestamp: float

    reason: str

    supporting_features: dict = {}

class LiveSessionStatus(BaseModel):

    session_id: str

    video_id: str

    state: str

    frame_id: int

    timestamp: float

    fps: float

    processing_fps: float

    students: list[StudentTrackState] = []

    violations: list[ViolationEvent] = []

    video_width: int = 0

    video_height: int = 0

    processing_latency_ms: float = 0.0

    frames_processed: int = 0

    total_frames: Optional[int] = None

class LiveAIState(BaseModel):

    frame_id: int

    timestamp: float

    fps: float

    processing_fps: float

    students: list[StudentTrackState] = []

    violations: list[ViolationEvent] = []

    video_width: int = 0

    video_height: int = 0

    processing_latency_ms: float = 0.0

def _event_to_violation(ev) -> dict:

    sev = getattr(ev.severity, "value", str(ev.severity)) if getattr(ev, "severity", None) is not None else "medium"

    st = getattr(ev.status, "value", str(ev.status)) if getattr(ev, "status", None) is not None else "active"

    reason = ""

    try:

        if isinstance(getattr(ev, "evidence", None), dict):

            reason = ev.evidence.get("reason", "") or ""

    except Exception:

        pass

    supporting = {}

    try:

        if isinstance(getattr(ev, "evidence", None), dict):

            supporting = ev.evidence.get("supporting_features", {}) or {}

    except Exception:

        pass

    start_time = float(getattr(ev, "timestamp", time.time()))

    try:

        from app.services.evidence_manager import _get_event_interval

        s, e = _get_event_interval(ev)

        start_time = float(s)

        end_time = float(e)

    except Exception:

        end_time = None

    return {

        "event_id": str(ev.event_id),

        "student_id": int(ev.student_id),

        "rule_id": str(ev.rule_id),

        "event_type": str(ev.event_type),

        "severity": str(sev),

        "status": str(st),

        "confidence": float(getattr(ev, "confidence", 0.0) or 0.0),

        "start_time": start_time,

        "end_time": end_time,

        "timestamp": float(getattr(ev, "timestamp", start_time)),

        "reason": str(reason or f"{ev.rule_id} for StudentTrack #{ev.student_id}"),

        "supporting_features": dict(supporting or {}),

    }

def _trigger_evidence_async(buffer: RollingOriginalBuffer, event, session_id: str, source_id: str, student_snapshot: Optional[dict] = None):

    def _run():

        try:

            # Wait for the requested amount of media time, which may take longer

            # than wall-clock post_seconds when inference is slower than the clip.

            mgr = EvidenceManager(buffer, base_output_dir="clips/evidence", wait_timeout_seconds=120.0)

            mgr.create_evidence(

                event,

                session_id=session_id,

                source_id=source_id,

                wait_for_post=True,

                student_snapshot=student_snapshot,

                crop_student_id=int(getattr(event, "student_id")),

                crop_padding_ratio=0.25,

            )

        except Exception as e:

            app_logger.warning(f"Live evidence failed for {event.event_id}: {e}")

    t = threading.Thread(target=_run, daemon=True)

    t.start()

def _live_worker(clip_path: pathlib.Path, session_id: str, max_frames: Optional[int], video_id: str):

    sess = _get_live_session(session_id)

    if not sess:

        return

    evidence_buffer = RollingOriginalBuffer(retention_seconds=settings.EVIDENCE_BUFFER_SECONDS)

    video = VideoManager(

        OpenCVCapture(str(clip_path)), source_id=video_id,

        evidence_buffer=evidence_buffer, preserve_all_frames=True,

        queue_size=2,

    )

    detector = DetectionManager(YOLODetector())

    tracker = TrackingManager()

    rule_engine = RuleEngine()

    fusion = BehaviorFusion()

    temporal = TemporalHistory(max_size=10, max_duration_seconds=10.0)

    if not video.start():

        with live_lock:

            if session_id in live_sessions:

                live_sessions[session_id]["state"] = "error"

                live_sessions[session_id]["error"] = "Failed to open video"

                live_sessions[session_id]["running"] = False

        return

    try:

        detector.start()

    except Exception as e:

        with live_lock:

            if session_id in live_sessions:

                live_sessions[session_id]["state"] = "error"

                live_sessions[session_id]["error"] = f"Detector start failed: {e}"

                live_sessions[session_id]["running"] = False

        try:

            video.stop()

        except Exception:

            pass

        return

    time.sleep(0.2)

    with live_lock:

        cur = live_sessions.get(session_id)

        can_run = bool(cur and cur.get("running", False))

        if can_run and cur is not None:

            live_sessions[session_id]["state"] = "running"

            live_sessions[session_id]["running"] = True

    if not can_run:

        try:

            video.stop()

        except Exception:

            pass

        try:

            detector.stop()

        except Exception:

            pass

        try:

            evidence_buffer.mark_complete()

        except Exception:

            pass

        with live_lock:

            cur = live_sessions.get(session_id)

            if cur is not None:

                cur["state"] = "stopped"

                cur["running"] = False

                cur["completed"] = True

        return

    prev_centers: dict[int, tuple[float, float, float]] = {}

    frames_ok = 0

    pending_frame_detection = None

    annotated_writer = None

    annotated_width = 0

    annotated_height = 0

    annotated_path = pathlib.Path(sess["processed_video_path"])

    # Evidence is finalized only after an ACTIVE event ends.
    event_snapshots: dict[str, dict] = {}
    finalized_event_ids: set[str] = set()
    evidence_primary_rules = {"HANDS_OFF_DESK", "PEEPING", "ITEM_EXCHANGE", "PHONE_DETECTED"}

    def _finalize_completed_events(completed_events) -> None:
        for completed in completed_events:
            try:
                event_id = str(completed.event_id)
            except Exception:
                continue
            if event_id in finalized_event_ids:
                continue
            finalized_event_ids.add(event_id)

            try:
                violation_service.update_event(completed, session_id=session_id)
            except Exception as exc:
                app_logger.warning(f"Could not update completed violation {event_id}: {exc}")

            rule_id = str(getattr(completed, "rule_id", ""))
            if rule_id not in evidence_primary_rules:
                continue

            snapshot = event_snapshots.get(event_id)
            student_id = int(getattr(completed, "student_id", -1))
            if student_id < 0:
                snapshot = snapshot or {"student_id": student_id, "state": "scene"}

            _trigger_evidence_async(
                evidence_buffer,
                completed,
                session_id,
                video_id,
                snapshot,
            )

    try:

        while True:

            with live_lock:

                cur = live_sessions.get(session_id)

                if not cur or not cur.get("running", True):

                    break

            if pending_frame_detection is not None:

                f, dets, inference_ms = pending_frame_detection

                pending_frame_detection = None

            else:

                f = video.read()

                if f is None:

                    try:

                        if not video.is_running():

                            break

                    except Exception:

                        break

                    time.sleep(0.005)

                    continue

                # Fill a two-frame inference batch, waiting briefly for the capture

                # thread while preserving all frames and their original order.

                batch_frames = [f]

                batch_limit = min(2, max_frames - frames_ok) if max_frames is not None else 2

                while len(batch_frames) < batch_limit:

                    with live_lock:

                        cur = live_sessions.get(session_id)

                        if not cur or not cur.get("running", True):

                            break

                    next_frame = video.read()

                    if next_frame is not None:

                        batch_frames.append(next_frame)

                    elif not video.is_running():

                        break

                    else:

                        time.sleep(0.002)

                t0 = time.perf_counter()

                try:

                    batch_detections = detector.detect_batch(batch_frames)

                except Exception as e:

                    app_logger.warning(f"Live detect failed frame {frames_ok}: {e}")

                    continue

                batch_inference_ms = (time.perf_counter() - t0) * 1000.0

                inference_per_frame_ms = batch_inference_ms / max(1, len(batch_frames))

                dets = batch_detections[0] if batch_detections else []

                if len(batch_frames) > 1:

                    pending_frame_detection = (batch_frames[1], batch_detections[1], inference_per_frame_ms)

                inference_ms = inference_per_frame_ms

            try:

                tracks = tracker.update(dets, frame=f)

            except Exception as e:

                app_logger.warning(f"Live track failed frame {frames_ok}: {e}")

                continue

            try:

                student_boxes = {}

                for track in tracks:

                    box = getattr(getattr(track, "motion", None), "bbox", None)

                    if box is not None:

                        student_boxes[str(int(track.track_id))] = [float(box.x1), float(box.y1), float(box.x2), float(box.y2)]

                evidence_buffer.update_frame_metadata(int(f.frame_id), {"student_boxes": student_boxes})

            except Exception as e:

                app_logger.debug(f"Could not attach track boxes to evidence frame {f.frame_id}: {e}")

            inference_ms = (time.perf_counter() - t0) * 1000.0

            frames_ok += 1

            students: list[dict] = []

            frame_violations: list[dict] = []

            try:

                fps_val = float(getattr(f, "fps", 25.0) or 25.0)

            except Exception:

                fps_val = 25.0

            # --- Phase 1: Build beh_states for all tracks (needed for ITEM_EXCHANGE which checks other students) ---

            _frame_beh_states: list = []

            _beh_by_id: dict[int, object] = {}

            _pre_collect: list[tuple] = []  # (t, bb, conf, state_name, head_dir, head_yaw, head_conf, lw_kpt, rw_kpt, body_pose_obs, head_pose_obs, movement_obs, beh_state)

            # We will collect in first pass, then evaluate in second pass

            for t in tracks:

                try:

                    if t.state.name == "REMOVED":

                        continue

                except Exception:

                    continue

                try:

                    bb = t.motion.bbox

                    x1, y1, x2, y2 = float(bb.x1), float(bb.y1), float(bb.x2), float(bb.y2)

                    conf = float(getattr(t, "confidence", 0.0) or 0.0)

                    state_name = t.state.name.lower() if hasattr(t.state, "name") else str(t.state).lower()

                    # head direction via YOLO keypoints (cheap geometry)

                    head_dir = "UNKNOWN"

                    head_yaw_val = None

                    head_pitch_val = None

                    head_roll_val = None

                    head_conf_val = None

                    head_detected = False

                    pose_dict = None

                    try:

                        if getattr(t, "pose_history", None) and len(t.pose_history) > 0:

                            lp = t.pose_history[-1]

                            lp_dict = lp if isinstance(lp, dict) else {

                                "yaw": float(getattr(lp, "yaw", 0.0)),

                                "pitch": float(getattr(lp, "pitch", 0.0)),

                                "roll": float(getattr(lp, "roll", 0.0)),

                                "confidence": float(getattr(lp, "confidence", 0.0)),

                                "direction": getattr(lp, "direction", "UNKNOWN"),

                            }

                            pose_is_fresh = int(lp_dict.get("frame_id", -1)) == int(f.frame_id)

                            head_yaw_val = float(lp_dict.get("yaw", 0.0))

                            head_pitch_val = float(lp_dict.get("pitch", 0.0))

                            head_roll_val = float(lp_dict.get("roll", 0.0))

                            head_conf_val = float(lp_dict.get("confidence", 0.0))

                            stored_dir = lp_dict.get("direction", "UNKNOWN")

                            if pose_is_fresh and stored_dir and stored_dir != "UNKNOWN":

                                head_dir = stored_dir

                                pose_dict = {"yaw": head_yaw_val, "pitch": 0.0, "roll": 0.0, "confidence": head_conf_val, "direction": head_dir}

                                head_detected = True

                        if not head_detected:

                            det_kpts = None

                            if hasattr(t, "last_detection") and t.last_detection and t.last_detection.frame_id == f.frame_id and getattr(t.last_detection, "metadata", None):

                                det_kpts = t.last_detection.metadata.get("keypoints")

                            if det_kpts is not None:

                                from app.ai.tracker.head_pose import classify_head_direction, HeadPoseEstimator

                                d, c = classify_head_direction(det_kpts)

                                if d != "UNKNOWN":

                                    head_dir = d

                                    head_conf_val = float(c)

                                    try:

                                        yaw_e, _, _, _, _, _ = HeadPoseEstimator().estimate_from_keypoints(det_kpts)

                                        head_yaw_val = float(yaw_e)

                                    except Exception:

                                        head_yaw_val = 0.0

                                    head_pitch_val = 0.0

                                    head_roll_val = 0.0

                                    pose_dict = {"yaw": head_yaw_val, "pitch": 0.0, "roll": 0.0, "confidence": head_conf_val, "direction": head_dir}

                                    head_detected = True

                    except Exception:

                        pass

                    lw_kpt = None

                    rw_kpt = None

                    try:

                        det_kpts = None

                        if hasattr(t, "last_detection") and t.last_detection and getattr(t.last_detection, "metadata", None):

                            det_kpts = t.last_detection.metadata.get("keypoints")

                        if det_kpts is not None and len(det_kpts) >= 11:

                            lw_kpt = (float(det_kpts[9][0]), float(det_kpts[9][1]), float(det_kpts[9][2]))

                            rw_kpt = (float(det_kpts[10][0]), float(det_kpts[10][1]), float(det_kpts[10][2]))

                    except Exception:

                        pass

                    # Build observations

                    head_pose_obs = HeadPoseObservation(

                        available=head_detected,

                        yaw=head_yaw_val,

                        pitch=head_pitch_val,

                        roll=head_roll_val,

                        direction=head_dir,

                        confidence=head_conf_val,

                        frame_id=int(f.frame_id),

                        timestamp=float(f.timestamp),

                    )

                    body_pose_obs = BodyPoseObservation(

                        available=(lw_kpt is not None or rw_kpt is not None),

                        left_wrist=lw_kpt,

                        right_wrist=rw_kpt,

                        frame_id=int(f.frame_id),

                        timestamp=float(f.timestamp),

                    )

                    # movement

                    try:

                        vx = float(t.motion.velocity_x)

                        vy = float(t.motion.velocity_y)

                    except Exception:

                        vx, vy = 0.0, 0.0

                    # body_movement via prev_centers (reuse outer prev_centers if available)

                    # For pre-collect we need body_movement placeholder, will compute later

                    movement_obs = MovementObservation(

                        available=False,

                        body_movement=None,

                        velocity_x=vx,

                        velocity_y=vy,

                        frame_id=int(f.frame_id),

                        timestamp=float(f.timestamp),

                    )

                    try:

                        beh_obs = fusion.fuse(

                            student_id=int(t.track_id),

                            frame_id=int(f.frame_id),

                            timestamp=float(f.timestamp),

                            head_pose=head_pose_obs,

                            body_pose=body_pose_obs,

                            movement=movement_obs,

                            track_bbox=bb,

                            confidence=conf,

                        )

                        beh_state = fusion.create_behavior_state(

                            student_id=int(t.track_id),

                            frame_id=int(f.frame_id),

                            timestamp=float(f.timestamp),

                            behavior_obs=beh_obs,

                            movement_features={},

                            track_bbox=bb,

                        )

                    except Exception as e:

                        app_logger.warning(f"Fusion failed track {t.track_id}: {e}")

                        beh_state = None

                    if beh_state is not None:

                        _frame_beh_states.append(beh_state)

                        _beh_by_id[int(t.track_id)] = beh_state

                    # stash for second pass (we will reuse head_dir etc)

                    _pre_collect.append((t, bb, conf, state_name, head_dir, head_yaw_val, head_pitch_val, head_roll_val, head_conf_val, head_detected, pose_dict, lw_kpt, rw_kpt, beh_state))

                except Exception as e:

                    app_logger.warning(f"Live pre-collect failed {getattr(t, 'track_id', '?')}: {e}")

                    continue

            # --- Phase 2: Evaluate rule engine per student with full frame context ---

            for (t, bb, conf, state_name, head_dir, head_yaw_val, head_pitch_val, head_roll_val, head_conf_val, head_detected, pose_dict, lw_kpt, rw_kpt, beh_state) in _pre_collect:

                # Re-extract per-track vars needed for HUD (x1,y1 etc already in bb)

                # Need x1,y1,x2,y2 for output

                try:

                    x1, y1, x2, y2 = float(bb.x1), float(bb.y1), float(bb.x2), float(bb.y2)

                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                    hits = int(t.statistics.hits) if hasattr(t.statistics, "hits") else 0

                    misses = int(t.statistics.misses) if hasattr(t.statistics, "misses") else 0

                    age = int(t.statistics.age) if hasattr(t.statistics, "age") else 0

                    try:

                        vx = float(t.motion.velocity_x)

                        vy = float(t.motion.velocity_y)

                    except Exception:

                        vx, vy = 0.0, 0.0

                    # body_movement calc (now with prev_centers)

                    body_movement = None

                    try:

                        prev = prev_centers.get(t.track_id)

                        if prev is not None:

                            px, py, pts = prev

                            dt = float(f.timestamp) - float(pts)

                            if dt > 0:

                                dist = math.hypot(cx - px, cy - py)

                                body_movement = float(dist / dt)

                    except Exception:

                        body_movement = None

                    try:

                        prev_centers[t.track_id] = (cx, cy, float(f.timestamp))

                    except Exception:

                        pass

                    # Update movement in beh_state if exists

                    if beh_state is not None and body_movement is not None:

                        beh_state.body_movement = body_movement

                        if body_movement > 50:

                            beh_state.movement_state = "moving"

                        elif body_movement < 5:

                            beh_state.movement_state = "stationary"

                        else:

                            beh_state.movement_state = "normal"

                    reid_emb = bool(getattr(t, "feature_history", None))

                    track_active_events: list[str] = []

                    if beh_state is not None:

                        try:

                            temporal.append(int(t.track_id), "movement", MovementObservation(available=body_movement is not None, body_movement=body_movement, velocity_x=vx, velocity_y=vy, frame_id=int(f.frame_id), timestamp=float(f.timestamp)), timestamp=float(f.timestamp))

                        except Exception:

                            pass

                        if state_name == "confirmed":

                            try:

                                # Pass full frame beh_states for ITEM_EXCHANGE, phones=[] for now (phone detector disabled)

                                events = rule_engine.evaluate(

                                    beh_state,

                                    timestamp=float(f.timestamp),

                                    frame_id=int(f.frame_id),

                                    students=_frame_beh_states,

                                    phones=[],

                                    frame_width=int(f.image.shape[1]) if f.image is not None else None,

                                    frame_height=int(f.image.shape[0]) if f.image is not None else None,

                                )

                            except Exception as e:

                                app_logger.warning(f"RuleEngine failed track {t.track_id}: {e}")

                                events = []

                            for ev in events:

                                try:

                                    ev_status = getattr(ev.status, "value", str(ev.status))

                                except Exception:

                                    ev_status = "active"

                                if ev_status not in ("confirmed", "active", "ended"):

                                    # Also allow YELLOW/RED style - our new engine uses confirmed/active

                                    if ev_status.lower() not in ("confirmed", "active", "ended", "yellow", "red"):

                                        continue

                                try:

                                    registered = violation_service.register_event(ev, session_id=session_id)

                                except Exception:

                                    registered = False

                                vdict = _event_to_violation(ev)

                                frame_violations.append(vdict)

                                if registered and ev_status == "active":

                                    snapshot = {

                                        "student_id": int(t.track_id),

                                        "state": state_name,

                                        "head_direction": head_dir or "UNKNOWN",

                                        "head_yaw": head_yaw_val,

                                        "movement_state": getattr(beh_state, "movement_state", "unknown") or "unknown",

                                        "left_wrist_status": getattr(beh_state, "left_wrist_status", "UNKNOWN") or "UNKNOWN",

                                        "right_wrist_status": getattr(beh_state, "right_wrist_status", "UNKNOWN") or "UNKNOWN",

                                        "active_violations": [str(active.rule_id) for active in events],

                                    }

                                    event_snapshots[str(ev.event_id)] = snapshot

                    # Check active events

                    try:

                        for act_ev in rule_engine.get_active_events():

                            if act_ev.student_id == int(t.track_id):

                                track_active_events.append(str(act_ev.rule_id))

                    except Exception:

                        pass

                    violation_active = len(track_active_events) > 0

                    head_dur = 0.0

                    try:

                        # Use rule results duration (monotonic-correct) instead of wall-clock minus monotonic

                        res_tmp = rule_engine.get_student_result(int(t.track_id))

                        if res_tmp is not None:

                            head_dur = float(res_tmp.peeping.duration or 0.0)

                        else:

                            r_state = rule_engine._get_rule_state(int(t.track_id), f"HEAD_TURN_{head_dir}")

                            if r_state and r_state.candidate_start:

                                # monotonic candidate_start, use monotonic now

                                import time as _t

                                head_dur = max(0.0, _t.monotonic() - r_state.candidate_start)

                    except Exception:

                        pass

                    left_w_dur = 0.0

                    right_w_dur = 0.0

                    try:

                        res_tmp2 = rule_engine.get_student_result(int(t.track_id))

                        if res_tmp2 is not None:

                            left_w_dur = float(res_tmp2.hands_off_desk.duration or 0.0)

                            right_w_dur = left_w_dur

                        else:

                            w_state = rule_engine._get_rule_state(int(t.track_id), "WRIST_OUTSIDE_REGION")

                            if w_state and w_state.candidate_start:

                                import time as _t2

                                left_w_dur = max(0.0, _t2.monotonic() - w_state.candidate_start)

                                right_w_dur = left_w_dur

                            else:

                                w2 = rule_engine._get_rule_state(int(t.track_id), "HANDS_OFF_DESK")

                                if w2 and w2.candidate_start:

                                    import time as _t3

                                    left_w_dur = max(0.0, _t3.monotonic() - w2.candidate_start)

                                    right_w_dur = left_w_dur

                    except Exception:

                        pass

                    lw_status = beh_state.left_wrist_status if beh_state else "UNKNOWN"

                    rw_status = beh_state.right_wrist_status if beh_state else "UNKNOWN"

                    desk_id_value = rule_engine.get_student_desk_id(int(t.track_id))

                    desk_status_value = rule_engine.get_student_desk_status(int(t.track_id))

                    students.append({

                        "track_id": int(t.track_id),

                        "bbox": [x1, y1, x2, y2],

                        "confidence": conf,

                        "state": state_name,

                        "hits": hits,

                        "misses": misses,

                        "age": age,

                        "pose": pose_dict,

                        "reid_embedding": reid_emb,

                        "velocity": {"x": vx, "y": vy},

                        "face_present": head_detected,

                        "eye_state": "unknown",

                        "body_movement": body_movement,

                        "movement_state": "moving" if (body_movement is not None and body_movement > 50) else ("stationary" if (body_movement is not None and body_movement < 5) else "unknown"),

                        "head_direction": head_dir,

                        "head_yaw": head_yaw_val,

                        "head_pitch": head_pitch_val,

                        "head_roll": head_roll_val,

                        "head_confidence": head_conf_val,

                        "head_duration": head_dur,

                        "left_wrist_status": lw_status,

                        "left_wrist_pos": list(lw_kpt) if lw_kpt else None,

                        "left_wrist_duration": left_w_dur,

                        "right_wrist_status": rw_status,

                        "right_wrist_pos": list(rw_kpt) if rw_kpt else None,

                        "right_wrist_duration": right_w_dur,

                        "desk_id": desk_id_value,

                        "desk_rule_status": desk_status_value,

                        "desk_wrist_duration": left_w_dur,

                        "violation_active": violation_active,

                        "active_violations": track_active_events,

                    })

                except Exception as e:

                    app_logger.warning(f"Live per-track failed {getattr(t, 'track_id', '?')}: {e}")

                    continue

            # Finalize events that returned to GREEN during this frame.

            try:

                _finalize_completed_events(rule_engine.get_completed_events())

            except Exception as exc:

                app_logger.warning(f"Could not finalize completed live events: {exc}")

            # --- End two-phase loop, HUD rendering continues ---

            # Skip original single loop - we have already populated students, so continue to with live_lock block

            # Need to handle the case where tracks was empty but we still need to update session

            with live_lock:

                cur = live_sessions.get(session_id)

                if cur is not None:

                    cur["students"] = students

                    if frame_violations:

                        # Keep the in-memory live feed bounded. All events are still

                        # persisted by the violation service for the evidence library.

                        cur["violations"] = (frame_violations + cur.get("violations", []))[:100]

                    cur["frame_id"] = int(f.frame_id)

                    cur["frames_processed"] = frames_ok

                    cur["timestamp"] = float(f.timestamp)

                    cur["fps"] = fps_val

                    try:

                        cur["processing_fps"] = 1000.0 / max(1.0, inference_ms)

                    except Exception:

                        cur["processing_fps"] = 0.0

                    cur["processing_latency_ms"] = float(inference_ms)

                    try:

                        if f.image is not None:

                            cur["video_width"] = int(f.image.shape[1])

                            cur["video_height"] = int(f.image.shape[0])

                    except Exception:

                        pass

            # Push Live Annotated JPEG for frontend MJPEG stream (red boxes for violations, head direction, HUD)

            try:

                with live_lock:

                    cur = live_sessions.get(session_id)

                    q = cur.get("queue") if cur else None

                if q is not None and f.image is not None:

                    disp = f.image.copy()

                    h_img, w_img = disp.shape[:2]

                    # Draw per-student bounding boxes, ID badges, head direction, wrists, and malpractice banners

                    for student_info in students:

                        s_id = student_info.get("track_id", "?")

                        bbox = student_info.get("bbox", [0, 0, 0, 0])

                        bx1, by1, bx2, by2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                        s_state = student_info.get("state", "CONFIRMED")

                        has_violation = student_info.get("violation_active", False)

                        active_viols = student_info.get("active_violations", [])

                        # Color: BOLD RED for active violation, GREEN for confirmed, YELLOW for tentative

                        if has_violation:

                            box_color = (0, 0, 255)     # RED

                            thickness = 3

                        elif s_state == "CONFIRMED":

                            box_color = (0, 255, 0)     # GREEN

                            thickness = 2

                        else:

                            box_color = (0, 255, 255)   # YELLOW

                            thickness = 1

                        # Draw Bounding Box

                        cv2.rectangle(disp, (bx1, by1), (bx2, by2), box_color, thickness)

                        # Draw Student ID Header Box

                        id_text = f"Student #{s_id}"

                        cv2.rectangle(disp, (bx1, max(0, by1 - 25)), (bx1 + 110, by1), box_color, -1)

                        text_color = (255, 255, 255) if box_color == (0, 0, 255) else (0, 0, 0)

                        cv2.putText(disp, id_text, (bx1 + 5, max(12, by1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

                        # Draw Head Direction Badge

                        head_dir = student_info.get("head_direction", "CENTER")

                        head_yaw = student_info.get("head_yaw", None)

                        if head_dir and head_dir != "UNKNOWN":

                            yaw_str = f" ({head_yaw:+.1f}deg)" if head_yaw is not None else ""

                            head_label = f"HEAD: {head_dir}{yaw_str}"

                            dir_color = (255, 100, 0) if head_dir == "LEFT" else (0, 100, 255) if head_dir == "RIGHT" else (0, 220, 0)

                            cv2.putText(disp, head_label, (bx1, max(30, by1 - 32)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, dir_color, 2, cv2.LINE_AA)

                        # Draw Wrists if positions available

                        lw_pos = student_info.get("left_wrist_pos")

                        rw_pos = student_info.get("right_wrist_pos")

                        if lw_pos and len(lw_pos) >= 2:

                            wx, wy = int(lw_pos[0]), int(lw_pos[1])

                            cv2.circle(disp, (wx, wy), 6, (255, 255, 255), -1)

                            cv2.circle(disp, (wx, wy), 4, (255, 0, 255), -1)

                        if rw_pos and len(rw_pos) >= 2:

                            wx, wy = int(rw_pos[0]), int(rw_pos[1])

                            cv2.circle(disp, (wx, wy), 6, (255, 255, 255), -1)

                            cv2.circle(disp, (wx, wy), 4, (0, 255, 255), -1)

                        # Draw Active Malpractice Banner under box

                        if has_violation:

                            viols_to_draw = active_viols if active_viols else ["MALPRACTICE DETECTED"]

                            for idx, ev in enumerate(viols_to_draw):

                                if isinstance(ev, str):

                                    e_type = ev

                                elif isinstance(ev, dict):

                                    e_type = ev.get("event_type") or ev.get("rule_id") or "MALPRACTICE"

                                else:

                                    e_type = getattr(ev, "event_type", getattr(ev, "rule_id", "MALPRACTICE"))

                                e_type_clean = str(e_type).replace("_", " ").upper()

                                viol_text = f"MALPRACTICE: {e_type_clean}"

                                banner_y = min(h_img - 10, by2 + 20 + idx * 22)

                                t_size = cv2.getTextSize(viol_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

                                cv2.rectangle(disp, (bx1, banner_y - 15), (bx1 + t_size[0] + 10, banner_y + 5), (0, 0, 200), -1)

                                cv2.putText(disp, viol_text, (bx1 + 5, banner_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    # Render Top HUD Bar (matching demo visualizer)

                    hud_h = 55

                    overlay = disp.copy()

                    cv2.rectangle(overlay, (0, 0), (w_img, hud_h), (20, 20, 20), -1)

                    disp = cv2.addWeighted(overlay, 0.75, disp, 0.25, 0)

                    cv2.line(disp, (0, hud_h), (w_img, hud_h), (0, 255, 255), 1)

                    # Derive this from the same data used for the boxes and banners.

                    all_active_cnt = count_active_violations(students)

                    cv2.putText(disp, "ExamInvigilationAI — LIVE AI STREAM", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.putText(disp, f"Frame: {f.frame_id} | FPS: {fps_val:.1f}", (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

                    conf_count = len([s for s in students if s.get("state") == "CONFIRMED"])

                    cv2.putText(disp, f"Students Tracked: {conf_count}", (340, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                    cv2.putText(

                        disp,

                        f"Rule Engine: ACTIVE (Head {settings.HEAD_TURN_DURATION_SECONDS:g}s / Desk {settings.WRIST_OUTSIDE_WARNING_SECONDS:g}-{settings.WRIST_OUTSIDE_CONFIRM_SECONDS:g}s)",

                        (340, 46),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,

                    )

                    viol_bg_col = (0, 0, 180) if all_active_cnt > 0 else (0, 120, 0)

                    cv2.rectangle(disp, (w_img - 280, 8), (w_img - 15, 48), viol_bg_col, -1)

                    cv2.rectangle(disp, (w_img - 280, 8), (w_img - 15, 48), (255, 255, 255), 1)

                    cv2.putText(disp, f"ACTIVE VIOLATIONS: {all_active_cnt}", (w_img - 270, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

                    # Record every annotated full-resolution frame into one browser-playable

                    # video. The live JSON/status path remains lightweight; the final video

                    # becomes available after the encoder is cleanly finalized.

                    if annotated_writer is None:

                        annotated_height, annotated_width = h_img - (h_img % 2), w_img - (w_img % 2)

                        try:

                            writer_fps = float(fps_val)

                        except (TypeError, ValueError):

                            writer_fps = 25.0

                        if not 1.0 <= writer_fps <= 120.0:

                            writer_fps = 25.0

                        annotated_path.parent.mkdir(parents=True, exist_ok=True)

                        for codec in ("avc1", "mp4v"):

                            writer = cv2.VideoWriter(

                                str(annotated_path), cv2.VideoWriter_fourcc(*codec), writer_fps,

                                (annotated_width, annotated_height),

                            )

                            if writer.isOpened():

                                annotated_writer = writer

                                break

                            writer.release()

                        if annotated_writer is None:

                            raise RuntimeError("Could not initialize an MP4 encoder (tried H.264 and MPEG-4)")

                        with live_lock:

                            cur = live_sessions.get(session_id)

                            if cur is not None:

                                cur["processed_video_codec"] = codec

                    encoded_frame = disp

                    if encoded_frame.shape[1] != annotated_width or encoded_frame.shape[0] != annotated_height:

                        encoded_frame = cv2.resize(encoded_frame, (annotated_width, annotated_height))

                    annotated_writer.write(encoded_frame)

                    # Keep the live inspection stream at the annotated source

                    # resolution. The previous 960px cap made student IDs,

                    # bounding boxes, pose keypoints, and violation text too

                    # small to verify manually in the browser.

                    #

                    # We deliberately do NOT resize `disp` here. The full-

                    # resolution annotated frame has already been produced and

                    # is also used by the processed-video writer above.

                    # JPEG quality 92 gives substantially clearer text/edges

                    # while remaining practical for a local manual-review UI.

                    ok, jpeg = cv2.imencode(

                        ".jpg",

                        disp,

                        [int(cv2.IMWRITE_JPEG_QUALITY), 92],

                    )

                    if ok:

                        try:

                            q.put_nowait(jpeg.tobytes())

                        except queue.Full:

                            try:

                                q.get_nowait()

                            except Exception:

                                pass

                            try:

                                q.put_nowait(jpeg.tobytes())

                            except Exception:

                                pass

            except Exception as e:

                app_logger.warning(f"Live JPEG push failed: {e}")

            if max_frames is not None and frames_ok >= max_frames:

                break

            time.sleep(0.005)

    except Exception as e:

        app_logger.exception(f"Live worker error: {e}")

        with live_lock:

            if session_id in live_sessions:

                live_sessions[session_id]["state"] = "error"

                live_sessions[session_id]["error"] = str(e)

                live_sessions[session_id]["running"] = False

    finally:

        if annotated_writer is not None:

            try:

                annotated_writer.release()

                with live_lock:

                    cur = live_sessions.get(session_id)

                    if cur is not None:

                        cur["processed_video_ready"] = annotated_path.exists() and annotated_path.stat().st_size > 0

            except Exception as e:

                app_logger.warning(f"Could not finalize annotated video for {session_id}: {e}")

        # If the source ends while a violation is still active, close the interval

        # at the last processed frame so its evidence is not lost.

        try:

            last_timestamp = float(sess.get("timestamp", time.time()))

            last_frame = int(sess.get("frame_id", 0))

            rule_engine.finalize_active_events(last_timestamp, last_frame)

            _finalize_completed_events(rule_engine.get_completed_events())

        except Exception as exc:

            app_logger.warning(f"Could not finalize active events at session end: {exc}")

        try:

            video.stop()

        except Exception:

            pass

        try:

            detector.stop()

        except Exception:

            pass

        try:

            evidence_buffer.mark_complete()

        except Exception:

            pass

        with live_lock:

            if session_id in live_sessions:

                if live_sessions[session_id].get("state") == "stopping":

                    live_sessions[session_id]["state"] = "stopped"

                elif live_sessions[session_id].get("state") not in ("error", "stopped"):

                    live_sessions[session_id]["state"] = "completed"

                live_sessions[session_id]["running"] = False

                live_sessions[session_id]["completed"] = True

@router.post("/start", response_model=LiveSessionStartResponse, summary="Start live exam session")

async def start_live_exam(req: LiveSessionStartRequest):

    video_id = (req.video_id or "").strip()

    if not video_id or "/" in video_id or "\\" in video_id or ".." in video_id:

        raise HTTPException(status_code=400, detail="Invalid video_id")

    clip_path = SAMPLE_DIR / f"{video_id}.mp4"

    if not clip_path.exists() or not clip_path.is_file():

        raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")

    session_id = (req.session_name or f"live_{video_id}_{uuid.uuid4().hex[:6]}").replace("/", "_").replace("\\", "_")

    if not session_id or ".." in session_id:

        raise HTTPException(status_code=400, detail="Invalid session_name")

    max_frames = None

    if req.max_frames is not None:

        try:

            max_frames = max(1, min(int(req.max_frames), 5_000_000))

        except (TypeError, ValueError):

            raise HTTPException(status_code=422, detail="max_frames must be a positive integer")

    capture = cv2.VideoCapture(str(clip_path))

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if capture.isOpened() else 0

    capture.release()

    if total_frames <= 0:

        total_frames = None

    with live_lock:

        if session_id in live_sessions:

            raise HTTPException(status_code=409, detail="Session already exists")

        processed_video_path = pathlib.Path(__file__).resolve().parents[4] / "clips" / "processed" / f"{session_id}.mp4"

        live_sessions[session_id] = {

            "queue": queue.Queue(maxsize=30),

            "running": True,

            "completed": False,

            "state": "starting",

            "frames_processed": 0,

            "total_frames": total_frames,

            "violations": [],

            "clip_name": video_id,

            "video_id": video_id,

            "error": None,

            "students": [],

            "frame_id": 0,

            "timestamp": time.time(),

            "fps": 0.0,

            "processing_fps": 0.0,

            "video_width": 0,

            "video_height": 0,

            "processing_latency_ms": 0.0,

            "processed_video_path": str(processed_video_path),

            "processed_video_ready": False,

        }

    t = threading.Thread(target=_live_worker, args=(clip_path, session_id, max_frames, video_id), daemon=True)

    with live_lock:

        if session_id in live_sessions:

            live_sessions[session_id]["thread"] = t

    t.start()

    return LiveSessionStartResponse(

        session_id=session_id,

        stream_url=f"/api/v1/live/{session_id}/stream",

        state_url=f"/api/v1/live/{session_id}/state",

        video_url=f"/api/v1/video/samples/{video_id}",

        processed_video_url=f"/api/v1/live/{session_id}/processed-video",

        total_frames=total_frames,

        message=f"Live exam started for {video_id} session {session_id}",

    )

@router.get("/{session_id}/processed-video", summary="Get the fully annotated analysis video")

async def get_processed_video(session_id: str):

    if not session_id or "/" in session_id or "\\" in session_id or ".." in session_id:

        raise HTTPException(status_code=400, detail="Invalid session_id")

    sess = _get_live_session(session_id)

    if not sess:

        raise HTTPException(status_code=404, detail="Session not found")

    if not sess.get("processed_video_ready"):

        if sess.get("state") == "error":

            raise HTTPException(status_code=409, detail=sess.get("error") or "Annotated video could not be created")

        raise HTTPException(status_code=425, detail="Annotated video is not ready yet")

    path = pathlib.Path(sess["processed_video_path"]).resolve()

    expected_dir = (pathlib.Path(__file__).resolve().parents[4] / "clips" / "processed").resolve()

    if path.parent != expected_dir or not path.is_file():

        raise HTTPException(status_code=404, detail="Annotated video is unavailable")

    return FileResponse(path, media_type="video/mp4", filename=f"exam-analysis-{session_id}.mp4")

@router.get("/{session_id}/stream", summary="Live MJPEG stream (annotated, real boxes)")

async def live_stream(session_id: str):

    if not session_id or "/" in session_id or "\\" in session_id or ".." in session_id:

        raise HTTPException(status_code=400, detail="Invalid session_id")

    sess = _get_live_session(session_id)

    if not sess:

        raise HTTPException(status_code=404, detail="Session not found")

    async def gen():

        while True:

            sess_cur = _get_live_session(session_id)

            if not sess_cur:

                break

            try:

                jpeg = sess_cur["queue"].get(timeout=0.5)

                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

            except queue.Empty:

                if not sess_cur.get("running", False) and sess_cur["queue"].empty():

                    break

                continue

            except Exception:

                break

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/{session_id}/state", response_model=LiveAIState, summary="Get current lightweight AI state")

async def get_live_state(session_id: str):

    if not session_id or "/" in session_id or "\\" in session_id or ".." in session_id:

        raise HTTPException(status_code=400, detail="Invalid session_id")

    sess = _get_live_session(session_id)

    if not sess:

        raise HTTPException(status_code=404, detail="Session not found")

    return LiveAIState(

        frame_id=int(sess.get("frame_id", 0)),

        timestamp=float(sess.get("timestamp", time.time())),

        fps=float(sess.get("fps", 0.0)),

        processing_fps=float(sess.get("processing_fps", 0.0)),

        students=[StudentTrackState(**s) for s in sess.get("students", [])],

        violations=[ViolationEvent(**v) for v in sess.get("violations", [])],

        video_width=int(sess.get("video_width", 0)),

        video_height=int(sess.get("video_height", 0)),

        processing_latency_ms=float(sess.get("processing_latency_ms", 0.0)),

    )

@router.get("/{session_id}/status", response_model=LiveSessionStatus, summary="Get live session status")

async def live_status(session_id: str):

    if not session_id or "/" in session_id or "\\" in session_id or ".." in session_id:

        raise HTTPException(status_code=400, detail="Invalid session_id")

    sess = _get_live_session(session_id)

    if not sess:

        raise HTTPException(status_code=404, detail="Session not found")

    return LiveSessionStatus(

        session_id=session_id,

        video_id=str(sess.get("video_id", "")),

        state=str(sess.get("state", "running" if sess.get("running") else "stopped")),

        frame_id=int(sess.get("frame_id", 0)),

        timestamp=float(sess.get("timestamp", time.time())),

        fps=float(sess.get("fps", 0.0)),

        processing_fps=float(sess.get("processing_fps", 0.0)),

        students=[StudentTrackState(**s) for s in sess.get("students", [])],

        violations=[ViolationEvent(**v) for v in sess.get("violations", [])],

        video_width=int(sess.get("video_width", 0)),

        video_height=int(sess.get("video_height", 0)),

        processing_latency_ms=float(sess.get("processing_latency_ms", 0.0)),

        frames_processed=int(sess.get("frames_processed", 0)),

        total_frames=sess.get("total_frames"),

    )

@router.delete("/{session_id}", summary="Stop live exam session")

async def stop_live(session_id: str):

    if not session_id or "/" in session_id or "\\" in session_id or ".." in session_id:

        raise HTTPException(status_code=400, detail="Invalid session_id")

    sess = _get_live_session(session_id)

    if not sess:

        raise HTTPException(status_code=404, detail="Session not found")

    with live_lock:

        state = str(sess.get("state", ""))

        if state not in ("completed", "error", "stopped"):

            sess["running"] = False

            sess["state"] = "stopping"

            state = "stopping"

    return {"message": f"Session {session_id} {state}", "state": state}
