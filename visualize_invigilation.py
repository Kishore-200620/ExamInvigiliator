"""
ExamInvigilationAI — Real-Time Visual Invigilation Pipeline Visualizer

Visually displays the live AI invigilation pipeline in action (2026-09-16 updated):
  1. YOLO26n-Pose Unified Detection (GREEN=Normal, RED=Malpractice Active) — single pass bbox + 17 keypoints
  2. Pure YOLO-keypoint Head Geometry (no MediaPipe) -> direction LEFT/CENTER/RIGHT + yaw proxy + confidence
  3. Wrist Containment via check_wrist_containment (INSIDE / OUTSIDE / UNKNOWN) + skeleton overlay
  4. TemporalHistory -> BehaviorFusion -> StudentBehaviorState -> RuleEngine (settings-driven head-turn timing with brief unknown-pose grace)
  5. Real-time HUD (thresholds from settings) + candidate progress + severity + video export

Pipeline (matches backend/app/ai/detector/yolo.py + tracker/manager.py + behavior/* + services/rule_engine.py):
  YOLO26n-Pose (Detection.metadata["keypoints"] (17,3)) -> TrackingManager (cost+hungarian+orphan rescue+dedup)
  -> _update_head_pose_from_keypoints (classify_head_direction, pose_history capped 10, per-frame 0.02ms, interval=1)
  -> BodyPoseAnalyzer.from_detection (zero extra inference) -> TemporalHistory append -> Fusion -> RuleEngine

Usage:
  python visualize_invigilation.py
  python visualize_invigilation.py --video sample_clips/clip5.mp4 --max-frames 300
  python visualize_invigilation.py --video sample_clips/clip5.mp4 --max-frames 500 --headless --output clips/visual_invigilation_demo.mp4
"""
import argparse
import pathlib
import sys
import time
import cv2
import numpy as np

# Ensure backend modules can be imported
_proj_root = pathlib.Path(__file__).resolve().parent
_backend_dir = _proj_root / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from app.config.settings import settings
from app.ai.video.capture import OpenCVCapture
from app.ai.video.manager import VideoManager
from app.ai.detector.yolo import YOLODetector
from app.ai.detector.manager import DetectionManager
from app.ai.tracker.manager import TrackingManager
from app.ai.behavior.body import BodyPoseAnalyzer
from app.ai.behavior.fusion import BehaviorFusion
from app.ai.behavior.schemas import BodyPoseObservation, HeadPoseObservation, check_wrist_containment
from app.services.temporal_history import TemporalHistory
from app.services.rule_engine import RuleEngine, EventStatus
from app.ai.tracker.head_pose import HeadPoseEstimator, classify_head_direction

# Skeleton pairs for COCO 17 (for overlay)
COCO_SKELETON = [
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


def _find_matched_detection(track, detections):
    """Find best matching Detection for a track (IoU >0.3 or center containment). Uses track.motion.bbox."""
    try:
        tbox = track.motion.bbox
        tcx = (tbox.x1 + tbox.x2) / 2.0
        tcy = (tbox.y1 + tbox.y2) / 2.0
        best = None
        best_score = -1.0
        # Try IoU via CostCalculator if available
        try:
            from app.ai.tracker.cost import CostCalculator
            calc = CostCalculator()
            for d in detections:
                iou = calc.iou(tbox, d.bbox)
                center_inside = (d.bbox.x1 <= tcx <= d.bbox.x2) and (d.bbox.y1 <= tcy <= d.bbox.y2)
                score = iou + (0.1 if center_inside else 0.0)
                if score > best_score and (iou >= 0.30 or center_inside):
                    best_score = score
                    best = d
        except Exception:
            # Fallback: center-based distance
            for d in detections:
                dcx = (d.bbox.x1 + d.bbox.x2) / 2.0
                dcy = (d.bbox.y1 + d.bbox.y2) / 2.0
                dist = abs(dcx - tcx) + abs(dcy - tcy)
                w = (tbox.x2 - tbox.x1)
                h = (tbox.y2 - tbox.y1)
                inside = (d.bbox.x1 <= tcx <= d.bbox.x2) and (d.bbox.y1 <= tcy <= d.bbox.y2)
                if inside or (w > 0 and h > 0 and dist < max(w, h) * 0.8):
                    if best is None or dist < best_score:
                        best_score = dist
                        best = d
        return best
    except Exception:
        return None


def _draw_skeleton(annotated, keypoints: dict, conf_thresh: float = 0.30):
    """Draw faint skeleton lines + keypoints."""
    if not keypoints:
        return
    # Lines
    for a, b in COCO_SKELETON:
        pa = keypoints.get(a)
        pb = keypoints.get(b)
        if pa and pb and pa[2] >= conf_thresh and pb[2] >= conf_thresh:
            cv2.line(annotated, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (180, 180, 180), 1, cv2.LINE_AA)
    # Points
    for name, pt in keypoints.items():
        if pt and pt[2] >= conf_thresh:
            x, y = int(pt[0]), int(pt[1])
            col = (0, 255, 255) if "wrist" in name else (255, 200, 0) if "eye" in name or "ear" in name or "nose" in name else (200, 200, 255)
            cv2.circle(annotated, (x, y), 3, (255, 255, 255), -1)
            cv2.circle(annotated, (x, y), 2, col, -1)


def run_visualization(video_path: str, max_frames: int = 500, headless: bool = False, output_path: str = "clips/visual_invigilation_demo.mp4"):
    print("=" * 70)
    print("  ExamInvigilationAI — Real-Time Visualizer (2026-09-16 unified)")
    print(f"  Input Video: {video_path}")
    print(f"  Output Video: {output_path}")
    print(f"  YOLO Model: {settings.YOLO_MODEL} | Device: {settings.DEVICE} | YOLO_POSE: {settings.YOLO_POSE}")
    print(f"  Thresholds: HEAD_OFFSET={settings.HEAD_OFFSET_THRESHOLD} yaw_legacy={settings.HEAD_YAW_THRESHOLD} | HEAD_TURN {settings.HEAD_TURN_DURATION_SECONDS}s | WRIST {settings.WRIST_OUTSIDE_DURATION_SECONDS}s (conf {settings.WRIST_CONFIDENCE_THRESHOLD}) | size {settings.YOLO_IMAGE_SIZE} conf {settings.YOLO_CONFIDENCE} iou {settings.YOLO_IOU}")
    print("=" * 70)

    clip_file = pathlib.Path(video_path)
    if not clip_file.exists():
        print(f"ERROR: Video file {video_path} does not exist!")
        return

    out_dir = pathlib.Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(clip_file))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Video Resolution: {width}x{height} @ {fps:.1f} FPS (Total frames: {total_video_frames})")

    # Initialize AI pipeline — matches production backend/app/ai/* + services/*
    capture = OpenCVCapture(str(clip_file))
    video_mgr = VideoManager(capture, source_id="visualizer")
    detector = DetectionManager(YOLODetector())
    # ReID: histogram CPU cheap; head pose is pure geometry (no device), pass pose_device unused
    tracker = TrackingManager(reid_enabled=True, reid_device="cpu", appearance_weight=0.2, motion_weight=0.8)
    # BodyPoseAnalyzer: only for from_detection conversion (zero inference), not analyze_frame
    # Device arg kept for compat but model not re-used for per-frame inference
    try:
        body_analyzer = BodyPoseAnalyzer(device="cuda" if getattr(settings, "DEVICE", "cpu") == "cuda" else "cpu")
    except Exception:
        body_analyzer = BodyPoseAnalyzer(device="cpu")
    fusion = BehaviorFusion()
    temporal = TemporalHistory(max_size=10, max_duration_seconds=10.0)
    # Also keep feature history implicitly via temporal; movement derived features optional for visualization
    rule_engine = RuleEngine(config={
        "head_turn_duration_seconds": settings.HEAD_TURN_DURATION_SECONDS,
        "wrist_outside_duration_seconds": settings.WRIST_OUTSIDE_DURATION_SECONDS,
        "movement_threshold": 50.0,
        "candidate_duration": 0.5,
        "confirmation_duration": 1.0,
        "event_end_duration": 1.0,
        "rule_cooldown": 5.0,
        "min_confidence": 0.3,
    })
    head_estimator = HeadPoseEstimator(device="cpu")

    print(f"RuleEngine: HEAD_TURN_LEFT/RIGHT {settings.HEAD_TURN_DURATION_SECONDS}s (offset {settings.HEAD_OFFSET_THRESHOLD}) | WRIST_OUTSIDE {settings.WRIST_OUTSIDE_DURATION_SECONDS}s | MOVEMENT 50 | cooldown 5s | min_conf 0.3")
    print(f"Tracker: max_misses 30 lost 5 tentative 3 grace 15 rescue 0.35 dedup 0.85 | pose per-frame geometry (~0.02ms) | ReID histogram 0.13ms event-triggered")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    video_mgr.start()
    detector.start()

    window_name = "ExamInvigilationAI - Real-Time AI Visualizer"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, width), min(720, height))

    frame_counter = 0
    paused = False
    start_time = time.time()
    sample_frame_saved = False

    print("\nStarting pipeline execution... Press 'Q' to quit, 'P' to pause, 'S' to snapshot.")

    try:
        while video_mgr.is_running() and frame_counter < max_frames:
            if paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('p') or key == ord('P'):
                    paused = False
                    print("Resumed.")
                elif key == ord('q') or key == ord('Q'):
                    break
                continue

            frame = video_mgr.read()
            if frame is None:
                time.sleep(0.005)
                if not video_mgr.is_running():
                    break
                continue

            frame_counter += 1
            curr_ts = frame.timestamp if frame.timestamp > 0 else (frame_counter / fps)

            # Step 1: Unified YOLO26n-Pose detection (bbox + 17 keypoints in metadata)
            dets = detector.detect(frame)

            # Step 2: Tracker update — includes _update_head_pose_from_keypoints (pure geometry, pose_history)
            tracks = tracker.update(dets, frame=frame)

            # Step 3: Per-track Behavior pipeline (no second YOLO inference)
            frame_active_violations = []
            # Cache per-track observations for rendering
            track_obs_cache = {}  # track_id -> {body_obs, head_obs, behavior_state, head_dir, head_conf, yaw_proxy, lw_status, rw_status}

            for track in tracks:
                if track.state.name != "CONFIRMED":
                    continue
                track_id = track.track_id
                tbox = track.motion.bbox

                # Find matched detection for this track (for keypoints)
                matched_det = _find_matched_detection(track, dets)
                    # Reuse last_detection only when it belongs to this frame.
                if matched_det is None and hasattr(track, "last_detection") and track.last_detection is not None:
                    try:
                        if track.last_detection.frame_id == frame.frame_id:
                            matched_det = track.last_detection
                    except Exception:
                        pass

                # Build BodyPoseObservation via from_detection (zero inference, preserves conf)
                body_obs = None
                if matched_det is not None:
                    try:
                        body_obs = body_analyzer.from_detection(matched_det, frame_id=frame.frame_id, timestamp=curr_ts)
                    except Exception:
                        body_obs = BodyPoseObservation(frame_id=frame.frame_id, timestamp=curr_ts, available=False)
                else:
                    body_obs = BodyPoseObservation(frame_id=frame.frame_id, timestamp=curr_ts, available=False)

                # Head pose: prefer tracker.pose_history (authoritative, already computed per-frame)
                # Fallback to direct geometry on matched_det keypoints
                head_dir = "UNKNOWN"
                head_conf = 0.0
                yaw_proxy = 0.0
                head_detected = False
                pitch = 0.0
                roll = 0.0
                pose_src = "none"
                try:
                    if getattr(track, "pose_history", None) and len(track.pose_history) > 0:
                        lp = track.pose_history[-1]
                        # pose_history is list[dict] in manager._update_head_pose_from_keypoints
                        pose_is_fresh = int(lp.get("frame_id", -1)) == int(frame.frame_id) if isinstance(lp, dict) else False
                        if isinstance(lp, dict) and pose_is_fresh and lp.get("direction") not in (None, "UNKNOWN"):
                            head_dir = lp.get("direction", "UNKNOWN")
                            head_conf = float(lp.get("confidence", 0.0))
                            yaw_proxy = float(lp.get("yaw", 0.0))
                            head_detected = True
                            pose_src = "tracker.pose_history"
                        elif hasattr(lp, "direction") and int(getattr(lp, "frame_id", -1)) == int(frame.frame_id):
                            d = getattr(lp, "direction", "UNKNOWN")
                            if d != "UNKNOWN":
                                head_dir = d
                                head_conf = float(getattr(lp, "confidence", 0.0))
                                yaw_proxy = float(getattr(lp, "yaw", 0.0))
                                head_detected = True
                                pose_src = "tracker.pose_history(obj)"
                except Exception:
                    head_dir, head_conf, head_detected = "UNKNOWN", 0.0, False

                if not head_detected:
                    # Direct geometry from matched_det keypoints or body_obs.keypoints
                    kpts_for_head = None
                    if body_obs and body_obs.keypoints:
                        kpts_for_head = body_obs.keypoints
                    elif matched_det and isinstance(matched_det.metadata, dict):
                        kpts_for_head = matched_det.metadata.get("keypoints")
                    if kpts_for_head is not None:
                        try:
                            # classify_head_direction handles dict or np.ndarray (17,3)
                            d, c = classify_head_direction(kpts_for_head)
                            if d != "UNKNOWN":
                                head_dir = d
                                head_conf = float(c)
                                # yaw proxy from normalized offset for display
                                try:
                                    # recompute normalized offset for yaw display (same as in head_pose.py)
                                    if isinstance(kpts_for_head, dict):
                                        nose = kpts_for_head.get("nose")
                                        le = kpts_for_head.get("left_eye")
                                        re = kpts_for_head.get("right_eye")
                                        lear = kpts_for_head.get("left_ear")
                                        rear = kpts_for_head.get("right_ear")
                                        # prefer eyes
                                        if nose and le and re and nose[2] >= 0.3 and le[2] >= 0.3 and re[2] >= 0.3:
                                            mx = (le[0] + re[0]) / 2
                                            inter = max(1.0, abs(le[0] - re[0]))
                                            norm = (nose[0] - mx) / inter
                                            yaw_proxy = float(np.clip(norm * 60.0, -60, 60))
                                        elif nose and lear and rear and nose[2] >= 0.3 and lear[2] >= 0.3 and rear[2] >= 0.3:
                                            mx = (lear[0] + rear[0]) / 2
                                            inter = max(1.0, abs(lear[0] - rear[0]))
                                            norm = (nose[0] - mx) / inter
                                            yaw_proxy = float(np.clip(norm * 60.0, -60, 60))
                                    else:
                                        # ndarray path via estimator
                                        yaw_proxy, _, _, _, _, _ = head_estimator.estimate_from_keypoints(kpts_for_head)
                                except Exception:
                                    yaw_proxy = 0.0
                                head_detected = True
                                pose_src = "direct.classify_head_direction"
                        except Exception:
                            head_dir, head_conf, head_detected = "UNKNOWN", 0.0, False
                    # Still not detected -> keep UNKNOWN (wrist still evaluated)

                # Stash on track for rendering (mirrors TrackingManager state for overlay)
                track.head_direction = head_dir  # type: ignore
                track.head_yaw = yaw_proxy  # type: ignore
                track.head_confidence = head_conf  # type: ignore

                # Build HeadPoseObservation (required for Fusion)
                head_obs = HeadPoseObservation(
                    available=head_detected,
                    yaw=yaw_proxy,
                    pitch=pitch,
                    roll=roll,
                    direction=head_dir,
                    confidence=head_conf if head_detected else 0.0,
                    frame_id=frame.frame_id,
                    timestamp=curr_ts,
                )

                # TemporalHistory: append per-signal (bounded 10, 10s) — now actually used
                try:
                    temporal.append(track_id, "body", body_obs, timestamp=curr_ts)
                    temporal.append(track_id, "head_pose", head_obs, timestamp=curr_ts)
                except Exception:
                    pass

                # Wrist containment precomputed for rendering (also done inside fusion)
                lw_status = check_wrist_containment(body_obs.left_wrist if body_obs else None, tbox, settings.WRIST_CONFIDENCE_THRESHOLD)
                rw_status = check_wrist_containment(body_obs.right_wrist if body_obs else None, tbox, settings.WRIST_CONFIDENCE_THRESHOLD)

                # Fuse -> StudentBehaviorState (input to RuleEngine)
                # Note: movement_features not computed here (needs temporal history); pass empty -> fusion handles None safely
                fusion_obs = fusion.fuse(
                    student_id=track_id,
                    frame_id=frame.frame_id,
                    timestamp=curr_ts,
                    head_pose=head_obs,
                    body_pose=body_obs,
                    track_bbox=tbox,
                )
                behavior_state = fusion.create_behavior_state(
                    student_id=track_id,
                    frame_id=frame.frame_id,
                    timestamp=curr_ts,
                    behavior_obs=fusion_obs,
                    track_bbox=tbox,
                )

                # Evaluate RuleEngine (per-student; interaction disabled—needs pair)
                events = rule_engine.evaluate(behavior_state, timestamp=curr_ts, frame_id=frame.frame_id)
                for ev in events:
                    if ev.status in (EventStatus.CONFIRMED, EventStatus.ACTIVE):
                        frame_active_violations.append(ev)
                        print(f"  🚨 [FRAME {frame_counter} | {curr_ts:.2f}s] VIOLATION: Student #{ev.student_id} -> {ev.event_type} ({ev.status.value}) duration={ev.duration:.1f}s conf={ev.confidence:.2f} bbox={tbox.x1},{tbox.y1},{tbox.x2},{tbox.y2} src={pose_src}")

                # For ENDED events (cooldown started), also log
                for ev in events:
                    if ev.status == EventStatus.ENDED:
                        print(f"  ✅ [FRAME {frame_counter}] ENDED: Student #{ev.student_id} {ev.event_type} total={ev.duration:.1f}s cooldown until {rule_engine.rule_states.get(ev.student_id, {}).get(ev.rule_id, {}).cooldown_until if ev.student_id in rule_engine.rule_states else '?'}")

                track_obs_cache[track_id] = {
                    "body_obs": body_obs,
                    "head_obs": head_obs,
                    "behavior_state": behavior_state,
                    "head_dir": head_dir,
                    "head_conf": head_conf,
                    "yaw_proxy": yaw_proxy,
                    "head_detected": head_detected,
                    "lw_status": lw_status,
                    "rw_status": rw_status,
                    "pose_src": pose_src,
                }

            # All active events (confirmed + active) for banner rendering
            all_active_rule_events = rule_engine.get_active_events()
            # Also collect candidate states for progress bars (NORMAL->CANDIDATE->CONFIRMED)
            # RuleState stored in rule_engine.rule_states[student_id][rule_id]

            # Step 4: Render Visual Annotations
            annotated = frame.image.copy()

            for track in tracks:
                t_id = track.track_id
                bx1, by1, bx2, by2 = int(track.motion.bbox.x1), int(track.motion.bbox.y1), int(track.motion.bbox.x2), int(track.motion.bbox.y2)
                # Clamp to image
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(width - 1, bx2), min(height - 1, by2)

                # Active malpractice for this student (confirmed/active only)
                student_active_events = [e for e in all_active_rule_events if e.student_id == t_id]
                has_violation = len(student_active_events) > 0

                # Color: RED violation > GREEN confirmed > YELLOW tentative > GRAY lost/tentative_lost
                sname = track.state.name if hasattr(track.state, "name") else str(track.state)
                if has_violation:
                    box_color = (0, 0, 255)
                    thickness = 3
                elif sname == "CONFIRMED":
                    box_color = (0, 255, 0)
                    thickness = 2
                elif sname == "TENTATIVE":
                    box_color = (0, 255, 255)
                    thickness = 1
                else:
                    box_color = (180, 180, 180)
                    thickness = 1

                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), box_color, thickness)

                # ID header
                id_text = f"#{t_id} {sname[:4]}"
                # Head direction + confidence for header (if available)
                cache = track_obs_cache.get(t_id)
                if cache and cache["head_dir"] != "UNKNOWN":
                    id_text += f" {cache['head_dir']} {cache['head_conf']:.2f}"
                cv2.rectangle(annotated, (bx1, max(0, by1 - 22)), (bx1 + 170, by1), box_color, -1)
                cv2.putText(annotated, id_text, (bx1 + 4, max(12, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0) if box_color != (0, 0, 255) else (255, 255, 255), 1, cv2.LINE_AA)

                # Head badge (second line above bbox, offset 0.20 threshold display)
                if cache:
                    head_dir = cache["head_dir"]
                    yaw_proxy = cache["yaw_proxy"]
                    head_conf = cache["head_conf"]
                    lw_status = cache["lw_status"]
                    rw_status = cache["rw_status"]
                    if head_dir and head_dir != "UNKNOWN":
                        yaw_str = f"{yaw_proxy:+.0f}deg" if yaw_proxy is not None else ""
                        head_label = f"HEAD:{head_dir} {yaw_str} c{head_conf:.2f}"
                        dir_color = (255, 160, 0) if head_dir == "LEFT" else (0, 140, 255) if head_dir == "RIGHT" else (0, 210, 0)
                        # Outline for readability
                        cv2.putText(annotated, head_label, (bx1, max(32, by1 - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(annotated, head_label, (bx1, max(32, by1 - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, dir_color, 1, cv2.LINE_AA)
                    # Wrist status label under head badge
                    wrist_label = f"L:{lw_status} R:{rw_status}"
                    # Color wrists: OUTSIDE=red, INSIDE=green, UNKNOWN=gray
                    lw_col = (0, 0, 255) if lw_status == "OUTSIDE" else (0, 220, 0) if lw_status == "INSIDE" else (150, 150, 150)
                    rw_col = (0, 0, 255) if rw_status == "OUTSIDE" else (0, 220, 0) if rw_status == "INSIDE" else (150, 150, 150)
                    # We'll render two separate texts for colors
                    # Background pill for wrist
                    pill_w = 150
                    pill_y = max(48, by1 - 44) + 14
                    # Only draw if confirmed track had body_obs
                    if sname == "CONFIRMED":
                        cv2.rectangle(annotated, (bx1, pill_y - 12), (bx1 + pill_w, pill_y + 4), (30, 30, 30), -1)
                        cv2.rectangle(annotated, (bx1, pill_y - 12), (bx1 + pill_w, pill_y + 4), (80, 80, 80), 1)
                        cv2.putText(annotated, f"L:{lw_status}", (bx1 + 4, pill_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, lw_col, 1, cv2.LINE_AA)
                        cv2.putText(annotated, f"R:{rw_status}", (bx1 + 70, pill_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, rw_col, 1, cv2.LINE_AA)

                # Candidate progress bars (before firing): show RuleState.candidate_start persistence
                if t_id in rule_engine.rule_states:
                    cand_y = by2 + 16
                    for rule_id in ["HEAD_TURN_LEFT", "HEAD_TURN_RIGHT", "WRIST_OUTSIDE_REGION", "SUSPICIOUS_MOVEMENT"]:
                        rs = rule_engine.rule_states[t_id].get(rule_id)
                        if rs and rs.state == "CANDIDATE" and rs.candidate_start is not None:
                            dur = curr_ts - rs.candidate_start
                            # Required duration per rule
                            req = rule_engine.config.get("head_turn_duration_seconds", 3.0) if "HEAD_TURN" in rule_id else rule_engine.config.get("wrist_outside_duration_seconds", 2.0) if "WRIST" in rule_id else rule_engine.config.get("candidate_duration", 0.5)
                            pct = min(0.99, dur / max(0.1, req))
                            bar_w = int((bx2 - bx1) * pct)
                            bar_w = max(2, min(bar_w, bx2 - bx1))
                            # Draw bar background
                            cv2.rectangle(annotated, (bx1, cand_y - 9), (bx2, cand_y + 2), (40, 40, 40), -1)
                            cv2.rectangle(annotated, (bx1, cand_y - 9), (bx1 + bar_w, cand_y + 2), (0, 165, 255), -1)  # orange progress
                            cv2.rectangle(annotated, (bx1, cand_y - 9), (bx2, cand_y + 2), (200, 200, 200), 1)
                            txt = f"{rule_id} CAND {dur:.1f}/{req:.0f}s"
                            cv2.putText(annotated, txt, (bx1 + 3, cand_y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
                            cand_y += 14
                            if cand_y > by2 + 40:
                                break

                # Draw skeleton + wrist halos from cached body_obs (not from empty TemporalHistory)
                if cache and cache["body_obs"] and cache["body_obs"].available and cache["body_obs"].keypoints:
                    _draw_skeleton(annotated, cache["body_obs"].keypoints, conf_thresh=settings.WRIST_CONFIDENCE_THRESHOLD)
                    # Extra halos for wrists
                    for w_name, w_col in [("left_wrist", (255, 0, 255)), ("right_wrist", (0, 255, 255))]:
                        kp = cache["body_obs"].keypoints.get(w_name)
                        if kp and kp[2] >= settings.WRIST_CONFIDENCE_THRESHOLD:
                            wx, wy = int(kp[0]), int(kp[1])
                            # Status-aware outer ring: red if OUTSIDE, green if INSIDE
                            status = lw_status if w_name == "left_wrist" else rw_status
                            ring = (0, 0, 255) if status == "OUTSIDE" else (0, 255, 0) if status == "INSIDE" else (180, 180, 180)
                            cv2.circle(annotated, (wx, wy), 10, ring, 2)
                            cv2.circle(annotated, (wx, wy), 7, (255, 255, 255), -1)
                            cv2.circle(annotated, (wx, wy), 5, w_col, -1)

                # Active violation banner under box (RED pill + severity + duration)
                if has_violation:
                    for idx, ev in enumerate(student_active_events):
                        # Severity color: HIGH/CRITICAL=red, MEDIUM=orange, LOW=yellow
                        sev_col = (0, 0, 200)
                        if ev.severity.value == "medium":
                            sev_col = (0, 140, 255)
                        elif ev.severity.value == "low":
                            sev_col = (0, 200, 200)
                        viol_text = f"MALPRACTICE: {ev.event_type} {ev.status.value} {ev.duration:.1f}s"
                        banner_y = min(height - 10, by2 + 20 + idx * 22)
                        t_size = cv2.getTextSize(viol_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
                        # Clamp pill to image width
                        pill_x2 = min(width - 4, bx1 + t_size[0] + 10)
                        cv2.rectangle(annotated, (bx1, banner_y - 14), (pill_x2, banner_y + 4), sev_col, -1)
                        cv2.rectangle(annotated, (bx1, banner_y - 14), (pill_x2, banner_y + 4), (255, 255, 255), 1)
                        cv2.putText(annotated, viol_text, (bx1 + 5, banner_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            # Step 5: Top HUD — thresholds from settings / rule_engine.config (not hard-coded)
            hud_h = 62
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (width, hud_h), (18, 18, 18), -1)
            annotated = cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0)
            cv2.line(annotated, (0, hud_h), (width, hud_h), (0, 255, 255), 1)

            elapsed = max(0.001, time.time() - start_time)
            proc_fps = frame_counter / elapsed
            active_viol_cnt = len(all_active_rule_events)
            confirmed_cnt = len([t for t in tracks if t.state.name == "CONFIRMED"])
            total_cnt = len(tracks)

            cv2.putText(annotated, "ExamInvigilationAI — LIVE DEMO (unified YOLO-pose + geometry)", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"Frame {frame_counter}/{max_frames} | {proc_fps:.1f} FPS | det {len(dets)} track {total_cnt} confirmed {confirmed_cnt}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
            # Dynamic thresholds line (robust to exam_invigiliator_2 config keys)
            _move_thr = rule_engine.config.get('movement_threshold', rule_engine.config.get('movement_threshold', 50))
            # config may use legacy keys; ensure fallback
            if _move_thr is None:
                _move_thr = 50.0
            hud_cfg_txt = f"HEAD {settings.HEAD_TURN_DURATION_SECONDS:.1f}s off {settings.HEAD_OFFSET_THRESHOLD:.2f} gap {settings.HEAD_TURN_UNKNOWN_GRACE_SECONDS:.1f}s | WRIST {settings.WRIST_OUTSIDE_DURATION_SECONDS:.0f}s c{settings.WRIST_CONFIDENCE_THRESHOLD:.2f} | MOVE {float(_move_thr):.0f} | pose per-frame geometry"
            cv2.putText(annotated, hud_cfg_txt, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 220, 255), 1, cv2.LINE_AA)

            # Right HUD boxes: students + violations
            # Students box
            cv2.rectangle(annotated, (max(0, width - 420), 7), (width - 145, 54), (0, 90, 0), -1)
            cv2.rectangle(annotated, (max(0, width - 420), 7), (width - 145, 54), (255, 255, 255), 1)
            cv2.putText(annotated, f"TRACKED: {confirmed_cnt}/{total_cnt}", (width - 412, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"YOLO {settings.YOLO_MODEL.split('/')[-1][:14]}", (width - 412, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 255, 220), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"{width}x{height} {fps:.0f}fps", (width - 412, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1, cv2.LINE_AA)

            # Violations box (red when active, green otherwise) + disabled rules hint
            viol_bg = (0, 0, 170) if active_viol_cnt > 0 else (0, 110, 0)
            cv2.rectangle(annotated, (width - 135, 7), (width - 8, 54), viol_bg, -1)
            cv2.rectangle(annotated, (width - 135, 7), (width - 8, 54), (255, 255, 255), 1)
            cv2.putText(annotated, f"VIOL", (width - 125, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"{active_viol_cnt}", (width - 120, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

            # Writer
            writer.write(annotated)

            if not sample_frame_saved and frame_counter >= 30:
                sample_img_path = pathlib.Path("clips/visual_invigilation_sample.jpg")
                cv2.imwrite(str(sample_img_path), annotated)
                print(f"Sample screenshot saved to {sample_img_path}")
                sample_frame_saved = True

            if not headless:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("User requested quit.")
                    break
                elif key == ord('p') or key == ord('P'):
                    paused = True
                    print("Paused. Press 'P' to resume.")
                elif key == ord('s') or key == ord('S'):
                    snap_path = f"clips/snapshot_frame_{frame_counter}.jpg"
                    cv2.imwrite(snap_path, annotated)
                    print(f"Saved snapshot to {snap_path}")

            if frame_counter % 50 == 0:
                # Report active + candidate counts
                cand_cnt = sum(1 for sid in rule_engine.rule_states for rid, rs in rule_engine.rule_states[sid].items() if rs.state == "CANDIDATE")
                print(f"  Processed frame {frame_counter}/{max_frames} ({proc_fps:.1f} FPS) | dets {len(dets)} confirmed {confirmed_cnt} active_viol {active_viol_cnt} candidate {cand_cnt} | pose_history avg {np.mean([len(t.pose_history) for t in tracks]) if tracks else 0:.1f}")

    finally:
        video_mgr.stop()
        detector.stop()
        writer.release()
        if not headless:
            cv2.destroyAllWindows()

    print("=" * 70)
    print("  Visualization Complete!")
    print(f"  Total Frames Processed: {frame_counter}")
    print(f"  Output Annotated Video: {output_path}")
    # Summary of events
    completed = rule_engine.get_completed_events()
    active = rule_engine.get_active_events()
    print(f"  Completed Events: {len(completed)} | Still Active: {len(active)}")
    if completed:
        for ev in completed[:10]:
            print(f"    - {ev.event_type} student {ev.student_id} {ev.duration:.1f}s {ev.status.value} frames {ev.start_frame}->{ev.end_frame}")
    if active:
        for ev in active[:10]:
            print(f"    * ACTIVE {ev.event_type} student {ev.student_id} {ev.duration:.1f}s")
    print(f"  Tracking: total pose extr {tracker.total_pose_extractions} reid extr {tracker.total_reid_extractions} dedup kept logic IoU 0.85")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Exam Invigilation AI Visualizer (unified YOLO-pose + geometry)")
    parser.add_argument("--video", type=str, default="sample_clips/clip5.mp4", help="Path to input exam video clip")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum frames to process (default: 300)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window (save video only)")
    parser.add_argument("--output", type=str, default="clips/visual_invigilation_demo.mp4", help="Output MP4 file path")
    args = parser.parse_args()

    run_visualization(video_path=args.video, max_frames=args.max_frames, headless=args.headless, output_path=args.output)
