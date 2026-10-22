"""
Clips API - Phase 9 Clip Test
Allows frontend to test with existing clips without live stream.
"""
import time
import uuid
import queue
import threading
import pathlib
from typing import Optional

import cv2
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Query, HTTPException
from fastapi.responses import FileResponse

from app.services.violation_service import violation_service
from app.services.rule_engine import MalpracticeEvent, EventStatus, Severity
from app.services.evidence_manager import EvidenceManager
from app.ai.video.capture import OpenCVCapture
from app.ai.video.manager import VideoManager
from app.ai.detector.manager import DetectionManager
from app.ai.detector.yolo import YOLODetector
from app.ai.tracker.manager import TrackingManager
from app.core.logger import app_logger

router = APIRouter(prefix="/clips", tags=["Clips"])

# Resolve sample_clips relative to project root (backend/app/api/v1 -> project root)
try:
    _proj_root = pathlib.Path(__file__).resolve().parents[4]
    _cand = _proj_root / "sample_clips"
    SAMPLE_DIR = _cand if _cand.exists() else pathlib.Path("sample_clips")
except:
    SAMPLE_DIR = pathlib.Path("sample_clips")

# Live streaming sessions: session_id -> {queue, running, violations, frames, thread, preview_path}
live_sessions: dict[str, dict] = {}
live_lock = threading.Lock()

def _get_live_session(session_id: str):
    with live_lock:
        return live_sessions.get(session_id)

class ClipInfo(BaseModel):
    name: str
    path: str
    size_mb: float | None = None
    exists: bool

class ProcessRequest(BaseModel):
    clip_name: str
    session_id: Optional[str] = None
    max_frames: int = 80
    create_synthetic_violation: bool = True

class ProcessResponse(BaseModel):
    clip_name: str
    session_id: str
    frames_processed: int
    detections_avg: float | None = None
    tracks_avg: float | None = None
    buffer_size: int
    violations_created: list[str]
    preview_url: str | None = None
    message: str

@router.get("", response_model=list[ClipInfo], summary="List available clips")
async def list_clips():
    if not SAMPLE_DIR.exists():
        return []
    clips = []
    for p in sorted(SAMPLE_DIR.glob("*.mp4")):
        try:
            size_mb = round(p.stat().st_size / (1024*1024), 2)
        except:
            size_mb = None
        clips.append(ClipInfo(name=p.name, path=str(p.as_posix()), size_mb=size_mb, exists=True))
    return clips

@router.get("/preview/{session_id}", summary="Get annotated preview video")
async def get_preview(session_id: str):
    if "/" in session_id or "\\" in session_id or ".." in session_id:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    # Resolve preview path under clips/evidence/<session_id>/preview_annotated.mp4
    try:
        _root = pathlib.Path(__file__).resolve().parents[4] / "clips" / "evidence" / session_id / "preview_annotated.mp4"
        # Also check alternative relative
        if not _root.exists():
            _alt = pathlib.Path("clips/evidence") / session_id / "preview_annotated.mp4"
            if _alt.exists():
                _root = _alt
        if not _root.exists() or not _root.is_file():
            raise HTTPException(status_code=404, detail="Preview not found")
        # Security check
        base = (pathlib.Path(__file__).resolve().parents[4] / "clips" / "evidence").resolve()
        target = _root.resolve()
        if not str(target).startswith(str(base)):
            raise HTTPException(status_code=404, detail="Preview not found")
        return FileResponse(path=str(target), media_type="video/mp4", filename="preview_annotated.mp4")
    except HTTPException:
        raise
    except Exception as e:
        app_logger.exception(e)
        raise HTTPException(status_code=500, detail="Failed to load preview")

class LiveStartRequest(BaseModel):
    clip_name: str
    session_id: Optional[str] = None
    max_frames: int = 200

class LiveStartResponse(BaseModel):
    session_id: str
    clip_name: str
    stream_url: str
    preview_url: str
    status_url: str
    message: str

def _live_worker(clip_path: pathlib.Path, session_id: str, max_frames: int, clip_name: str):
    """Background worker for live streaming: runs pipeline, pushes JPEGs to queue, creates evidence."""
    sess = _get_live_session(session_id)
    if not sess:
        return
    q: queue.Queue = sess["queue"]
    video = VideoManager(OpenCVCapture(str(clip_path)), source_id=clip_name)
    detector = DetectionManager(YOLODetector())
    tracker = TrackingManager()
    preview_writer = None
    preview_path = None
    try:
        _root = pathlib.Path(__file__).resolve().parents[4] / "clips" / "evidence" / session_id
        _root.mkdir(parents=True, exist_ok=True)
        preview_path = _root / "preview_annotated.mp4"
        sess["preview_path"] = str(preview_path)
    except:
        preview_path = pathlib.Path("clips/evidence") / session_id / "preview_annotated.mp4"
        sess["preview_path"] = str(preview_path)

    if not video.start():
        sess["error"] = "Failed to open video"
        sess["running"] = False
        return
    detector.start()
    time.sleep(0.2)
    frames_ok = 0
    try:
        for _ in range(max_frames * 3):
            if not sess.get("running", True):
                break
            f = video.read()
            if f is None:
                if not video.is_running():
                    break
                time.sleep(0.005)
                continue
            dets = detector.detect(f)
            tracks = tracker.update(dets, frame=f)
            frames_ok += 1
            sess["frames_processed"] = frames_ok

            # Draw annotated frame
            try:
                if preview_writer is None:
                    h, w = f.image.shape[:2]
                    w_even = w - (w % 2)
                    h_even = h - (h % 2)
                    fps = getattr(f, 'fps', 25.0)
                    if not fps or fps < 1 or fps > 120:
                        fps = 25.0
                    sess["w_even"], sess["h_even"], sess["fps"] = w_even, h_even, fps
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    preview_writer = cv2.VideoWriter(str(preview_path), fourcc, fps, (w_even, h_even))
                if preview_writer is not None and preview_writer.isOpened():
                    annotated = f.image.copy()
                    for t in tracks:
                        try:
                            state = t.state.name if hasattr(t.state, 'name') else str(t.state)
                            color = (0, 255, 0) if state == "CONFIRMED" else (0, 255, 255) if state == "TENTATIVE" else (0, 0, 255)
                            box = t.motion.bbox
                            cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), color, 2)
                            cv2.putText(annotated, f"ID:{t.track_id}", (box.x1, max(0, box.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        except:
                            pass
                    for det in dets:
                        try:
                            kpts = det.metadata.get("keypoints") if hasattr(det, "metadata") and isinstance(det.metadata, dict) else None
                            if kpts is not None:
                                for idx in [0,1,2,3,4,5,6,7,8,9,10]:
                                    if idx < len(kpts):
                                        x,y,c = kpts[idx]
                                        if c > 0.3:
                                            cv2.circle(annotated, (int(x),int(y)), 4, (255,255,255), -1)
                                            cv2.circle(annotated, (int(x),int(y)), 2, (0,165,255), -1)
                        except:
                            pass
                    # Red border if violation already created and this frame is within violation window
                    # For live, we will create violations after some frames, so check sess violations
                    if sess.get("violations_created"):
                        # Draw red border to indicate malpractice window
                        cv2.rectangle(annotated, (0,0), (w_even-1, h_even-1), (0,0,255), 6)
                        cv2.putText(annotated, "MALPRACTICE DETECTED", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    if annotated.shape[1] != sess["w_even"] or annotated.shape[0] != sess["h_even"]:
                        annotated = cv2.resize(annotated, (sess["w_even"], sess["h_even"]))
                    preview_writer.write(annotated)
                    # Push JPEG for live MJPEG stream
                    # Resize to 960 width for bandwidth if original is 2560
                    disp = annotated
                    if disp.shape[1] > 960:
                        scale = 960 / disp.shape[1]
                        disp = cv2.resize(disp, (960, int(disp.shape[0]*scale)))
                    ok, jpeg = cv2.imencode('.jpg', disp, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    if ok:
                        # Non-blocking put, drop oldest if full
                        try:
                            q.put_nowait(jpeg.tobytes())
                        except queue.Full:
                            try: q.get_nowait()
                            except: pass
                            try: q.put_nowait(jpeg.tobytes())
                            except: pass
            except Exception as e:
                app_logger.warning(f"Live frame draw failed: {e}")

            if frames_ok >= max_frames:
                break
            time.sleep(0.005)
    finally:
        if preview_writer is not None:
            try: preview_writer.release()
            except: pass
        # Create synthetic violations after processing (so preview already has some frames)
        try:
            all_buf = video.evidence_buffer.get_all()
            if len(all_buf) >= 20:
                mid = len(all_buf)//2
                ev_start = all_buf[mid].timestamp
                ev_end = ev_start + 1.0
                student_id = 1
                try:
                    if tracker.tracks:
                        confirmed = [t for t in tracker.tracks if t.state.name == "CONFIRMED"]
                        student_id = confirmed[0].track_id if confirmed else tracker.tracks[0].track_id
                except: pass
                ev = MalpracticeEvent(
                    event_id=str(uuid.uuid4()),
                    rule_id="SUSPICIOUS_MOVEMENT",
                    student_id=int(student_id),
                    event_type="SUSPICIOUS_MOVEMENT",
                    severity=Severity.MEDIUM,
                    timestamp=float(ev_end),
                    frame_number=int(all_buf[mid].frame_id + 25),
                    confidence=0.89,
                    status=EventStatus.ENDED,
                    evidence={"reason": f"Live {clip_name} synthetic", "supporting_features": {"movement": 60}},
                    start_frame=int(all_buf[mid].frame_id),
                    end_frame=int(all_buf[mid].frame_id + 25),
                    duration=1.0,
                )
                violation_service.register_event(ev, session_id=session_id)
                sess["violations_created"].append(ev.event_id)
                mgr = EvidenceManager(video.evidence_buffer, base_output_dir="clips/evidence", wait_timeout_seconds=1.0)
                try: mgr.create_evidence(ev, session_id=session_id, source_id=clip_name, wait_for_post=False)
                except Exception as e: app_logger.exception(f"Live evidence failed: {e}")
        except Exception as e:
            app_logger.exception(f"Live synthetic violation failed: {e}")
        try: video.stop()
        except: pass
        try: detector.stop()
        except: pass
        sess["running"] = False
        sess["completed"] = True

@router.post("/live/start", response_model=LiveStartResponse, summary="Start live streaming clip")
async def start_live(req: LiveStartRequest):
    clip_name = req.clip_name.strip()
    if "/" in clip_name or "\\" in clip_name or ".." in clip_name:
        raise HTTPException(status_code=400, detail="Invalid clip_name")
    clip_path = SAMPLE_DIR / clip_name
    if not clip_path.exists() or not clip_path.is_file():
        raise HTTPException(status_code=404, detail=f"Clip not found: {clip_name}")
    session_id = (req.session_id or f"live_{clip_path.stem}_{uuid.uuid4().hex[:6]}").replace("/", "_").replace("\\", "_")
    max_frames = max(1, min(req.max_frames, 500))
    if session_id in live_sessions:
        raise HTTPException(status_code=409, detail="Session already exists")
    with live_lock:
        live_sessions[session_id] = {
            "queue": queue.Queue(maxsize=30),
            "running": True,
            "completed": False,
            "frames_processed": 0,
            "violations_created": [],
            "clip_name": clip_name,
            "preview_path": None,
            "error": None,
        }
    t = threading.Thread(target=_live_worker, args=(clip_path, session_id, max_frames, clip_name), daemon=True)
    with live_lock:
        live_sessions[session_id]["thread"] = t
    t.start()
    return LiveStartResponse(
        session_id=session_id,
        clip_name=clip_name,
        stream_url=f"/api/v1/clips/live/{session_id}/stream",
        preview_url=f"/api/v1/clips/preview/{session_id}",
        status_url=f"/api/v1/clips/live/{session_id}/status",
        message=f"Live streaming {clip_name} session {session_id}",
    )

@router.get("/live/{session_id}/stream", summary="Live MJPEG stream")
async def live_stream(session_id: str):
    if "/" in session_id or "\\" in session_id or ".." in session_id:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    sess = _get_live_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    async def gen():
        # MJPEG boundary
        while True:
            sess_cur = _get_live_session(session_id)
            if not sess_cur:
                break
            try:
                jpeg = sess_cur["queue"].get(timeout=0.5)
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            except queue.Empty:
                if not sess_cur.get("running", False) and sess_cur["queue"].empty():
                    # Send last frame and break
                    break
                # Keep-alive empty
                continue
            except Exception:
                break

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/live/{session_id}/status", summary="Live status")
async def live_status(session_id: str):
    if "/" in session_id or "\\" in session_id or ".." in session_id:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    sess = _get_live_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "clip_name": sess.get("clip_name"),
        "running": sess.get("running", False),
        "completed": sess.get("completed", False),
        "frames_processed": sess.get("frames_processed", 0),
        "violations_created": sess.get("violations_created", []),
        "preview_url": f"/api/v1/clips/preview/{session_id}" if sess.get("preview_path") else None,
        "error": sess.get("error"),
    }

@router.delete("/live/{session_id}", summary="Stop live session")
async def stop_live(session_id: str):
    if "/" in session_id or "\\" in session_id or ".." in session_id:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    sess = _get_live_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    sess["running"] = False
    return {"message": f"Session {session_id} stopping"}

@router.post("/process", response_model=ProcessResponse, summary="Process clip and generate evidence")
async def process_clip(req: ProcessRequest):
    clip_name = req.clip_name.strip()
    if "/" in clip_name or "\\" in clip_name or ".." in clip_name:
        raise HTTPException(status_code=400, detail="Invalid clip_name")
    clip_path = SAMPLE_DIR / clip_name
    if not clip_path.exists() or not clip_path.is_file():
        raise HTTPException(status_code=404, detail=f"Clip not found: {clip_name}")
    session_id = (req.session_id or f"clip_{clip_path.stem}_{uuid.uuid4().hex[:6]}").replace("/", "_").replace("\\", "_")
    max_frames = max(1, min(req.max_frames, 500))

    # Run pipeline for max_frames
    video = VideoManager(OpenCVCapture(str(clip_path)), source_id=clip_name)
    detector = DetectionManager(YOLODetector())
    tracker = TrackingManager()

    if not video.start():
        raise HTTPException(status_code=500, detail="Failed to open video")
    detector.start()
    time.sleep(0.3)

    frames_ok = 0
    dets_total = 0
    tracks_total = 0
    # Preview video writer (annotated live visualization)
    preview_writer = None
    preview_path = None
    try:
        # Resolve preview path: <project_root>/clips/evidence/<session_id>/preview_annotated.mp4
        try:
            _root = pathlib.Path(__file__).resolve().parents[4] / "clips" / "evidence" / session_id
        except:
            _root = pathlib.Path("clips/evidence") / session_id
        _root.mkdir(parents=True, exist_ok=True)
        preview_path = _root / "preview_annotated.mp4"
        # Use same pattern as smoke test: detect every 2 frames throttle
        for _ in range(max_frames * 3):  # allow queue to fill
            f = video.read()
            if f is None:
                if not video.is_running():
                    break
                time.sleep(0.005)
                continue
            dets = detector.detect(f)
            tracks = tracker.update(dets, frame=f)
            frames_ok += 1
            dets_total += len(dets)
            tracks_total += len(tracks)

            # --- Generate annotated preview frame (live visualization) ---
            try:
                # Initialize writer on first frame
                if preview_writer is None:
                    h, w = f.image.shape[:2]
                    # Use even dimensions for mp4
                    w_even = w - (w % 2)
                    h_even = h - (h % 2)
                    fps = getattr(f, 'fps', 25.0)
                    if not fps or fps < 1 or fps > 120:
                        fps = 25.0
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    preview_writer = cv2.VideoWriter(str(preview_path), fourcc, fps, (w_even, h_even))
                if preview_writer is not None and preview_writer.isOpened():
                    annotated = f.image.copy()
                    # Draw tracks (frozen tracker visualization)
                    for t in tracks:
                        try:
                            state = t.state.name if hasattr(t.state, 'name') else str(t.state)
                            if state == "CONFIRMED":
                                color = (0, 255, 0)
                            elif state == "TENTATIVE":
                                color = (0, 255, 255)
                            else:
                                color = (0, 0, 255)
                            box = t.motion.bbox
                            cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), color, 2)
                            cv2.putText(annotated, f"ID:{t.track_id}", (box.x1, max(0, box.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            # Draw YOLO pose dots if available in detection metadata
                            # Find matching detection for this track approx by bbox overlap
                            # Instead, draw from dets directly (pose dots)
                        except:
                            pass
                    # Draw YOLO pose dots from dets (nose, eyes, ears, shoulders, elbows, wrists)
                    for det in dets:
                        try:
                            kpts = det.metadata.get("keypoints") if hasattr(det, "metadata") and isinstance(det.metadata, dict) else None
                            if kpts is not None:
                                for idx in [0,1,2,3,4,5,6,7,8,9,10]:
                                    if idx < len(kpts):
                                        x,y,c = kpts[idx]
                                        if c > 0.3:
                                            cv2.circle(annotated, (int(x),int(y)), 4, (255,255,255), -1)
                                            cv2.circle(annotated, (int(x),int(y)), 2, (0,165,255), -1)
                        except:
                            pass
                    # Ensure size matches writer
                    if annotated.shape[1] != w_even or annotated.shape[0] != h_even:
                        annotated = cv2.resize(annotated, (w_even, h_even))
                    preview_writer.write(annotated)
            except Exception as e:
                app_logger.warning(f"Preview frame draw failed: {e}")

            if frames_ok >= max_frames:
                break
            time.sleep(0.005)
    finally:
        if preview_writer is not None:
            try:
                preview_writer.release()
            except:
                pass
        # Keep video buffer for evidence creation

    buffer_size = video.evidence_buffer.size()
    # Give buffer a moment to settle
    time.sleep(0.1)

    violations_created: list[str] = []

    if frames_ok == 0:
        video.stop()
        detector.stop()
        raise HTTPException(status_code=500, detail="No frames processed (video read failed)")

    if req.create_synthetic_violation:
        # Create 1-2 synthetic violations using actual buffer timestamps so evidence is valid
        try:
            all_buf = video.evidence_buffer.get_all()
            if len(all_buf) >= 20:
                mid = len(all_buf)//2
                ev_start = all_buf[mid].timestamp
                ev_end = ev_start + 1.0
                # Create event for a tracked ID if available, else student 1
                student_id = 1
                try:
                    if tracker.tracks:
                        # Pick a CONFIRMED track id
                        confirmed = [t for t in tracker.tracks if t.state.name == "CONFIRMED"]
                        if confirmed:
                            student_id = confirmed[0].track_id
                        else:
                            student_id = tracker.tracks[0].track_id
                except:
                    pass
                ev = MalpracticeEvent(
                    event_id=str(uuid.uuid4()),
                    rule_id="SUSPICIOUS_MOVEMENT",
                    student_id=int(student_id),
                    event_type="SUSPICIOUS_MOVEMENT",
                    severity=Severity.MEDIUM,
                    timestamp=float(ev_end),
                    frame_number=int(all_buf[mid].frame_id + 25),
                    confidence=0.89,
                    status=EventStatus.ENDED,
                    evidence={"reason": f"Clip test {clip_name} synthetic", "supporting_features": {"movement": 60}},
                    start_frame=int(all_buf[mid].frame_id),
                    end_frame=int(all_buf[mid].frame_id + 25),
                    duration=1.0,
                )
                # Register
                violation_service.register_event(ev, session_id=session_id)
                violations_created.append(ev.event_id)
                # Create evidence via EvidenceManager using same buffer
                # Use base_dir clips/evidence (default) so frontend can discover
                mgr = EvidenceManager(video.evidence_buffer, base_output_dir="clips/evidence", wait_timeout_seconds=1.0)
                try:
                    mgr.create_evidence(ev, session_id=session_id, source_id=clip_name, wait_for_post=False)
                except Exception as e:
                    app_logger.exception(f"Evidence creation failed for {clip_name}: {e}")
                # Optionally create second violation for another student to demo multiple
                if len(all_buf) >= 40 and frames_ok >= 40:
                    # Second event 0.5s later
                    ev2_start = all_buf[mid+10].timestamp if mid+10 < len(all_buf) else ev_start + 0.5
                    ev2_end = ev2_start + 0.8
                    student2 = student_id + 1 if isinstance(student_id, int) else 2
                    ev2 = MalpracticeEvent(
                        event_id=str(uuid.uuid4()),
                        rule_id="SUSPICIOUS_MOVEMENT",
                        student_id=int(student2),
                        event_type="SUSPICIOUS_MOVEMENT",
                        severity=Severity.HIGH,
                        timestamp=float(ev2_end),
                        frame_number=int(all_buf[mid].frame_id + 40),
                        confidence=0.86,
                        status=EventStatus.ENDED,
                        evidence={"reason": f"Clip test {clip_name} second synthetic", "supporting_features": {"movement": 70}},
                        start_frame=int(all_buf[mid].frame_id + 10),
                        end_frame=int(all_buf[mid].frame_id + 35),
                        duration=0.8,
                    )
                    violation_service.register_event(ev2, session_id=session_id)
                    violations_created.append(ev2.event_id)
                    try:
                        mgr.create_evidence(ev2, session_id=session_id, source_id=clip_name, wait_for_post=False)
                    except Exception as e:
                        app_logger.exception(f"Second evidence failed: {e}")
            else:
                app_logger.warning(f"Buffer too small for synthetic violation: {len(all_buf)}")
        except Exception as e:
            app_logger.exception(f"Synthetic violation creation failed: {e}")

    # Cleanup
    try:
        video.stop()
    except: pass
    try:
        detector.stop()
    except: pass

    avg_dets = round(dets_total / max(1, frames_ok), 1) if frames_ok else None
    avg_tracks = round(tracks_total / max(1, frames_ok), 1) if frames_ok else None

    # Determine preview URL if file exists
    preview_url = None
    try:
        if preview_path is not None and pathlib.Path(preview_path).exists() and pathlib.Path(preview_path).stat().st_size > 0:
            preview_url = f"/api/v1/clips/preview/{session_id}"
    except:
        pass

    return ProcessResponse(
        clip_name=clip_name,
        session_id=session_id,
        frames_processed=frames_ok,
        detections_avg=avg_dets,
        tracks_avg=avg_tracks,
        buffer_size=buffer_size,
        violations_created=violations_created,
        preview_url=preview_url,
        message=f"Processed {frames_ok} frames from {clip_name}, created {len(violations_created)} violation(s) in session {session_id}",
    )
@router.get("/history")
def list_historical_evidence(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List evidence clips saved on disk, including older unregistered events."""
    from app.services.violation_service import violation_service

    base_dir = Path(violation_service.base_dir).resolve()
    if not base_dir.is_dir():
        return {"items": [], "total": 0, "limit": limit, "offset": offset}

    items = []

    try:
        session_dirs = list(base_dir.iterdir())
    except OSError:
        raise HTTPException(status_code=500, detail="Unable to scan evidence storage")

    for session_dir in session_dirs:
        if not session_dir.is_dir():
            continue

        try:
            session_dir.resolve().relative_to(base_dir)
        except (ValueError, OSError):
            continue

        try:
            event_dirs = list(session_dir.iterdir())
        except OSError:
            continue

        for event_dir in event_dirs:
            if not event_dir.is_dir():
                continue

            try:
                event_dir.resolve().relative_to(base_dir)
            except (ValueError, OSError):
                continue

            video_path = event_dir / "original_evidence.mp4"
            metadata_path = event_dir / "metadata.json"

            video_exists = video_path.is_file()
            metadata_exists = metadata_path.is_file()

            # Ignore preview files and unrelated directories.
            if not video_exists and not metadata_exists:
                continue

            metadata = None
            if metadata_exists:
                try:
                    with metadata_path.open("r", encoding="utf-8") as handle:
                        loaded = json.load(handle)
                    if isinstance(loaded, dict):
                        metadata = loaded
                except (OSError, ValueError):
                    metadata = None

            try:
                modified_at = max(
                    path.stat().st_mtime
                    for path in (video_path, metadata_path)
                    if path.is_file()
                )
                created_at = datetime.fromtimestamp(
                    modified_at, tz=timezone.utc
                ).isoformat()
            except (OSError, ValueError):
                created_at = None
                modified_at = 0

            items.append({
                "event_id": event_dir.name,
                "session_id": session_dir.name,
                "video_exists": video_exists,
                "metadata_exists": metadata_exists,
                "video_url": (
                    f"/api/v1/clips/history/"
                    f"{session_dir.name}/{event_dir.name}/video"
                    if video_exists else None
                ),
                "metadata": metadata,
                "created_at": created_at,
                "_sort_time": modified_at,
            })

    # Newest evidence first; deterministic tie-breaking.
    items.sort(
        key=lambda item: (item["_sort_time"], item["event_id"]),
        reverse=True,
    )
    total = len(items)
    page = items[offset:offset + limit]

    for item in page:
        item.pop("_sort_time", None)

    return {
        "items": page,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/history/{session_id}/{event_id}/video")
def get_historical_evidence_video(session_id: str, event_id: str):
    """Serve a saved evidence video without requiring a database event."""
    from app.services.violation_service import violation_service

    # Only accept single path segments.
    for value in (session_id, event_id):
        if (
            not value
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
        ):
            raise HTTPException(status_code=400, detail="Invalid evidence identifier")

    base_dir = Path(violation_service.base_dir).resolve()
    video_path = (
        base_dir / session_id / event_id / "original_evidence.mp4"
    ).resolve()

    try:
        video_path.relative_to(base_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid evidence path")

    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Evidence video not found")

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename="evidence.mp4",
        content_disposition_type="inline",
        headers={"Cache-Control": "no-cache"},
    )
