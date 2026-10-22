"""

Evidence Manager - Phase 7

Converts MalpracticeEvent -> original_evidence.mp4 + metadata.json

Uses RollingOriginalBuffer for 2s pre/post with actual timestamps.

"""


import os
import shutil
import subprocess
import json

import time

import uuid

import threading
import traceback

import tempfile

import shutil

from pathlib import Path

from typing import Optional, List

from datetime import datetime, timezone



import cv2



from app.ai.video.rolling_buffer import RollingOriginalBuffer

from app.services.rule_engine import MalpracticeEvent

from app.config.settings import settings





def _get_event_interval(event: MalpracticeEvent):

    """

    Extract event_start, event_end from MalpracticeEvent flexibly.

    Supports multiple attribute names for frozen phases 1-6.

    """

    # Try explicit attributes first

    event_start = None

    event_end = None



    # Possible names

    for attr in ["event_start", "start_time", "start_timestamp"]:

        if hasattr(event, attr):

            val = getattr(event, attr)

            if val is not None:

                try:

                    event_start = float(val)

                    break

                except:  # noqa

                    pass



    for attr in ["event_end", "end_time", "end_timestamp"]:

        if hasattr(event, attr):

            val = getattr(event, attr)

            if val is not None:

                try:

                    event_end = float(val)

                    break

                except:  # noqa

                    pass



    # Fallback: use timestamp + duration

    # MalpracticeEvent has timestamp (end) and duration (timestamp - candidate_start)

    if event_start is None:

        if hasattr(event, "duration") and event.duration and hasattr(event, "timestamp"):

            try:

                event_start = float(event.timestamp) - float(event.duration)

            except:  # noqa

                event_start = float(event.timestamp) if hasattr(event, "timestamp") else time.time()

        elif hasattr(event, "timestamp"):

            event_start = float(event.timestamp)

        else:

            event_start = time.time()



    if event_end is None:

        if hasattr(event, "timestamp"):

            event_end = float(event.timestamp)

        else:

            event_end = event_start



    # Ensure start <= end

    if event_start > event_end:

        event_start, event_end = event_end, event_start



    return float(event_start), float(event_end)





def _build_student_crop_windows(frames, student_id: int, padding_ratio: float):

    """Build a fixed-size, padded crop window centered on the student's box per frame."""

    boxes = []

    for frame in frames:

        student_boxes = (getattr(frame, "metadata", None) or {}).get("student_boxes", {})

        box = student_boxes.get(str(student_id), student_boxes.get(student_id)) if isinstance(student_boxes, dict) else None

        if isinstance(box, (list, tuple)) and len(box) == 4:

            try:

                x1, y1, x2, y2 = (float(value) for value in box)

                if x2 > x1 and y2 > y1:

                    boxes.append((x1, y1, x2, y2))

                    continue

            except (TypeError, ValueError):

                pass

        boxes.append(None)



    available = [box for box in boxes if box is not None]

    if not available or not frames:

        return None



    padding_ratio = min(max(float(padding_ratio), 0.0), 1.0)

    first_h, first_w = frames[0].image.shape[:2]

    crop_w = min(first_w, max(2, int(round(max(box[2] - box[0] for box in available) * (1 + 2 * padding_ratio)))))

    crop_h = min(first_h, max(2, int(round(max(box[3] - box[1] for box in available) * (1 + 2 * padding_ratio)))))

    crop_w -= crop_w % 2

    crop_h -= crop_h % 2



    windows = []

    for index, frame in enumerate(frames):

        h, w = frame.image.shape[:2]

        box = boxes[index]

        if box is None:

            nearest_index = min(

                (idx for idx, candidate in enumerate(boxes) if candidate is not None),

                key=lambda idx: abs(idx - index),

            )

            box = boxes[nearest_index]

        x1, y1, x2, y2 = box

        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        frame_crop_w = min(crop_w, w - (w % 2))

        frame_crop_h = min(crop_h, h - (h % 2))

        left = max(0, min(w - frame_crop_w, int(round(center_x - frame_crop_w / 2))))

        top = max(0, min(h - frame_crop_h, int(round(center_y - frame_crop_h / 2))))

        windows.append((left, top, left + frame_crop_w, top + frame_crop_h))

    return windows





class EvidenceManager:

    """

    Phase 7 Evidence Manager.

    Thread-safe, supports multiple students/events simultaneously.

    Does NOT modify RuleEngine thresholds or create new rules.

    """



    def __init__(

        self,

        rolling_buffer: RollingOriginalBuffer,

        base_output_dir: str | Path = "clips/evidence",

        pre_seconds: float | None = None,

        post_seconds: float | None = None,

        wait_timeout_seconds: float = 10.0,

        wait_poll_interval: float = 0.05,

    ):

        self.buffer = rolling_buffer

        # Resolve to absolute project root if relative "clips/evidence"

        _base = Path(base_output_dir)

        if not _base.is_absolute() and _base == Path("clips/evidence"):

            try:

                _root = Path(__file__).resolve().parents[3]

                _cand = _root / "clips" / "evidence"

                _base = _cand

            except:

                pass

        self.base_dir = Path(_base)

        # Use settings if not provided

        self.pre_seconds = float(pre_seconds) if pre_seconds is not None else float(settings.EVIDENCE_PRE_SECONDS)

        self.post_seconds = float(post_seconds) if post_seconds is not None else float(settings.EVIDENCE_POST_SECONDS)

        self.wait_timeout = float(wait_timeout_seconds)

        self.poll_interval = float(wait_poll_interval)

        self._lock = threading.RLock()

        # For deduplication check (optional)

        self._created_events: set[str] = set()



    def _evidence_interval(self, event: MalpracticeEvent):

        event_start, event_end = _get_event_interval(event)

        evidence_start = event_start - self.pre_seconds

        evidence_end = event_end + self.post_seconds

        return event_start, event_end, evidence_start, evidence_end



    def create_evidence(

        self,

        event: MalpracticeEvent,

        session_id: str = "default_session",

        source_id: Optional[str] = None,

        wait_for_post: bool = True,

        student_snapshot: Optional[dict] = None,

        crop_student_id: Optional[int] = None,

        crop_padding_ratio: float = 0.25,

    ) -> dict:

        """

        Create evidence for a single MalpracticeEvent.

        Returns metadata dict (also written to metadata.json).

        Thread-safe per-event (event_id unique path, no overwrite).



        wait_for_post: if True, wait until buffer has frames through evidence_end (up to timeout)

        """

        event_start, event_end, evidence_start, evidence_end = self._evidence_interval(event)



        # Hold the pre-event frames before waiting for post-event footage. The

        # rolling buffer may evict them while the post window is being captured.

        pre_event_frames = self.buffer.get_range(evidence_start, event_end)



        
        # Determine whether the requested post-event window is available.
        post_window_complete = False

        if wait_for_post:
            wait_until = getattr(self.buffer, "wait_until", None)

            if callable(wait_until):
                post_window_complete = bool(
                    wait_until(evidence_end, timeout=self.wait_timeout)
                )
            else:
                deadline = time.monotonic() + self.wait_timeout

                while time.monotonic() < deadline:
                    latest = self.buffer.get_latest_timestamp()
                    if latest is not None and latest >= evidence_end:
                        post_window_complete = True
                        break
                    time.sleep(self.poll_interval)

        else:
            latest = self.buffer.get_latest_timestamp()
            post_window_complete = (
                latest is not None and latest >= evidence_end
            )




        # Combine the retained pre-event snapshot with frames captured after the

        # configured post window. De-duplicate by frame id and preserve timeline.

        if wait_for_post:

            post_event_frames = self.buffer.get_range(event_end, evidence_end)

            frames_by_id = {frame.frame_id: frame for frame in pre_event_frames}

            frames_by_id.update({frame.frame_id: frame for frame in post_event_frames})

            frames = sorted(frames_by_id.values(), key=lambda frame: (frame.timestamp, frame.frame_id))

        else:

            frames = self.buffer.get_range(evidence_start, evidence_end)

        crop_windows = _build_student_crop_windows(frames, int(crop_student_id), crop_padding_ratio) if crop_student_id is not None else None



        # Handle non-contiguous / missing frame_ids: already filtered by timestamp, not frame_id

        # Frames are already sorted by timestamp insertion order (buffer maintains order)



       
        # Calculate actual durations using frames inside the requested interval.
        requested_pre = self.pre_seconds
        requested_post = self.post_seconds

        actual_pre = 0.0
        actual_post = 0.0
        actual_duration = 0.0
        fps = 25.0
        width = 0
        height = 0
        frame_start = None
        frame_end = None

        if frames:
            frames = sorted(
                frames,
                key=lambda frame: (float(frame.timestamp), frame.frame_id),
            )

            # Enforce the requested evidence boundaries.
            frames = [
                frame for frame in frames
                if evidence_start <= float(frame.timestamp) <= evidence_end
            ]

        if frames:
            timestamps = [float(frame.timestamp) for frame in frames]
            earliest_ts = timestamps[0]
            latest_ts = timestamps[-1]

            fps_values = [
                float(frame.fps)
                for frame in frames
                if frame.fps and float(frame.fps) > 1
            ]
            if fps_values:
                fps = sorted(fps_values)[len(fps_values) // 2]

            height, width = frames[0].image.shape[:2]
            width, height = int(width), int(height)

            
            # Measure actual coverage independently on each side of the event.
            # Clip the measured timestamps to the requested evidence boundaries.
            pre_start = max(evidence_start, earliest_ts)
            pre_end = min(event_start, latest_ts)

            post_start = max(event_end, earliest_ts)
            post_end = min(evidence_end, latest_ts)

            actual_pre = min(
                requested_pre,
                max(0.0, pre_end - pre_start),
            )
            actual_post = min(
                requested_post,
                max(0.0, post_end - post_start),
            )

            actual_duration = max(0.0, latest_ts - earliest_ts)

            frame_start = frames[0].frame_id
            frame_end = frames[-1].frame_id
        else:
            frames = []




        # Prepare output directory: base/session_id/event_id

        event_id = str(getattr(event, "event_id", str(uuid.uuid4())))

        # Ensure event_id safe for path

        safe_event_id = str(event_id).replace("/", "_").replace("\\", "_")

        safe_session = str(session_id).replace("/", "_").replace("\\", "_")

        out_dir = self.base_dir / safe_session / safe_event_id

        out_dir.mkdir(parents=True, exist_ok=True)

        final_video = out_dir / "original_evidence.mp4"

        final_meta = out_dir / "metadata.json"



        # Prevent overwrite? Spec says do not overwrite, event_id unique ensures no overwrite

        # But if evidence already exists for same event_id, we should not overwrite silently

        # We'll write with _lock and check existence; if exists, we return existing metadata

        # To avoid race for simultaneous same event_id, use lock

        with self._lock:

            if final_video.exists() and final_meta.exists():

                # Already created, return existing metadata

                try:

                    with open(final_meta, "r", encoding="utf-8") as f:

                        return json.load(f)

                except:  # noqa

                    pass



       
        # Write evidence video using H.264 for browser compatibility.
        tmp_video = out_dir / "_tmp_original_evidence.mp4"
        tmp_mjpeg = out_dir / "_tmp_original_evidence.avi"
        tmp_meta = out_dir / "metadata.json.tmp"

        video_write_success = False
        encoded_frame_count = 0
        fps_write = fps if fps and 1 < fps < 240 else 25.0
        video_write_error = None
        original_evidence_path = str(final_video.as_posix())

        writer = None
        stage = "initialization"
        output_index = None
        source_index = None
        last_image_shape = None
        last_image_dtype = None
        last_image_contiguous = None
        submitted_frame_count = 0

        try:
            if not frames:
                raise RuntimeError("No evidence frames available to encode")

            if crop_student_id is not None and crop_windows is None:
                raise RuntimeError(
                    f"No tracked bounding boxes available to crop Student #{crop_student_id}"
                )

            if crop_windows:
                width = crop_windows[0][2] - crop_windows[0][0]
                height = crop_windows[0][3] - crop_windows[0][1]
            else:
                height, width = frames[0].image.shape[:2]

            w_even = width - (width % 2)
            h_even = height - (height % 2)

            if w_even <= 0 or h_even <= 0:
                raise RuntimeError(f"Invalid evidence dimensions: {width}x{height}")

            # Stage 1: write frames to a temporary AVI using MJPEG.
            stage = "open_mjpeg_writer"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(
                str(tmp_mjpeg), fourcc, fps_write, (w_even, h_even)
            )

            if not writer.isOpened():
                raise RuntimeError("OpenCV could not open the temporary MJPEG writer")

            source_timestamps = [float(fr.timestamp) for fr in frames]
            timeline_start = source_timestamps[0]
            timeline_end = source_timestamps[-1]
            output_frame_count = max(
                1, int(round((timeline_end - timeline_start) * fps_write)) + 1
            )

            source_index = 0

            for output_index in range(output_frame_count):
                stage = "select_source_frame"
                target_timestamp = timeline_start + output_index / fps_write

                while source_index + 1 < len(frames):
                    current_distance = abs(
                        source_timestamps[source_index] - target_timestamp
                    )
                    next_distance = abs(
                        source_timestamps[source_index + 1] - target_timestamp
                    )
                    if next_distance < current_distance:
                        source_index += 1
                    else:
                        break

                img = frames[source_index].image

                if crop_windows is not None:
                    stage = "crop_frame"
                    x1, y1, x2, y2 = crop_windows[source_index]
                    img = img[y1:y2, x1:x2]

                if img.shape[1] != w_even or img.shape[0] != h_even:
                    stage = "resize_frame"
                    img = cv2.resize(img, (w_even, h_even))

                last_image_shape = img.shape
                last_image_dtype = str(img.dtype)
                last_image_contiguous = bool(img.flags.c_contiguous)

                if not last_image_contiguous:
                    img = img.copy()

                stage = "write_mjpeg_frame"
                writer.write(img)
                submitted_frame_count += 1

            writer.release()
            writer = None
            encoded_frame_count = output_frame_count

            if not tmp_mjpeg.exists() or tmp_mjpeg.stat().st_size == 0:
                raise RuntimeError("Temporary MJPEG video was not created")

            # Stage 2: transcode to H.264 MP4 for browser playback.
            stage = "ffmpeg_h264_conversion"
            ffmpeg_path = shutil.which("ffmpeg")
            if not ffmpeg_path:
                raise RuntimeError(
                    "FFmpeg was not found. Install FFmpeg and ensure it is on PATH."
                )

            result = subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-v", "error",
                    "-i", str(tmp_mjpeg),
                    "-an",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-r", str(fps_write),
                    str(tmp_video),
                ],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg conversion failed: {result.stderr.strip()}"
                )

            if not tmp_video.exists() or tmp_video.stat().st_size == 0:
                raise RuntimeError("FFmpeg did not produce a valid MP4 file")

            # Verify the converted file can be opened and decoded.
            stage = "validate_h264_video"
            check = cv2.VideoCapture(str(tmp_video))
            try:
                opened = bool(check.isOpened())
                decoded_ok, _decoded_frame = (
                    check.read() if opened else (False, None)
                )
            finally:
                check.release()

            if not opened or not decoded_ok:
                raise RuntimeError("H.264 MP4 failed decode validation")

            # Replace the final video only after conversion succeeds.
            stage = "replace_final_video"
            os.replace(str(tmp_video), str(final_video))
            video_write_success = True

        except Exception as e:
            video_write_error = (
                f"stage={stage}; output_index={output_index}; "
                f"source_index={source_index}; image_shape={last_image_shape}; "
                f"image_dtype={last_image_dtype}; "
                f"image_contiguous={last_image_contiguous}; "
                f"submitted_frames={submitted_frame_count}; error={e!r}; "
                f"traceback={traceback.format_exc()}"
            )
            original_evidence_path = None
            video_write_success = False

        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass

            for temporary_file in (tmp_video, tmp_mjpeg):
                try:
                    if temporary_file.exists():
                        temporary_file.unlink()
                except OSError:
                    pass

        if not video_write_success:
            original_evidence_path = None




        # Build metadata

        # Extract fields from event flexibly, use None if unavailable (not fake)

        rule_id = getattr(event, "rule_id", None)

        event_type = getattr(event, "event_type", None)

        confidence = getattr(event, "confidence", None)

        # reason may be in evidence dict or reason attr

        reason = getattr(event, "reason", None)

        if reason is None and hasattr(event, "evidence") and isinstance(event.evidence, dict):

            reason = event.evidence.get("reason") or event.evidence.get("supporting_features")

        supporting = getattr(event, "evidence", None)

        if supporting is None:

            supporting = getattr(event, "supporting_features", None)

        # Status

        status = getattr(event, "status", None)

        try:

            status = status.value if hasattr(status, "value") else str(status) if status is not None else None

        except:  # noqa

            status = str(status) if status is not None else None



        # Frame numbers from event's own start/end if available

        evt_start_frame = getattr(event, "start_frame", None)

        evt_end_frame = getattr(event, "end_frame", None)

        # If None, use frame_start/end from evidence

        if evt_start_frame is None:

            evt_start_frame = frame_start

        if evt_end_frame is None:

            evt_end_frame = frame_end



        created_at = datetime.now(timezone.utc).isoformat()



        metadata = {

            "event_id": event_id,

            "student_id": getattr(event, "student_id", None),

            "rule_id": rule_id,

            "event_type": event_type,

            "status": status,

            "event_start": event_start,

            "event_end": event_end,

            "evidence_start": evidence_start,

            "evidence_end": evidence_end,

            "requested_pre_duration": requested_pre,

            "actual_pre_duration": round(float(actual_pre), 3),

            "requested_post_duration": requested_post,

            "actual_post_duration": round(float(actual_post), 3),

            "duration": round(float(actual_duration), 3) if frames else 0.0,

            "event_duration": round(float(event_end - event_start), 3),

            "confidence": confidence,

            "reason": reason,

            "supporting_features": supporting,

            "source_id": source_id,
            "session_id": session_id,

            "frame_start": frame_start,

            "frame_end": frame_end,

            "event_start_frame": evt_start_frame,

            "event_end_frame": evt_end_frame,

            "fps": round(float(fps), 2) if fps else None,

            "width": width if width else None,

            "height": height if height else None,

            "frame_count": encoded_frame_count if video_write_success else 0,
            "source_frame_count": len(frames),
            "encoded_frame_count": encoded_frame_count if video_write_success else 0,
            "encoded_fps": round(float(fps_write), 2) if video_write_success else None,
            "encoded_duration": round((encoded_frame_count - 1) / fps_write, 3) if video_write_success and encoded_frame_count else 0.0,

            "frame_ids": [int(frame.frame_id) for frame in frames],

            "post_window_complete": bool(post_window_complete),

            
            "evidence_complete": bool(
                video_write_success
                and actual_pre + 1e-3 >= requested_pre
                and actual_post + 1e-3 >= requested_post
                and post_window_complete
            ),


            "created_at": created_at,

            "original_evidence_path": original_evidence_path if video_write_success else None,

            "annotated_evidence_path": None,

            "evidence_pre_seconds": self.pre_seconds,

            "evidence_post_seconds": self.post_seconds,

            "evidence_crop": {

                "student_id": int(crop_student_id),

                "padding_ratio": float(crop_padding_ratio),

                "mode": "tracked_student_bbox",

                "status": "cropped" if crop_windows else "unavailable",

            } if crop_student_id is not None else None,

            "video_write_error": video_write_error,

        }

        if student_snapshot is not None:

            # Keep the detection context captured at the moment this event fired.

            metadata["student_snapshot"] = student_snapshot



        # Atomic metadata write

        try:

            with open(tmp_meta, "w", encoding="utf-8") as f:

                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # Atomic rename

            if final_meta.exists():

                try:

                    final_meta.unlink()

                except:  # noqa

                    pass

            tmp_meta.replace(final_meta)

        except Exception:

            try:

                if tmp_meta.exists():

                    tmp_meta.unlink()

            except:  # noqa

                pass

            # Do not leave corrupt video if metadata failed

            if video_write_success and final_video.exists():

                try:

                    final_video.unlink()

                except:  # noqa

                    pass

            raise



        # Mark created

        with self._lock:

            self._created_events.add(event_id)



        from app.services.violation_service import violation_service

        violation_service.store_evidence(

            event_id,

            session_id,

            metadata,

            final_video if video_write_success else None,

        )



        return metadata



    def create_evidence_batch(self, events: List[MalpracticeEvent], session_id: str = "default_session", source_id: Optional[str] = None) -> List[dict]:

        """Batch helper for multiple events (separate clips, no dedup)."""

        results = []

        for ev in events:

            meta = self.create_evidence(ev, session_id=session_id, source_id=source_id)

            results.append(meta)

        return results



    def get_info(self) -> dict:

        return {

            "pre_seconds": self.pre_seconds,

            "post_seconds": self.post_seconds,

            "wait_timeout": self.wait_timeout,

            "base_dir": str(self.base_dir),

            "created_events": len(self._created_events),

        }
 