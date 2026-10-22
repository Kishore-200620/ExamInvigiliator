import queue
import threading
import time
from typing import Optional

from app.ai.video.base import VideoSource
from app.ai.video.frame import Frame


class VideoEngine:
    """
    Production-grade video engine.

    Responsibilities:
    - Run a background capture thread.
    - Maintain a frame queue for downstream processing.
    - Convert raw OpenCV frames into Frame objects.
    - Calculate FPS.
    - Feed original frames into the rolling evidence buffer.
    - Use media-timeline timestamps for prerecorded video files.
    - Use wall-clock timestamps for live camera sources.
    """

    def __init__(
        self,
        source: VideoSource,
        source_id: str = "camera_0",
        queue_size: int = 30,
        evidence_buffer=None,
        preserve_all_frames: bool = False,
    ):
        self.source = source
        self.source_id = source_id

        # Preserve the existing queue behavior.
        self.frame_queue = queue.Queue(maxsize=queue_size)

        self.running = False
        self.thread = None
        self.frame_id = 0

        self.previous_time = time.perf_counter()
        self._fps_ema = 30.0

        # Original frames are buffered before entering the processing queue.
        self.evidence_buffer = evidence_buffer
        self.preserve_all_frames = preserve_all_frames

        # Video timing configuration.
        self.native_fps = 30.0
        self.is_video_file = False
        self.timestamp_origin = None

    def start(self) -> bool:
        """Open the source and start the capture thread."""

        if not self.source.open():
            return False

        # Reset per-run state.
        self.frame_id = 0
        self.previous_time = time.perf_counter()
        self._fps_ema = 30.0

        self.native_fps = 30.0
        self.is_video_file = False

        cap = getattr(self.source, "capture", None)

        # Read source properties after opening the source.
        if cap is not None:
            try:
                native_fps = float(cap.get(5))  # cv2.CAP_PROP_FPS

                if 1.0 < native_fps < 240.0:
                    self.native_fps = native_fps

                frame_count = float(cap.get(7))  # cv2.CAP_PROP_FRAME_COUNT

                # A positive frame count identifies a finite video source
                # for common OpenCV-backed file inputs.
                self.is_video_file = (
                    frame_count > 0
                    and 1.0 < self.native_fps < 240.0
                )
            except (TypeError, ValueError, AttributeError):
                self.is_video_file = False
            except Exception:
                self.is_video_file = False

        # Use an epoch-like origin so timestamps remain compatible with
        # existing event/evidence code that expects Unix timestamps.
        self.timestamp_origin = time.time()

        self.running = True

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"VideoEngine-{self.source_id}",
        )
        self.thread.start()

        return True

    def _capture_loop(self):
        """Capture frames, timestamp them, buffer originals, and queue them."""

        while self.running:
            loop_start = time.perf_counter()

            success, image = self.source.read()

            if not success:
                self.running = False
                break

            self.frame_id += 1

            current_time = time.perf_counter()
            elapsed_since_previous = max(
                1e-6,
                current_time - self.previous_time,
            )

            if self.is_video_file:
                # Use the source video's timeline. Queue blocking must not
                # artificially increase event durations.
                frame_timestamp = (
                    self.timestamp_origin
                    + (self.frame_id - 1) / self.native_fps
                )
                instant_fps = self.native_fps
            else:
                # Live camera timestamps represent actual wall-clock time.
                frame_timestamp = time.time()
                instant_fps = 1.0 / elapsed_since_previous

            self._fps_ema = (
                0.9 * self._fps_ema
                + 0.1 * instant_fps
            )
            fps = self._fps_ema

            self.previous_time = current_time

            frame = Frame(
                image=image,
                frame_id=self.frame_id,
                timestamp=frame_timestamp,
                fps=self.native_fps if self.is_video_file else fps,
                source_id=self.source_id,
            )

            # Store the original frame before the processing queue.
            # This preserves evidence frames even if the processing queue
            # drops frames or inference is slower than capture.
            if self.evidence_buffer is not None:
                try:
                    self.evidence_buffer.add(frame)
                except Exception:
                    # Buffer failures must not crash video capture.
                    pass

            if self.preserve_all_frames:
                # Preserve every frame for prerecorded-video analysis.
                # Stop-aware retries prevent indefinite waiting during stop.
                while self.running:
                    try:
                        self.frame_queue.put(frame, timeout=0.1)
                        break
                    except queue.Full:
                        continue
            else:
                # For live streams, prefer recent frames over stale frames.
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    pass

            # Pace finite video playback to its native FPS.
            # Do not sleep an additional time after slow queue processing.
            if self.is_video_file:
                elapsed = time.perf_counter() - loop_start
                target_delay = 1.0 / self.native_fps
                sleep_needed = target_delay - elapsed

                if sleep_needed > 0:
                    time.sleep(sleep_needed)

    def read(self) -> Optional[Frame]:
        """Return the next queued frame, or None when the queue is empty."""

        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def read_latest(self) -> Optional[Frame]:
        """Drain the queue and return its newest available frame."""

        latest = None

        while True:
            try:
                latest = self.frame_queue.get_nowait()
            except queue.Empty:
                break

        return latest

    @property
    def qsize(self) -> int:
        """Number of frames currently waiting in the processing queue."""

        return self.frame_queue.qsize()

    def stop(self):
        """Stop capture, wait for the thread, and release the source."""

        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=5.0)

            # Do not release the source while its capture thread is still
            # reading from it.
            if self.thread.is_alive():
                return

            self.thread = None

        self.source.release()