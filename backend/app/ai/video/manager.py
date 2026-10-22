from app.ai.video.base import VideoSource
from app.ai.video.engine import VideoEngine
from app.ai.video.frame import Frame
from app.ai.video.rolling_buffer import RollingOriginalBuffer
from app.config.settings import settings


class VideoManager:
    """
    High-level interface for the video subsystem.
    Phase 7: includes rolling original evidence buffer at ingestion.
    """

    def __init__(
        self,
        source: VideoSource,
        source_id: str = "camera_0",
        evidence_buffer: RollingOriginalBuffer | None = None,
        evidence_buffer_seconds: float | None = None,
        preserve_all_frames: bool = False,
        queue_size: int = 30,
    ):
        # Create rolling buffer if not provided - retains 3s default (settings.EVIDENCE_BUFFER_SECONDS)
        if evidence_buffer is None:
            retention = evidence_buffer_seconds if evidence_buffer_seconds is not None else settings.EVIDENCE_BUFFER_SECONDS
            evidence_buffer = RollingOriginalBuffer(retention_seconds=retention)
        self.evidence_buffer = evidence_buffer
        self.engine = VideoEngine(
            source=source,
            source_id=source_id,
            queue_size=queue_size,
            evidence_buffer=self.evidence_buffer,
            preserve_all_frames=preserve_all_frames,
        )

    def start(self) -> bool:
        return self.engine.start()

    def read(self) -> Frame | None:
        return self.engine.read()

    def read_latest(self) -> Frame | None:
        return self.engine.read_latest()

    def is_running(self) -> bool:
        return self.engine.running

    def qsize(self) -> int:
        return self.engine.qsize

    def stop(self) -> None:
        self.engine.stop()

    def get_evidence_buffer(self) -> RollingOriginalBuffer:
        return self.evidence_buffer
