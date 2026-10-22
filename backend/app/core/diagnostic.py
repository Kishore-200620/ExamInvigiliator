"""
Diagnostic logger for tracking pipeline - temporary debug mode.
When DIAGNOSTIC_MODE=False (default), all calls are no-ops with near-zero overhead.
When enabled and frame in [START, END], writes detailed trace to tracking_diagnostic.log
"""
from pathlib import Path
from typing import Optional

from app.config.settings import settings


def is_diagnostic_frame(frame_id: int) -> bool:
    """Check if frame should be logged - fast path when disabled."""
    if not settings.DIAGNOSTIC_MODE:
        return False
    return settings.DIAGNOSTIC_START_FRAME <= frame_id <= settings.DIAGNOSTIC_END_FRAME


def _log_path() -> Path:
    # Resolve relative to project root (backend/..)
    # settings.DIAGNOSTIC_LOG_PATH is "backend/logs/tracking_diagnostic.log"
    # When run from backend/, it's "logs/tracking_diagnostic.log" - handle both
    p = Path(settings.DIAGNOSTIC_LOG_PATH)
    if not p.is_absolute():
        # Try from project root first
        # __file__ is backend/app/core/diagnostic.py -> parents[3] is project root
        try:
            project_root = Path(__file__).resolve().parents[3]
            candidate = project_root / p
            # If candidate's parent exists, use it
            candidate.parent.mkdir(parents=True, exist_ok=True)
            return candidate
        except Exception:
            pass
        # Fallback: relative to cwd
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def diagnostic_log(frame_id: int, text: str):
    """Append text to diagnostic log if enabled for this frame."""
    if not is_diagnostic_frame(frame_id):
        return
    try:
        path = _log_path()
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass  # Never crash production on log failure


def diagnostic_log_block(frame_id: int, lines: list[str]):
    """Log multiple lines as a block."""
    if not is_diagnostic_frame(frame_id):
        return
    diagnostic_log(frame_id, "\n".join(lines))


def clear_diagnostic_log():
    """Clear log file (call at start of diagnostic run)."""
    if not settings.DIAGNOSTIC_MODE:
        return
    try:
        path = _log_path()
        if path.exists():
            path.write_text("", encoding="utf-8")
    except Exception:
        pass
