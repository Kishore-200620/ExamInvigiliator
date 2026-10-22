"""
Video Samples API - Phase 9
Provides access to sample exam videos.
"""
import pathlib
import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter(prefix="/video", tags=["Video"])
MAX_VIDEO_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024  # 5 GiB

# Resolve sample_clips relative to project root (backend/app/api/v1 -> project root)
try:
    _proj_root = pathlib.Path(__file__).resolve().parents[4]
    _cand = _proj_root / "sample_clips"
    SAMPLE_DIR = _cand if _cand.exists() else pathlib.Path("sample_clips")
except:
    SAMPLE_DIR = pathlib.Path("sample_clips")


class SampleVideo(BaseModel):
    id: str
    name: str
    path: str
    size_mb: float | None = None
    exists: bool


@router.get("/samples", response_model=list[SampleVideo], summary="List available sample videos")
async def list_samples():
    if not SAMPLE_DIR.exists():
        return []
    clips = []
    for p in sorted(SAMPLE_DIR.glob("*.mp4")):
        try:
            size_mb = round(p.stat().st_size / (1024*1024), 2)
        except:
            size_mb = None
        clips.append(SampleVideo(name=p.name, id=p.stem, path=str(p.as_posix()), size_mb=size_mb, exists=True))
    return clips


@router.get("/samples/{video_id}", summary="Stream sample video")
async def stream_sample(video_id: str):
    if "/" in video_id or "\\" in video_id or ".." in video_id:
        raise HTTPException(status_code=400, detail="Invalid video_id")
    clip_path = SAMPLE_DIR / f"{video_id}.mp4"
    if not clip_path.exists() or not clip_path.is_file():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")
    # Security check
    try:
        base = (pathlib.Path(__file__).resolve().parents[4] / "sample_clips").resolve()
        target = clip_path.resolve()
        if not str(target).startswith(str(base)):
            raise HTTPException(status_code=404, detail="Video not found")
    except:
        pass
    return FileResponse(path=str(clip_path), media_type="video/mp4", filename=clip_path.name)


@router.post("/upload", response_model=SampleVideo, summary="Upload an MP4 for live exam analysis")
async def upload_video(file: UploadFile = File(...)):
    """Stream an MP4 to disk so the live pipeline can process large exam videos."""
    filename = file.filename or ""
    if pathlib.Path(filename).suffix.lower() != ".mp4":
        raise HTTPException(status_code=415, detail="Only MP4 video uploads are supported")
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    video_id = f"upload_{uuid.uuid4().hex}"
    clip_path = SAMPLE_DIR / f"{video_id}.mp4"
    total = 0
    try:
        with clip_path.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_VIDEO_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video exceeds the 5 GiB upload limit")
                target.write(chunk)
    except Exception:
        clip_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return SampleVideo(id=video_id, name=filename, path=clip_path.as_posix(), size_mb=round(total / (1024 * 1024), 2), exists=True)
