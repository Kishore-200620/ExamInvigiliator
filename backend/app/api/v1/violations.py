"""
Violations API - Phase 8
Exposes MalpracticeEvent + evidence via REST.
"""
from typing import Optional, Literal
from pathlib import Path

from fastapi import APIRouter, Query, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, JSONResponse, Response

from app.services.violation_service import violation_service
from app.schemas.violation import (
    ViolationResponse,
    ViolationListResponse,
    EvidenceResponse,
    ViolationSummaryResponse,
    malpractice_to_violation,
)
from app.core.logger import app_logger

router = APIRouter(prefix="/violations", tags=["Violations"])


class ReviewRequest(BaseModel):
    action: Literal["CONFIRMED", "REJECTED"]
    reviewer: str = Field(default="unspecified", max_length=120)
    notes: str = Field(default="", max_length=2000)


@router.get("", response_model=ViolationListResponse, summary="List violations")
async def list_violations(
    student_id: Optional[int] = Query(None, description="Filter by student_id"),
    rule_id: Optional[str] = Query(None, description="Filter by rule_id"),
    event_type: Optional[str] = Query(None, description="Filter by event_type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    session_id: Optional[str] = Query(None, description="Filter by session_id"),
    start_time: Optional[float] = Query(None, description="Filter events overlapping start_time"),
    end_time: Optional[float] = Query(None, description="Filter events overlapping end_time"),
    limit: int = Query(50, ge=0, le=1000, description="Pagination limit"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    try:
        items = violation_service.list_events(
            student_id=student_id,
            rule_id=rule_id,
            event_type=event_type,
            status=status,
            severity=severity,
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset,
        )
        total = violation_service.count_events(
            student_id=student_id,
            rule_id=rule_id,
            event_type=event_type,
            status=status,
            severity=severity,
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
        )
        violations = []
        for item in items:
            ev = item["event"]
            sid = item["session_id"]
            info = violation_service.get_evidence_info(str(ev.event_id))
            violations.append(malpractice_to_violation(ev, sid, info))
        return ViolationListResponse(total=total, limit=limit, offset=offset, violations=violations)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        app_logger.exception(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/summary", response_model=ViolationSummaryResponse, summary="Violation summary")
async def violation_summary():
    try:
        s = violation_service.get_summary()
        return ViolationSummaryResponse(**s)
    except Exception as e:
        app_logger.exception(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/{event_id}", response_model=ViolationResponse, summary="Get violation by ID")
async def get_violation(event_id: str):
    # Path traversal protection: violation_service.get_event already checks
    if "/" in event_id or "\\" in event_id or ".." in event_id:
        raise HTTPException(status_code=400, detail="Invalid event_id")
    item = violation_service.get_event(event_id)
    if not item:
        raise HTTPException(status_code=404, detail="Event not found")
    ev = item["event"]
    sid = item["session_id"]
    info = violation_service.get_evidence_info(event_id)
    return malpractice_to_violation(ev, sid, info)


@router.get("/{event_id}/review", summary="Get persisted review decision")
async def get_violation_review(event_id: str):
    review = violation_service.get_review(event_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    return {"status": "saved", "violation_id": event_id, "review_status": review["action"], "review": review}


@router.post("/{event_id}/review", summary="Persist a review decision")
async def submit_violation_review(event_id: str, body: ReviewRequest):
    review = violation_service.save_review(event_id, body.action, body.reviewer.strip() or "unspecified", body.notes)
    if not review:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"status": "saved", "violation_id": event_id, "review_status": body.action, "review": review}


@router.get("/{event_id}/evidence", response_model=EvidenceResponse, summary="Get evidence info")
async def get_evidence(event_id: str):
    if "/" in event_id or "\\" in event_id or ".." in event_id:
        raise HTTPException(status_code=400, detail="Invalid event_id")
    item = violation_service.get_event(event_id)
    if not item:
        raise HTTPException(status_code=404, detail="Event not found")
    info = violation_service.get_evidence_info(event_id)
    if not info:
        raise HTTPException(status_code=404, detail="Evidence not found")
    # Build URLs
    video_url = f"/api/v1/violations/{event_id}/evidence/video" if info["video_exists"] else None
    meta_url = f"/api/v1/violations/{event_id}/metadata" if info["metadata_exists"] else None
    return EvidenceResponse(
        event_id=event_id,
        session_id=info["session_id"],
        available=info["available"],
        video_exists=info["video_exists"],
        metadata_exists=info["metadata_exists"],
        video_url=video_url,
        metadata_url=meta_url,
        video_path=str(info["video_path"]) if info["video_path"] else None,
    )


@router.get("/{event_id}/metadata", summary="Get metadata.json")
async def get_metadata(event_id: str):
    if "/" in event_id or "\\" in event_id or ".." in event_id:
        raise HTTPException(status_code=400, detail="Invalid event_id")
    item = violation_service.get_event(event_id)
    if not item:
        raise HTTPException(status_code=404, detail="Event not found")
    info = violation_service.get_evidence_info(event_id)
    if not info or not info["metadata_exists"]:
        raise HTTPException(status_code=404, detail="Metadata not found")
    stored = violation_service.get_stored_evidence(event_id)
    if stored:
        return JSONResponse(content=stored["metadata"])
    try:
        import json
        with open(info["metadata_path"], "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Metadata not found")
    except Exception as e:
        app_logger.exception(e)
        raise HTTPException(status_code=500, detail="Failed to read metadata")


@router.get("/{event_id}/evidence/video", summary="Stream evidence video")
async def get_evidence_video(event_id: str, request: Request):
    if "/" in event_id or "\\" in event_id or ".." in event_id:
        raise HTTPException(status_code=400, detail="Invalid event_id")
    item = violation_service.get_event(event_id)
    if not item:
        raise HTTPException(status_code=404, detail="Event not found")
    info = violation_service.get_evidence_info(event_id)
    if not info or not info["video_exists"]:
        raise HTTPException(status_code=404, detail="Evidence video not found")
    stored = violation_service.get_stored_evidence(event_id)
    if stored and stored["video"] is not None:
        return Response(
            content=stored["video"],
            media_type="video/mp4",
            headers={"Content-Disposition": 'inline; filename="original_evidence.mp4"', "Cache-Control": "no-cache"},
        )
    if info["video_path"] is None:
        raise HTTPException(status_code=404, detail="Evidence video not found")
    video_path: Path = info["video_path"]  # type: ignore
    # Security: ensure path is within base_dir and is file
    try:
        
        base_resolved = violation_service.base_dir.resolve()
        target_resolved = Path(video_path).resolve()

        if not target_resolved.is_relative_to(base_resolved):
            raise HTTPException(status_code=404, detail="Evidence video not found")

        if not target_resolved.is_file():
            raise HTTPException(status_code=404, detail="Evidence video not found")

    except HTTPException:
        raise
    except Exception as e:
        app_logger.exception(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
    # Use FileResponse with correct media type
    return FileResponse(
        path=str(target_resolved),
        media_type="video/mp4",
        filename="original_evidence.mp4",
        headers={"Cache-Control": "no-cache"},
    )
