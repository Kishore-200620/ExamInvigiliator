"""
Violation API schemas - Phase 8
Pydantic models for violation/event API.
"""
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field


class ViolationResponse(BaseModel):
    event_id: str
    student_id: Optional[int] = None
    rule_id: Optional[str] = None
    event_type: Optional[str] = None
    status: Optional[str] = None
    severity: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None
    supporting_features: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None
    evidence_available: bool = False
    evidence_video_url: Optional[str] = None
    evidence_metadata_url: Optional[str] = None

    class Config:
        json_encoders = {}


class ViolationListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    violations: List[ViolationResponse]


class EvidenceResponse(BaseModel):
    event_id: str
    session_id: Optional[str] = None
    available: bool
    video_exists: bool
    metadata_exists: bool
    video_url: Optional[str] = None
    metadata_url: Optional[str] = None
    video_path: Optional[str] = None  # internal, not exposed to frontend ideally but useful
    metadata: Optional[Dict[str, Any]] = None  # optional embedded


class ViolationSummaryResponse(BaseModel):
    total_violations: int
    violations_by_rule: Dict[str, int] = Field(default_factory=dict)
    violations_by_severity: Dict[str, int] = Field(default_factory=dict)
    violations_by_student: Dict[str, int] = Field(default_factory=dict)
    violations_by_status: Dict[str, int] = Field(default_factory=dict)
    violations_by_session: Dict[str, int] = Field(default_factory=dict)


def malpractice_to_violation(event, session_id: Optional[str], evidence_info: Optional[dict] = None) -> ViolationResponse:
    """Convert MalpracticeEvent to ViolationResponse."""
    from app.services.evidence_manager import _get_event_interval
    # Extract times
    try:
        start_time, end_time = _get_event_interval(event)
    except:
        start_time = getattr(event, "timestamp", None)
        end_time = start_time
        try:
            start_time = float(start_time) if start_time else None
            end_time = float(end_time) if end_time else None
        except:
            start_time = end_time = None
    duration = getattr(event, "duration", None)
    try:
        duration = float(duration) if duration is not None else (end_time - start_time if start_time and end_time else None)
    except:
        duration = None

    # Status/severity
    status = getattr(event, "status", None)
    try:
        status = status.value if hasattr(status, "value") else str(status) if status else None
    except:
        status = str(status) if status else None
    severity = getattr(event, "severity", None)
    try:
        severity = severity.value if hasattr(severity, "value") else str(severity) if severity else None
    except:
        severity = str(severity) if severity else None

    reason = getattr(event, "reason", None)
    if reason is None and hasattr(event, "evidence") and isinstance(event.evidence, dict):
        reason = event.evidence.get("reason")

    supporting = getattr(event, "evidence", None)
    if supporting is None:
        supporting = getattr(event, "supporting_features", None)

    # Evidence URLs
    evidence_available = False
    video_url = None
    meta_url = None
    if evidence_info:
        evidence_available = evidence_info.get("available", False)
        if evidence_info.get("video_exists"):
            video_url = f"/api/v1/violations/{event.event_id}/evidence/video"
        if evidence_info.get("metadata_exists"):
            meta_url = f"/api/v1/violations/{event.event_id}/metadata"

    return ViolationResponse(
        event_id=str(getattr(event, "event_id")),
        student_id=getattr(event, "student_id", None),
        rule_id=getattr(event, "rule_id", None),
        event_type=getattr(event, "event_type", None),
        status=status,
        severity=severity,
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        confidence=getattr(event, "confidence", None),
        reason=reason,
        supporting_features=supporting,
        session_id=session_id,
        frame_start=getattr(event, "start_frame", None),
        frame_end=getattr(event, "end_frame", None),
        evidence_available=evidence_available,
        evidence_video_url=video_url,
        evidence_metadata_url=meta_url,
    )
