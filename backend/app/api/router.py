from fastapi import APIRouter

from app.api.v1.health import router as health_router
from app.api.v1.violations import router as violations_router
from app.api.v1.clips import router as clips_router
from app.api.v1.video import router as video_router
from app.api.v1 import live as live_router_module
from app.api.v1.live import router as live_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health_router)
api_router.include_router(violations_router)
api_router.include_router(clips_router)
api_router.include_router(video_router)
api_router.include_router(live_router)

# Session support: GET /api/v1/sessions/{session_id}/violations alias
from fastapi import APIRouter as _APIRouter

# Additional session route reuses same logic via query param
# We expose /api/v1/sessions/{session_id}/violations as convenience
from app.services.violation_service import violation_service
from app.schemas.violation import ViolationListResponse, malpractice_to_violation
from fastapi import Query
from typing import Optional

session_router = _APIRouter(prefix="/sessions", tags=["Sessions"])

@session_router.post("", summary="Create & start session alias")
@session_router.post("/start", summary="Start session alias")
async def create_start_session(req: live_router_module.LiveSessionStartRequest):
    return await live_router_module.start_live_exam(req)

@session_router.post("/{session_id}/start", summary="Start session with session_id alias")
async def start_session_id(session_id: str, video_id: str = Query("clip5")):
    req = live_router_module.LiveSessionStartRequest(video_id=video_id, session_name=session_id)
    return await live_router_module.start_live_exam(req)

@session_router.post("/{session_id}/stop", summary="Stop session alias")
@session_router.delete("/{session_id}", summary="Delete/stop session alias")
async def stop_session_alias(session_id: str):
    return await live_router_module.stop_live(session_id)

@session_router.get("/{session_id}/status", summary="Session status alias")
async def session_status_alias(session_id: str):
    return await live_router_module.live_status(session_id)

@session_router.get("/{session_id}/state", summary="Session state alias")
async def session_state_alias(session_id: str):
    return await live_router_module.get_live_state(session_id)

@session_router.get("/{session_id}/violations", response_model=ViolationListResponse, summary="List violations by session")
async def list_session_violations(
    session_id: str,
    limit: int = Query(50, ge=0, le=1000),
    offset: int = Query(0, ge=0),
):
    if "/" in session_id or "\\" in session_id or ".." in session_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid session_id")
    items = violation_service.list_events(session_id=session_id, limit=limit, offset=offset)
    total = violation_service.count_events(session_id=session_id)
    violations = []
    for item in items:
        ev = item["event"]
        sid = item["session_id"]
        info = violation_service.get_evidence_info(str(ev.event_id))
        violations.append(malpractice_to_violation(ev, sid, info))
    return ViolationListResponse(total=total, limit=limit, offset=offset, violations=violations)

api_router.include_router(session_router)