from fastapi import APIRouter

from app.schemas.response import APIResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=APIResponse,
    summary="Health Check",
    description="Returns the current health status of the ExamInvigilatorAI backend.",
)
async def health():
    return APIResponse(
        message="Health check successful",
        data={
            "status": "healthy",
            "service": "ExamInvigilatorAI Backend",
            "version": "1.0.0",
        },
    )