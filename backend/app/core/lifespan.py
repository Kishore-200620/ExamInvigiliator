from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.logger import app_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_logger.info("Starting ExamInvigilatorAI...")

    # Future startup tasks:
    # - Load AI models
    # - Connect to database
    # - Initialize caches
    # - Start background workers

    yield

    app_logger.info("Shutting down ExamInvigilatorAI...")

    # Future shutdown tasks:
    # - Release AI models
    # - Close database connections
    # - Stop background workers