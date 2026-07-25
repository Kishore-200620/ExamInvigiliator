"""
utils/logger.py

Centralized logger for ExamGuardian AI.
"""

from loguru import logger

from config.settings import LOG_FILE

logger.remove()

logger.add(
    LOG_FILE,
    rotation="10 MB",
    retention="10 days",
    level="INFO",
    enqueue=True,
)

logger.add(
    sink=lambda message: print(message, end=""),
    level="INFO",
)

__all__ = ["logger"]