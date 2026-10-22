import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logger import app_logger


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()

        response = await call_next(request)

        process_time = (time.perf_counter() - start_time) * 1000

        app_logger.info(
            f"{request.client.host} | "
            f"{request.method} | "
            f"{request.url.path} | "
            f"{response.status_code} | "
            f"{process_time:.2f} ms"
        )

        return response