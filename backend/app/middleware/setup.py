from fastapi import FastAPI

from app.middleware.cors import setup_cors
from app.middleware.logging import LoggingMiddleware


def setup_middlewares(app: FastAPI) -> None:
    setup_cors(app)

    app.add_middleware(LoggingMiddleware)