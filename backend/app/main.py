from fastapi import FastAPI

from app.api.router import api_router
from app.config.settings import settings
from app.core.handlers import register_exception_handlers
from app.core.lifespan import lifespan
from app.middleware.setup import setup_middlewares

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Register global exception handlers
register_exception_handlers(app)

# Register middleware
setup_middlewares(app)

# Register API routes
app.include_router(api_router)