import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import SQLAlchemyError

from app import models  # noqa: F401 - registers all ORM mappings with Base
from app.api.health import router as health_router
from app.api.analysis import router as analysis_router
from app.api.correlation import router as correlation_router
from app.api.threat_intel import router as threat_intel_router
from app.api.ml import router as ml_router
from app.api.novelty import router as novelty_router
from app.api.exposure import router as exposure_router
from app.api.risk import router as risk_router
from app.api.alerts import router as alerts_router
from app.api.sessions import router as sessions_router
from app.config import get_settings
from app.db import Base, engine
from app.logging_config import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Bootstrap convenience for local development. Production schema changes belong in migrations.
    if settings.environment.lower() == "development":
        Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
_static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=_static_dir), name="static")
_extension_dir = Path(__file__).resolve().parent.parent / "browser_extension"
if _extension_dir.is_dir():
    app.mount("/browser-extension", StaticFiles(directory=_extension_dir), name="browser-extension")


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(_static_dir / "index.html")


@app.middleware("http")
async def local_ui_security_headers(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith(("/static/", "/browser-extension/")):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
            "media-src 'self' blob:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=(), geolocation=()"
    return response


if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )


@app.exception_handler(SQLAlchemyError)
async def database_error_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    logger.exception("Database operation failed for %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "A database operation failed"})


app.include_router(health_router, prefix=settings.api_v1_prefix)
app.include_router(analysis_router, prefix=settings.api_v1_prefix)
app.include_router(correlation_router, prefix=settings.api_v1_prefix)
app.include_router(threat_intel_router, prefix=settings.api_v1_prefix)
app.include_router(ml_router, prefix=settings.api_v1_prefix)
app.include_router(novelty_router, prefix=settings.api_v1_prefix)
app.include_router(exposure_router, prefix=settings.api_v1_prefix)
app.include_router(risk_router, prefix=settings.api_v1_prefix)
app.include_router(alerts_router, prefix=settings.api_v1_prefix)
app.include_router(sessions_router, prefix=settings.api_v1_prefix)
