"""
Application entry point.

Wires together:
- FastAPI application instance
- Lifespan handler (model preload on startup)
- CORS middleware
- Rate limiter (slowapi)
- API router registration
- Global exception handlers
- Health check and root endpoints
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.routes.detection import router as detection_router
from app.api.routes.training import router as training_router
from app.core.config import get_settings
from app.core.model_loader import ModelLoader
from app.schemas.detection_schema import ErrorResponse, HealthResponse

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger("app.main")

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application startup and shutdown.

    On startup  : Load YOLO model into memory.
    On shutdown : Log graceful shutdown message.
    """
    logger.info("=" * 60)
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    logger.info("Log level  : %s", settings.log_level)
    logger.info("CUDA       : %s", torch.cuda.is_available())
    logger.info("Model path : %s", settings.model_path)
    logger.info("=" * 60)

    # Pre-load model so the first request doesn't pay the loading cost
    try:
        ModelLoader.get_instance(model_path=settings.model_path)
        logger.info("YOLO model ready ✔")
    except Exception as exc:
        logger.critical("Failed to load model on startup: %s", exc)
        # Allow the app to start — requests will return 503 until fixed
        # (useful in Kubernetes readiness-probe based scenarios)

    yield  # ← application is running

    logger.info("Shutting down %s …", settings.app_name)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Factory function — creates and configures the FastAPI application."""

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-ready Real-Time Object Detection API powered by "
            "YOLOv8 and FastAPI. Supports image uploads, video frame-by-frame "
            "analysis, and base64 image inference."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ---- Rate limiter state ----
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ---- CORS ----
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Request timing middleware ----
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
        return response

    # ---- Global exception handlers ----
    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        logger.warning(
            "HTTP %d at %s — %s", exc.status_code, request.url.path, exc.detail
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(detail=str(exc.detail)).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled exception at %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                detail="An internal server error occurred.",
                error_code="INTERNAL_ERROR",
            ).model_dump(),
        )

    # ---- Routers ----
    app.include_router(
        detection_router,
        prefix=settings.api_prefix,
    )
    app.include_router(
        training_router,
        prefix=settings.api_prefix,
    )

    # ---- Health check ----
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Health check",
        description="Returns model load status and runtime environment info.",
    )
    @limiter.limit("120/minute")
    async def health_check(request: Request) -> HealthResponse:
        loader = ModelLoader.get_instance(model_path=settings.model_path)
        return HealthResponse(
            status="healthy" if loader.is_loaded else "degraded",
            model_loaded=loader.is_loaded,
            model_path=settings.model_path,
            model_device=loader.device,
            cuda_available=loader.is_cuda_available,
            app_version=settings.app_version,
        )

    # ---- Root ----
    @app.get(
        "/",
        tags=["System"],
        summary="API root",
        include_in_schema=False,
    )
    async def root() -> dict[str, Any]:
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
        }

    return app


# ---------------------------------------------------------------------------
# Application instance (imported by uvicorn)
# ---------------------------------------------------------------------------

app = create_app()

# ---------------------------------------------------------------------------
# Development runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
    )
