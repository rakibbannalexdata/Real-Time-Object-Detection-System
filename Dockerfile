# ──────────────────────────────────────────────────────────────────────────
# Production-ready Dockerfile for the YOLOv8 Object Detection System
# ──────────────────────────────────────────────────────────────────────────
# Stages:
#   1. builder  — install Python dependencies into an isolated venv
#   2. runtime  — copy only the venv + app code into a lean final image
# ──────────────────────────────────────────────────────────────────────────

# ---- Stage 1: dependency builder ----------------------------------------
FROM python:3.11-slim AS builder

# Install build tools needed by some Python wheels (e.g. Pillow, OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# Create a venv to keep deps isolated from system Python
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: lean runtime image ----------------------------------------
FROM python:3.11-slim AS runtime

# Runtime system libraries required by OpenCV headless + torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy application source code
COPY app/ ./app/

# Change ownership
RUN chown -R appuser:appuser /app

USER appuser

# ── Environment defaults (override at runtime with -e or --env-file) ──────
ENV APP_NAME="Real-Time Object Detection System" \
    APP_VERSION="1.0.0" \
    MODEL_PATH="yolov8n.pt" \
    CONFIDENCE_THRESHOLD="0.25" \
    LOG_LEVEL="INFO" \
    MAX_IMAGE_SIZE="1280" \
    VIDEO_FRAME_SKIP="1" \
    MAX_VIDEO_FRAMES="500"

EXPOSE 8000

# Healthcheck — verify the API responds within 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Production server — 2 workers, preload for fast model sharing across workers
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--access-log"]
