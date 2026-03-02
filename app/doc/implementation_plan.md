# Real-Time Object Detection System — Implementation Plan

Build a production-ready REST API that runs YOLOv8 object detection on uploaded images and videos. The system follows clean architecture with singleton model loading, dependency injection, structured JSON responses, and Docker support.

## Proposed Changes

### Core Infrastructure

#### [NEW] [config.py](file:///home/rakib-ul-banna/projects/test-detection-system/app/core/config.py)
- Pydantic `BaseSettings` for all env-configurable values: `MODEL_PATH`, `CONFIDENCE_THRESHOLD`, `MAX_IMAGE_SIZE`, `API_RATE_LIMIT`, `LOG_LEVEL`
- Reads from `.env` file automatically

#### [NEW] [model_loader.py](file:///home/rakib-ul-banna/projects/test-detection-system/app/core/model_loader.py)
- `ModelLoader` singleton class
- Loads `yolov8n.pt` once at startup
- Detects CUDA availability and uses GPU if present
- Raises descriptive `RuntimeError` if model file is missing

---

### Schemas

#### [NEW] [detection_schema.py](file:///home/rakib-ul-banna/projects/test-detection-system/app/schemas/detection_schema.py)
- `BoundingBox` — `[x1, y1, x2, y2]` as list of floats
- `Detection` — `class_name`, `confidence`, `bbox`
- `ImageDetectionResponse` — `detections: list[Detection]`, `image_width`, `image_height`, `inference_time_ms`
- `VideoDetectionResponse` — `total_frames`, `processed_frames`, `frame_detections`, `processing_time_ms`
- `HealthResponse` — model status, CUDA info, version
- `ErrorResponse` — error detail

---

### Services

#### [NEW] [detection_service.py](file:///home/rakib-ul-banna/projects/test-detection-system/app/services/detection_service.py)
- `DetectionService` class injected with `ModelLoader`
- `detect_image(image: np.ndarray, confidence_threshold: float) → list[Detection]`
- `detect_video(video_path: str, confidence_threshold: float) → VideoDetectionResponse`
  - Frame-by-frame processing with `cv2.VideoCapture`
  - Properly releases resources via `finally`
  - Skips frames optionally to save memory

---

### Utils

#### [NEW] [image_utils.py](file:///home/rakib-ul-banna/projects/test-detection-system/app/utils/image_utils.py)
- `decode_image(data: bytes) → np.ndarray` — decode uploaded bytes to OpenCV array, raise on corruption
- `resize_if_large(img, max_size) → np.ndarray` — resize if largest dimension > max_size
- `base64_to_image(b64_str: str) → np.ndarray` — decode base64 string to image

---

### API Routes

#### [NEW] [detection.py](file:///home/rakib-ul-banna/projects/test-detection-system/app/api/routes/detection.py)
- `POST /detect/image` — accepts `UploadFile`, validates MIME type, runs image detection, returns `ImageDetectionResponse`
- `POST /detect/video` — accepts `UploadFile`, saves to temp file, runs video detection, cleans up temp file
- `POST /detect/base64` — accepts JSON body with `base64_image`, runs detection (bonus)
- Query param `confidence_threshold` on all detection endpoints (default from config)

---

### Main App

#### [NEW] [main.py](file:///home/rakib-ul-banna/projects/test-detection-system/app/main.py)
- FastAPI app with `lifespan` context manager for startup/shutdown
- Mounts routes under `/api/v1`
- `GET /health` endpoint
- `GET /` root welcome endpoint
- CORS middleware
- Structured JSON logging via `logging`
- Basic rate limiting using `slowapi`
- Custom exception handlers for `HTTPException` and unhandled errors

---

### Package Init

#### [NEW] [__init__.py](file:///home/rakib-ul-banna/projects/test-detection-system/app/__init__.py)
- Empty package marker

---

### Docker & Requirements

#### [NEW] [requirements.txt](file:///home/rakib-ul-banna/projects/test-detection-system/requirements.txt)
```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
ultralytics>=8.2.0
opencv-python-headless>=4.9.0
numpy>=1.26.0
python-multipart>=0.0.9
pydantic>=2.7.0
pydantic-settings>=2.2.0
slowapi>=0.1.9
python-dotenv>=1.0.0
aiofiles>=23.2.1
```

#### [NEW] [Dockerfile](file:///home/rakib-ul-banna/projects/test-detection-system/Dockerfile)
- Base: `python:3.11-slim`
- System deps: `libgl1`, `libglib2.0-0` (OpenCV headless requirements)
- `COPY requirements.txt` → `pip install --no-cache-dir`
- Non-root user for security
- `EXPOSE 8000`
- `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]`

#### [NEW] [.dockerignore](file:///home/rakib-ul-banna/projects/test-detection-system/.dockerignore)
- Excludes `__pycache__`, `*.pt` model files (downloaded at runtime), `.env`, `venv`

#### [NEW] [README.md](file:///home/rakib-ul-banna/projects/test-detection-system/README.md)
- Local setup, curl examples, Docker instructions, example responses

---

## Verification Plan

### Automated (curl-based API tests)

After running locally (`uvicorn app.main:app --reload`):

```bash
# 1. Health check
curl http://localhost:8000/health

# 2. Image detection
curl -X POST http://localhost:8000/api/v1/detect/image \
  -F "file=@/path/to/image.jpg"

# 3. Image detection with custom threshold
curl -X POST "http://localhost:8000/api/v1/detect/image?confidence_threshold=0.5" \
  -F "file=@/path/to/image.jpg"

# 4. Video detection
curl -X POST http://localhost:8000/api/v1/detect/video \
  -F "file=@/path/to/video.mp4"

# 5. Base64 image detection
curl -X POST http://localhost:8000/api/v1/detect/base64 \
  -H "Content-Type: application/json" \
  -d '{"base64_image": "<base64_string>"}'
```

### Docker Verification

```bash
docker build -t object-detection .
docker run -p 8000:8000 object-detection
# Then run same curl commands against localhost:8000
```

### Error Handling Checks
- Upload a `.txt` file → expect 400 "Unsupported file type"
- Upload a corrupted image bytes → expect 422 "Failed to decode image"
