# Real-Time Object Detection System

A **production-ready** REST API for real-time object detection powered by **YOLOv8** and **FastAPI**.

## ✨ Features

- 📸 **Image detection** — upload JPEG/PNG/BMP/WEBP and get structured JSON back
- 🎬 **Video detection** — frame-by-frame analysis with automatic resource cleanup
- 🔡 **Base64 support** — send encoded images via JSON for UI/mobile clients
- ⚡ **Singleton model** — YOLO loads once at startup, shared across all requests
- 🚀 **GPU auto-detection** — uses CUDA → MPS → CPU automatically
- 🛡️ **Rate limiting** — 60 requests/minute per IP via `slowapi`
- 🐳 **Docker-ready** — multi-stage, non-root, health-checked image

---

## 📁 Project Structure

```
app/
├── main.py                  # FastAPI app, lifespan, middleware, exception handlers
├── core/
│   ├── config.py            # Pydantic BaseSettings (env-configurable)
│   └── model_loader.py      # Thread-safe YOLO singleton
├── api/
│   └── routes/
│       └── detection.py     # POST /detect/image, /video, /base64
├── schemas/
│   └── detection_schema.py  # Pydantic response models
├── services/
│   └── detection_service.py # Business logic (image + video inference)
└── utils/
    └── image_utils.py       # Decode, resize, base64 helpers
requirements.txt
Dockerfile
.dockerignore
```

---

## 🚀 Local Setup

### 1. Clone & set up a virtual environment

```bash
git clone <repo-url>
cd test-detection-system
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure (optional)

Copy the sample env file and edit as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `yolov8n.pt` | YOLO model name / path |
| `CONFIDENCE_THRESHOLD` | `0.25` | Default detection threshold |
| `LOG_LEVEL` | `INFO` | `DEBUG/INFO/WARNING/ERROR` |
| `MAX_IMAGE_SIZE` | `1280` | Max image dimension (px) |
| `VIDEO_FRAME_SKIP` | `1` | Process every N-th frame |
| `MAX_VIDEO_FRAMES` | `500` | Hard cap on frames per video |
| `RATE_LIMIT_REQUESTS` | `60` | Requests per minute per IP |

### 4. Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at → **http://localhost:8000/docs**

---

## 🧪 Testing with curl

### Health check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "yolov8n.pt",
  "model_device": "cuda",
  "cuda_available": true,
  "app_version": "1.0.0"
}
```

---

### Image detection

```bash
curl -X POST http://localhost:8000/api/v1/detect/image \
  -F "file=@/path/to/image.jpg"
```

With custom confidence threshold:
```bash
curl -X POST "http://localhost:8000/api/v1/detect/image?confidence_threshold=0.5" \
  -F "file=@/path/to/image.jpg"
```

**Response:**
```json
{
  "status": "success",
  "detections": [
    {
      "class": "person",
      "confidence": 0.9241,
      "bbox": [125.4, 80.2, 380.1, 620.5],
      "class_id": 0
    },
    {
      "class": "car",
      "confidence": 0.8712,
      "bbox": [400.0, 200.0, 750.3, 480.9],
      "class_id": 2
    }
  ],
  "total_detections": 2,
  "image_width": 1280,
  "image_height": 720,
  "confidence_threshold": 0.25,
  "inference_time_ms": 34.21,
  "model_device": "cuda"
}
```

---

### Video detection

```bash
curl -X POST http://localhost:8000/api/v1/detect/video \
  -F "file=@/path/to/video.mp4"
```

**Response:**
```json
{
  "status": "success",
  "total_frames": 300,
  "processed_frames": 300,
  "frame_detections": [
    {
      "frame_index": 0,
      "timestamp_ms": 0.0,
      "detections": [{"class": "car", "confidence": 0.88, "bbox": [...], "class_id": 2}],
      "total_detections": 1
    }
  ],
  "confidence_threshold": 0.25,
  "processing_time_ms": 1420.5,
  "video_fps": 30.0,
  "video_width": 1920,
  "video_height": 1080
}
```

---

### Base64 image detection

```bash
BASE64=$(base64 -w 0 /path/to/image.jpg)

curl -X POST http://localhost:8000/api/v1/detect/base64 \
  -H "Content-Type: application/json" \
  -d "{\"base64_image\": \"$BASE64\", \"confidence_threshold\": 0.3}"
```

---

## 🐳 Docker

### Build

```bash
docker build -t object-detection:latest .
```

### Run

```bash
docker run -d \
  -p 8000:8000 \
  --name object-detection \
  -e CONFIDENCE_THRESHOLD=0.3 \
  -e LOG_LEVEL=INFO \
  object-detection:latest
```

### With GPU support

```bash
docker run -d \
  -p 8000:8000 \
  --gpus all \
  --name object-detection-gpu \
  object-detection:latest
```

### Docker Compose (quick start)

```yaml
# docker-compose.yml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=yolov8n.pt
      - CONFIDENCE_THRESHOLD=0.25
      - LOG_LEVEL=INFO
    restart: unless-stopped
```

```bash
docker compose up -d
```

---

## 🔗 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API root |
| `GET` | `/health` | Health + model status |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc UI |
| `POST` | `/api/v1/detect/image` | Detect objects in image |
| `POST` | `/api/v1/detect/video` | Detect objects in video |
| `POST` | `/api/v1/detect/base64` | Detect from base64 image |

---

## ⚡ Performance Notes

- The YOLO model loads **once** at startup via a thread-safe singleton
- Large images are **automatically resized** (configurable via `MAX_IMAGE_SIZE`)
- Video frames are skippable via `VIDEO_FRAME_SKIP` (e.g., `2` = process every other frame)
- GPU is auto-selected when CUDA is available — no code changes required
- Responses include `inference_time_ms` for latency monitoring

---

## 🛠️ Development

```bash
# Run with auto-reload
uvicorn app.main:app --reload --log-level debug

# Run with multiple workers (production, no reload)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

---

## 📜 License

MIT
