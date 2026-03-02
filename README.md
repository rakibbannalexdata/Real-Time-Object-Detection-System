# Real-Time Object Detection & Segmentation System

A **production-ready** REST API for real-time object detection and segmentation model training powered by **YOLOv8** and **FastAPI**.

## ✨ Features

- 📸 **Image detection** — upload JPEG/PNG/BMP/WEBP and get structured JSON back
- 🎬 **Video detection** — frame-by-frame analysis with automatic resource cleanup
- 🔡 **Base64 support** — send encoded images via JSON for UI/mobile clients
- 🧠 **Segmentation Training** — trigger YOLOv8 segmentation training via API with strict dataset validation
- ⚡ **Singleton model** — YOLO loads once at startup, shared across all requests
- 🚀 **GPU auto-detection** — uses CUDA → MPS → CPU automatically
- 🛡️ **Rate limiting** — 60 requests/minute per IP via `slowapi`
- �️ **Automation** — full project lifecycle managed via `Makefile`
- �🐳 **Docker-ready** — multi-stage, non-root, health-checked image

---

## 📁 Project Structure

```
app/
├── main.py                  # FastAPI app, lifespan, middleware, routers
├── core/
│   ├── config.py            # Pydantic BaseSettings (env-configurable)
│   └── model_loader.py      # Thread-safe YOLO singleton
├── api/
│   └── routes/
│       ├── detection.py     # POST /detect/image, /video, /base64
│       └── training.py      # GET /training/dataset/info, POST /training/start
├── schemas/
│   ├── detection_schema.py  # Pydantic response models
│   └── training_schema.py   # Training request/response models
├── services/
│   ├── detection_service.py # Detection logic
│   └── training_service.py  # Dataset validation & training logic
└── utils/
    └── image_utils.py       # Decode, resize, base64 helpers
Makefile                     # Automation commands
requirements.txt
Dockerfile
```

---

## 🚀 Local Setup

The easiest way to set up the project is using the provided `Makefile`.

### 1. Installation

```bash
make install
```
This creates a virtual environment and installs all dependencies.

### 2. Configure (optional)

Copy the sample env file and edit as needed:
```bash
cp .env.example .env
```

### 3. Run the server

```bash
make run
```
API docs available at → **http://localhost:8000/docs**

---

## 🛠️ Automation (Makefile)

| Command | Description |
|---|---|
| `make install` | Setup venv and install dependencies |
| `make run` | Start the FastAPI development server |
| `make test-data` | Generate dummy test datasets for training verification |
| `make verify` | Run automated curl tests against training endpoints |
| `make clean` | Remove venv, __pycache__, and temporary training runs |

---

## � Segmentation Training

### 1. Dataset Info (Validation)
Check if your dataset is correctly formatted for YOLO segmentation.

```bash
curl -s "http://localhost:8000/api/v1/training/dataset/info?project=my_project" | jq .
```

### 2. Start Training
Trigger a background training task.

```bash
curl -X POST "http://localhost:8000/api/v1/training/start" \
     -H "Content-Type: application/json" \
     -d '{"project_name": "my_project", "epochs": 10, "imgsz": 640, "batch": 16}'
```

---

## 🧪 Detection API

### Image detection
```bash
curl -X POST http://localhost:8000/api/v1/detect/image -F "file=@/path/to/image.jpg"
```

### Video detection
```bash
curl -X POST http://localhost:8000/api/v1/detect/video -F "file=@/path/to/video.mp4"
```

---

## 🐳 Docker

### Build & Run
```bash
docker build -t object-detection:latest .
docker run -d -p 8000:8000 --name object-detection object-detection:latest
```

---

## 🔗 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health + model status |
| `POST` | `/api/v1/detect/image` | Detect objects in image |
| `POST` | `/api/v1/detect/video` | Detect objects in video |
| `GET` | `/api/v1/training/dataset/info` | Validate segmentation dataset |
| `POST` | `/api/v1/training/start` | Start segmentation training |

---

## 📜 License

MIT
