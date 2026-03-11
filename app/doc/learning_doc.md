# 🎓 ML Learning Doc: Real-Time Object Detection System

Welcome! This guide is designed for **Node.js/Backend Developers** who are new to Machine Learning (ML). It explains how this system works, the key concepts used, and how the codebase is structured.

---

## 🏗️ Architecture Overview

Think of this system as a standard REST API, but instead of just querying a database, it queries a **Neural Network (Model)**.

| Concept | Node.js Equivalent | This Project's Implementation |
| :--- | :--- | :--- |
| **Web Framework** | Express / NestJS | **FastAPI** (Python's ultra-fast modern framework) |
| **Data Validation** | Joi / Zod | **Pydantic** (Models located in `app/schemas/`) |
| **Execution Engine** | Node Runtime | **PyTorch** + **Ultralytics YOLOv8** |
| **Imaging Library** | Sharp / Canvas | **OpenCV** (Industrial-grade Computer Vision) |

---

## 🛠️ The ML Tech Stack

1.  **Ultralytics YOLOv8**: This is the "Brain". "YOLO" stands for *You Only Look Once*. It’s a specialized model that can detect and segment objects in an image instantly.
2.  **OpenCV (`cv2`)**: Used for handling image and video streams, resizing frames, and drawing boxes.
3.  **FastAPI**: Provides the HTTP layer, handling uploads, background tasks, and documentation (found at `/docs`).
4.  **Aiofiles**: Handles asynchronous file I/O (like `fs.promises` in Node).

---

## 🧠 Core ML Concepts

### 1. Detection vs. Segmentation
*   **Object Detection**: Drawing a matching square (**Bounding Box**) around an object.
*   **Segmentation**: Drawing a precise outline (**Polygon**) around the object. This project supports both!

### 2. Inference (The "Read" Operation)
In ML, "Inference" means using a pre-trained model to make predictions on new data.
*   **Input**: An image or video frame.
*   **Output**: A list of objects found, including their `class` (what it is), `bbox` (where it is), and `confidence` (how sure it is).

### 3. Confidence & IoU
*   **Confidence Score (0.0 to 1.0)**: Like a probability. If it's 0.95, the model is 95% sure it found the object.
*   **IoU (Intersection over Union)**: Used to filter out duplicate boxes over the same object.

### 4. Training (The "Write" Operation)
"Training" is the process of teaching the model to recognize new things.
*   **Dataset**: A collection of images + label files.
*   **Epochs**: How many times the model looks at the entire dataset. Too few = "Underfitting" (doesn't learn enough); Too many = "Overfitting" (memorizes but can't generalize).
*   **Batch Size**: How many images the model looks at at once.

---

## 📂 Project Structure & Key Files

*   **`app/api/routes/`**: Your controllers.
    *   `detection.py`: Handles `/detect/image`, `/detect/video`.
    *   `training.py`: Handles `/training/start`.
*   **`app/services/`**: The core logic.
    *   `detection_service.py`: Wraps the YOLO model and runs `model.predict()`.
    *   `training_service.py`: Handles dataset validation and the `model.train()` loop.
*   **`convert_coco_txt.py`**: A utility to convert raw data from the COCO format (standard in ML) into the YOLO format required by this project.
*   **`datasets/`**: This is where you store your training data.
*   **`models/`**: This is where your custom-trained `.pt` (PyTorch) model files are stored.

---

## 🚀 How to Use It

### 1. Setting Up Data
Before training, you need to prepare your data:
```bash
python convert_coco_txt.py --src main_data_sets/my_dataset --project my_project
```
This prepares the images and labels in `datasets/my_project`.

### 2. Starting a Local Train
You can train a model without the API using the CLI tool:
```bash
python train_local.py --project my_project --epochs 10 --imgsz 640
```

### 3. Running the API
Start the server:
```bash
python -m app.main
```
Then visit `http://localhost:8000/docs` to see the interactive Swagger UI and test the endpoints.

---

## 💡 Quick Tips for Node.js Devs
*   **Virtual Environments (`venv`)**: Similar to `node_modules`. Always ensure your environment is activated.
*   **GIL (Global Interpreter Lock)**: Python handles concurrency differently. We use `BackgroundTasks` in FastAPI to run heavy training without blocking the API.
*   **Async/Await**: Works almost exactly like JavaScript. You'll see `async def` and `await` everywhere in the API layer.

---
*Created for the Object Detection System Project.*
