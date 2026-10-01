# 🚁 Drone Head Detection & Model Comparison Hub

A local web application built with **FastAPI**, **Ultralytics YOLO**, and **OpenCV** designed for testing, evaluating, and benchmarking custom head-detection object detection models on uploaded drone and aerial footage.

---

## 📌 Overview

When training custom object detection models for aerial and drone surveillance, finding the best-performing model on real-world footage requires testing multiple iterations on the exact same video under identical parameters.

This application provides a local evaluation workbench to:
1. **Upload drone videos** (`.mp4`, `.avi`, `.mov`, `.mkv`).
2. **Select trained models** using interactive option buttons or dropdowns.
3. **Run frame-by-frame inference** with bounding box annotations (`head <confidence>`) and an on-screen heads-up display (HUD).
4. **Stream and watch the processed video** directly inside your browser.
5. **Download the annotated video** with 1 click.
6. **Compare multiple models side-by-side** using synchronized dual video players and an objective benchmarking metrics table.

> 🔒 **Privacy & Weights Protection**: Trained `.pt` models, uploaded videos, and processed outputs are ignored by `.gitignore` by default. Your proprietary model weights remain private and are not pushed to GitHub.

---

## ✨ Features

- **🎯 Interactive Model Option Buttons**: All `.pt` weights inside the `models/` directory are automatically discovered and displayed as clickable option tiles with file size badges.
- **⚡ Hardware Acceleration (CUDA GPU + CPU Fallback)**: Automatically detects NVIDIA CUDA GPUs and falls back to CPU if unavailable, with manual device override options (`Auto`, `CUDA`, `CPU`).
- **🧠 Memory-Efficient Processing**: OpenCV streams frames one by one through the loaded YOLO model without loading entire videos into RAM. Model weights are loaded once per session.
- **🏷️ Smart Class Labeling**:
  - Single-class models are automatically treated and labeled as `head <confidence>`.
  - Multi-class models with `head` / `person` classes are accurately recognized and labeled.
- **📊 Real-time Heads-Up Display (HUD)**: Burned-in corner badge on the video displaying the active model name, real-time head count per frame, and total cumulative detections.
- **⏩ Processing Range Selector**: Choose between **Full Video** or a **Quick Test (First 60, 150, or 300 Frames)** to test model behavior in seconds on CPU before processing full length videos.
- **⚖️ Side-by-Side Dual Comparison**: Select **Model A** and **Model B** to evaluate the exact same video concurrently with synchronized dual video players (`▶ Play Both in Sync`, `⏸ Pause Both`, synchronized scrubbing).
- **📋 Persistent Comparison Table**: Tracks and compiles evaluation measurements across multiple test runs on the same video (Total Detections, Avg Heads/Frame, Max Heads/Frame, Avg Confidence, Inference FPS, Processing Time).
- **📥 Web-Streamable H.264 Playback & Download**: Encodes videos using H.264 with `+faststart` for smooth in-browser streaming and fast download.

---

## 📁 Project Structure

```text
head_detection_app/
│
├── app.py                      # FastAPI REST API backend & background workers
├── requirements.txt            # Python dependencies
├── .gitignore                  # Excludes models, uploads, outputs, and cache
├── README.md                   # Project documentation
│
├── models/                     # Place your custom YOLO .pt weights here (excluded from Git)
│   └── .gitkeep
│
├── uploads/                    # Stores uploaded source drone videos (excluded from Git)
│   └── .gitkeep
│
├── outputs/                    # Processed output videos with bounding boxes (excluded from Git)
│   └── .gitkeep
│
├── services/
│   ├── __init__.py
│   ├── model_manager.py        # Model discovery, device resolution, and caching
│   └── video_processor.py      # OpenCV video loop, YOLO inference, HUD, and H.264 encode
│
└── frontend/
    ├── index.html              # Clean single-page application UI
    ├── style.css               # Modern dark slate ML theme and responsive layouts
    └── script.js               # Dynamic option buttons, progress polling, dual video sync
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher installed.
- (Optional) NVIDIA GPU with CUDA drivers installed for GPU acceleration.

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 3. Create and Activate a Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📦 Adding Your Trained Models

Place your custom Ultralytics YOLO head-detection model weights (`.pt` files) into the `models/` directory:

```text
models/
    best_model_1.pt
    best_model_2.pt
    drone_head_yolov8.pt
```

Alternatively, launch the web application and click the **`📁 + Add .pt Model`** button in the interface to import weights directly from your computer.

Click **`↻ Refresh`** at any time to rescan the folder without restarting the application.

---

## 🏃 Running the Application

Start the local server with Uvicorn:

```bash
uvicorn app:app --reload
```

Open your browser and navigate to:
```text
http://127.0.0.1:8000
```

---

## 📖 How to Use

### Single Model Evaluation
1. **Upload Drone Video**: Drag and drop your aerial footage (`.mp4`, `.avi`, `.mov`, `.mkv`). The video resolution, FPS, duration, and frame count will appear.
2. **Select Model**: Click the option button for the model you want to test.
3. **Configure Parameters**:
   - **Confidence Threshold**: Filter weak predictions (default: `0.25`).
   - **IoU Threshold**: Non-Maximum Suppression threshold (default: `0.45`).
   - **Processing Range**: Choose `Full Video` or `Quick Test: First 60 / 150 / 300 Frames`.
4. **Run Inference**: Click **`🚀 Run Head Detection`**.
5. **View Predictions**: The output video automatically autoplays upon completion, showing bounding boxes around detected heads and the HUD status badge.
6. **Download**: Click **`📥 Download Processed Video (.mp4)`** to save the video to disk.

### Side-by-Side Model Comparison
1. Switch to the **`⚖️ Side-by-Side Comparison`** tab.
2. Choose **Model A** (Left) and **Model B** (Right).
3. Click **`⚖️ Compare Models Side-by-Side`**.
4. Both models will process the video, presenting dual video players.
5. Use **`▶ Play Both in Sync`** or scrub the timeline to inspect how both models detect heads on the exact same frame.

---

## 🛠️ API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend application |
| `GET` | `/api/system` | Returns CUDA / GPU acceleration status |
| `GET` | `/api/models` | Lists all `.pt` files inside `models/` |
| `POST` | `/api/models/upload` | Uploads a `.pt` model weight file directly into `models/` |
| `POST` | `/api/upload` | Uploads and inspects a drone video file |
| `POST` | `/api/detect` | Starts an asynchronous single model evaluation job |
| `POST` | `/api/compare` | Starts an asynchronous side-by-side comparison job |
| `GET` | `/api/progress/{job_id}` | Returns live progress percentage, FPS, frame counts, and stats |
| `POST` | `/api/cancel/{job_id}` | Aborts a running processing job |
| `GET` | `/api/result/{filename}` | Streams processed video (supports HTTP 206 Partial Content) |
| `GET` | `/api/history` | Returns model benchmark records for the comparison table |
| `DELETE` | `/api/history` | Clears benchmark history |

---

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.
