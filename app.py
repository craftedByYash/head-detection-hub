import os
import uuid
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from services.model_manager import ModelManager
from services.video_processor import VideoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("head_detection_app")

# Base directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
FRONTEND_DIR = BASE_DIR / "frontend"

for folder in [MODELS_DIR, UPLOADS_DIR, OUTPUTS_DIR, FRONTEND_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Services
model_manager = ModelManager(models_dir=MODELS_DIR)
video_processor = VideoProcessor(outputs_dir=OUTPUTS_DIR)

# App setup
app = FastAPI(
    title="Drone Head Detection & Model Comparison Hub",
    description="Test and compare custom head-detection YOLO models on uploaded drone videos.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory jobs tracking
jobs_lock = threading.Lock()
jobs: Dict[str, Dict[str, Any]] = {}

# Evaluation history for comparison table
history_lock = threading.Lock()
evaluation_history = []

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@app.get("/api/system")
async def get_system_info():
    """Returns CUDA/GPU and compute device information."""
    return model_manager.get_system_device_info()


@app.get("/api/models")
async def list_models():
    """Returns all available .pt models found in the models/ folder."""
    models = model_manager.list_models()
    system_info = model_manager.get_system_device_info()
    return {
        "models": models,
        "count": len(models),
        "system": system_info,
    }


@app.post("/api/models/upload")
async def upload_model_file(model_file: UploadFile = File(...)):
    """Uploads a .pt model directly into the models/ folder."""
    filename = Path(model_file.filename).name
    if not filename.lower().endswith(".pt"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model file '{filename}'. Only .pt model files are supported.",
        )

    target_path = MODELS_DIR / filename
    try:
        with open(target_path, "wb") as buffer:
            while chunk := await model_file.read(1024 * 1024):
                buffer.write(chunk)
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save model: {str(e)}")

    models = model_manager.list_models()
    return {
        "message": f"Model '{filename}' uploaded successfully to models/ folder.",
        "filename": filename,
        "models": models,
    }



@app.post("/api/upload")
async def upload_video(video: UploadFile = File(...)):
    """Uploads a drone video to the uploads/ directory."""
    original_filename = Path(video.filename).name
    ext = Path(original_filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Sanitize and create safe storage filename
    timestamp = int(time.time())
    safe_stem = "".join(c for c in Path(original_filename).stem if c.isalnum() or c in ("-", "_")).strip()
    if not safe_stem:
        safe_stem = "drone_video"
    target_filename = f"{safe_stem}_{timestamp}{ext}"
    target_path = UPLOADS_DIR / target_filename

    try:
        with open(target_path, "wb") as buffer:
            while chunk := await video.read(1024 * 1024):  # 1MB chunks
                buffer.write(chunk)
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded video: {str(e)}")

    # Verify and inspect video properties
    try:
        info = video_processor.inspect_video(target_path)
    except Exception as e:
        if target_path.exists():
            target_path.unlink()
        logger.error(f"Invalid video file uploaded: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"The uploaded file is not a valid video or cannot be decoded: {str(e)}",
        )

    return {
        "filename": target_filename,
        "original_name": original_filename,
        "size_mb": round(target_path.stat().st_size / (1024 * 1024), 2),
        "resolution": f"{info['width']} × {info['height']}",
        "width": info["width"],
        "height": info["height"],
        "fps": round(info["fps"], 2),
        "total_frames": info["total_frames"],
        "duration": info["duration_formatted"],
        "url": f"/api/uploads/{target_filename}",
    }


def _run_single_detection(
    job_id: str,
    video_filename: str,
    model_name: str,
    conf: float,
    iou: float,
    device: str,
    max_frames: int,
    cancel_event: threading.Event,
):
    """Background worker for single model evaluation."""
    try:
        video_path = UPLOADS_DIR / video_filename
        if not video_path.exists():
            raise FileNotFoundError(f"Video '{video_filename}' not found in uploads folder.")

        with jobs_lock:
            jobs[job_id]["status"] = "loading_model"
            jobs[job_id]["message"] = f"Loading model '{model_name}'..."

        # Load YOLO model once
        model_instance, active_device = model_manager.load_model(model_name, device=device)

        with jobs_lock:
            jobs[job_id]["status"] = "processing"
            jobs[job_id]["message"] = f"Processing video with {model_name} on {active_device}..."
            jobs[job_id]["device_used"] = active_device

        def progress_cb(update_dict: Dict[str, Any]):
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id].update(update_dict)

        def cancel_check() -> bool:
            return cancel_event.is_set()

        stats = video_processor.process_video(
            video_path=video_path,
            model=model_instance,
            model_name=model_name,
            device_str=active_device,
            conf_threshold=conf,
            iou_threshold=iou,
            max_frames=max_frames if max_frames > 0 else None,
            progress_callback=progress_cb,
            cancel_check=cancel_check,
        )


        with jobs_lock:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["result"] = stats
            jobs[job_id]["message"] = "Processing completed successfully!"

        # Add to history
        with history_lock:
            history_record = {
                "id": job_id,
                "timestamp": int(time.time()),
                "video": video_filename,
                "model": model_name,
                "confidence": conf,
                "iou": iou,
                "avg_confidence": stats["avg_confidence"],
                "avg_heads_per_frame": stats["avg_heads_per_frame"],
                "max_heads_per_frame": stats["max_heads_per_frame"],
                "total_detections": stats["total_head_detections"],
                "inference_fps": stats["inference_fps"],
                "processing_time": stats["processing_time"],
                "output_url": stats["output_url"],
                "download_url": stats["download_url"],
            }
            evaluation_history.append(history_record)

    except InterruptedError:
        logger.info(f"Job {job_id} cancelled.")
        with jobs_lock:
            jobs[job_id]["status"] = "cancelled"
            jobs[job_id]["message"] = "Detection was cancelled."
    except Exception as e:
        logger.error(f"Error in detection job {job_id}: {e}", exc_info=True)
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["message"] = f"Detection failed: {str(e)}"


@app.post("/api/detect")
async def start_detection(
    video: str = Form(...),
    model: str = Form(...),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
    device: str = Form("auto"),
    max_frames: int = Form(0),
):
    """Initiates an asynchronous head detection evaluation job."""
    video_path = UPLOADS_DIR / Path(video).name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file '{video}' not found in uploads.")

    model_path = MODELS_DIR / Path(model).name
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file '{model}' not found in models/.")

    # Validate parameters
    confidence = max(0.01, min(0.99, float(confidence)))
    iou = max(0.01, min(0.99, float(iou)))
    max_frames = max(0, int(max_frames))

    job_id = uuid.uuid4().hex
    cancel_event = threading.Event()

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "mode": "single",
            "status": "queued",
            "progress": 0,
            "current_frame": 0,
            "total_frames": 0,
            "elapsed_seconds": 0,
            "inference_fps": 0,
            "video": video,
            "model": model,
            "cancel_event": cancel_event,
            "error": None,
            "result": None,
            "message": "Detection job queued...",
        }

    thread = threading.Thread(
        target=_run_single_detection,
        args=(job_id, video, model, confidence, iou, device, max_frames, cancel_event),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}


def _run_comparison(
    job_id: str,
    video_filename: str,
    model_a_name: str,
    model_b_name: str,
    conf: float,
    iou: float,
    device: str,
    max_frames: int,
    cancel_event: threading.Event,
):

    """Background worker for side-by-side comparison of two models."""
    try:
        video_path = UPLOADS_DIR / video_filename
        if not video_path.exists():
            raise FileNotFoundError(f"Video '{video_filename}' not found.")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        a_stem = Path(model_a_name).stem
        b_stem = Path(model_b_name).stem
        comp_dir_name = f"comparison_{a_stem}_{b_stem}_{timestamp}"
        comp_dir = OUTPUTS_DIR / comp_dir_name
        comp_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for step_idx, m_name in enumerate([model_a_name, model_b_name], start=1):
            if cancel_event.is_set():
                raise InterruptedError("Comparison was cancelled.")

            with jobs_lock:
                jobs[job_id]["status"] = "processing"
                jobs[job_id]["current_model"] = m_name
                jobs[job_id]["message"] = f"Running Model {step_idx}/2: '{m_name}'..."

            model_inst, active_dev = model_manager.load_model(m_name, device=device)

            out_filename = f"{comp_dir_name}/model_{step_idx}_{Path(m_name).stem}.mp4"

            def make_progress_cb(model_idx: int):
                def cb(update_dict: Dict[str, Any]):
                    with jobs_lock:
                        if job_id in jobs:
                            pct = update_dict.get("progress", 0)
                            # Overall progress: 0-50% for model 1, 50-100% for model 2
                            overall_pct = round(((model_idx - 1) * 50.0) + (pct * 0.5), 1)
                            jobs[job_id]["progress"] = overall_pct
                            jobs[job_id]["current_frame"] = update_dict.get("current_frame", 0)
                            jobs[job_id]["total_frames"] = update_dict.get("total_frames", 0)
                            jobs[job_id]["inference_fps"] = update_dict.get("inference_fps", 0)
                            jobs[job_id]["elapsed_seconds"] = update_dict.get("elapsed_seconds", 0)
                return cb

            stats = video_processor.process_video(
                video_path=video_path,
                model=model_inst,
                model_name=m_name,
                device_str=active_dev,
                conf_threshold=conf,
                iou_threshold=iou,
                max_frames=max_frames if max_frames > 0 else None,
                output_filename=out_filename,
                progress_callback=make_progress_cb(step_idx),
                cancel_check=lambda: cancel_event.is_set(),
            )
            key = "model_a" if step_idx == 1 else "model_b"
            results[key] = stats

            with history_lock:
                evaluation_history.append({
                    "id": f"{job_id}_{key}",
                    "timestamp": int(time.time()),
                    "video": video_filename,
                    "model": m_name,
                    "confidence": conf,
                    "iou": iou,
                    "avg_confidence": stats["avg_confidence"],
                    "avg_heads_per_frame": stats["avg_heads_per_frame"],
                    "max_heads_per_frame": stats["max_heads_per_frame"],
                    "total_detections": stats["total_head_detections"],
                    "inference_fps": stats["inference_fps"],
                    "processing_time": stats["processing_time"],
                    "output_url": stats["output_url"],
                    "download_url": stats["download_url"],
                })

        with jobs_lock:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["result"] = results
            jobs[job_id]["message"] = "Comparison finished successfully!"

    except InterruptedError:
        logger.info(f"Comparison job {job_id} cancelled.")
        with jobs_lock:
            jobs[job_id]["status"] = "cancelled"
            jobs[job_id]["message"] = "Comparison was cancelled."
    except Exception as e:
        logger.error(f"Error in comparison job {job_id}: {e}", exc_info=True)
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["message"] = f"Comparison failed: {str(e)}"


@app.post("/api/compare")
async def start_comparison(
    video: str = Form(...),
    model_a: str = Form(...),
    model_b: str = Form(...),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
    device: str = Form("auto"),
    max_frames: int = Form(0),
):
    """Initiates side-by-side comparison of two models on the same video."""
    video_path = UPLOADS_DIR / Path(video).name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file '{video}' not found.")

    for m in [model_a, model_b]:
        if not (MODELS_DIR / Path(m).name).exists():
            raise HTTPException(status_code=404, detail=f"Model '{m}' not found in models/.")

    confidence = max(0.01, min(0.99, float(confidence)))
    iou = max(0.01, min(0.99, float(iou)))
    max_frames = max(0, int(max_frames))

    job_id = uuid.uuid4().hex
    cancel_event = threading.Event()

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "mode": "comparison",
            "status": "queued",
            "progress": 0,
            "current_frame": 0,
            "total_frames": 0,
            "elapsed_seconds": 0,
            "inference_fps": 0,
            "video": video,
            "model_a": model_a,
            "model_b": model_b,
            "cancel_event": cancel_event,
            "error": None,
            "result": None,
            "message": "Comparison job queued...",
        }

    thread = threading.Thread(
        target=_run_comparison,
        args=(job_id, video, model_a, model_b, confidence, iou, device, max_frames, cancel_event),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}



@app.get("/api/progress/{job_id}")
async def get_progress(job_id: str):
    """Returns the live progress and status of a processing job."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")

        # Return a copy excluding non-serializable objects (like threading.Event)
        return {
            "job_id": job["job_id"],
            "mode": job.get("mode", "single"),
            "status": job.get("status", "unknown"),
            "progress": job.get("progress", 0),
            "current_frame": job.get("current_frame", 0),
            "total_frames": job.get("total_frames", 0),
            "elapsed_seconds": job.get("elapsed_seconds", 0),
            "inference_fps": job.get("inference_fps", 0),
            "message": job.get("message", ""),
            "error": job.get("error"),
            "result": job.get("result"),
            "current_model": job.get("current_model"),
        }


@app.post("/api/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Requests cancellation of a running detection job."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")

        cancel_ev = job.get("cancel_event")
        if cancel_ev:
            cancel_ev.set()
        job["status"] = "cancelling"
        job["message"] = "Cancelling..."

    return {"status": "cancelling", "job_id": job_id}


@app.get("/api/result/{filename:path}")
async def get_result_video(filename: str, download: Optional[int] = Query(0)):
    """
    Serves the processed video from outputs/ with HTTP 206 Range support
    for smooth scrubbing and streaming in HTML5 video players.
    """
    # Prevent directory traversal
    clean_path = (OUTPUTS_DIR / filename).resolve()
    if not clean_path.is_relative_to(OUTPUTS_DIR.resolve()) or not clean_path.exists():
        raise HTTPException(status_code=404, detail="Result video not found.")

    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{clean_path.name}"'
    else:
        headers["Content-Disposition"] = f'inline; filename="{clean_path.name}"'

    return FileResponse(
        str(clean_path),
        media_type="video/mp4",
        headers=headers,
    )


@app.get("/api/uploads/{filename:path}")
async def get_uploaded_video(filename: str):
    """Serves the uploaded source video for browser playback."""
    clean_path = (UPLOADS_DIR / filename).resolve()
    if not clean_path.is_relative_to(UPLOADS_DIR.resolve()) or not clean_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded video not found.")

    return FileResponse(
        str(clean_path),
        media_type="video/mp4",
        headers={"Content-Disposition": f'inline; filename="{clean_path.name}"'},
    )


@app.get("/api/history")
async def get_history(video: Optional[str] = None):
    """
    Returns model evaluation records.
    Optionally filters by uploaded video filename.
    """
    with history_lock:
        if video:
            records = [h for h in evaluation_history if h.get("video") == video]
        else:
            records = list(evaluation_history)
        return {"history": list(reversed(records))}


@app.delete("/api/history")
async def clear_history():
    """Clears evaluation history."""
    with history_lock:
        evaluation_history.clear()
    return {"message": "History cleared."}


# Serve static frontend files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def root():
    """Serves the main frontend page."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "Frontend index.html not found. Ensure frontend/ is populated."},
        )
    return FileResponse(str(index_file))
