import os
import sys
import time
import shutil
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def get_ffmpeg_executable() -> Optional[str]:
    """Finds available ffmpeg binary from imageio-ffmpeg or system PATH."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    return None


def format_duration(seconds: float) -> str:
    """Formats seconds into MM:SS or HH:MM:SS."""
    seconds = int(max(0, seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class VideoProcessor:
    """
    Handles frame-by-frame object detection, bounding box annotation,
    progress monitoring, and browser-ready video generation using OpenCV & YOLO.
    """

    def __init__(self, outputs_dir: Optional[Path] = None):
        if outputs_dir is None:
            self.outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
        else:
            self.outputs_dir = Path(outputs_dir)

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_exe = get_ffmpeg_executable()
        if self.ffmpeg_exe:
            logger.info(f"Using FFmpeg encoder at: {self.ffmpeg_exe}")
        else:
            logger.warning("No FFmpeg found; video encoding will rely exclusively on OpenCV.")

    def inspect_video(self, video_path: Path) -> Dict[str, Any]:
        """
        Validates the video file and inspects resolution, FPS, and frame count.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(
                f"Unable to read video file '{video_path.name}'. "
                f"The file might be corrupted or in an unsupported codec."
            )

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if width <= 0 or height <= 0:
                raise ValueError("Video has invalid dimensions (width/height <= 0).")

            if fps <= 0 or np.isnan(fps):
                fps = 30.0

            duration_sec = total_frames / fps if total_frames > 0 else 0.0

            return {
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total_frames,
                "duration_sec": duration_sec,
                "duration_formatted": format_duration(duration_sec),
            }
        finally:
            cap.release()

    def process_video(
        self,
        video_path: Path,
        model: Any,
        model_name: str,
        device_str: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        output_filename: Optional[str] = None,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Processes a video frame-by-frame:
        1. Reads frame with OpenCV VideoCapture
        2. Performs YOLO inference
        3. Annotates bounding boxes with 'head {conf:.2f}' or '{cls_name} {conf:.2f}'
        4. Writes frame to output video
        5. Converts to web-streamable H.264 MP4 if FFmpeg is available
        6. Computes comprehensive detection statistics
        """
        video_info = self.inspect_video(video_path)
        width = video_info["width"]
        height = video_info["height"]
        fps = video_info["fps"]
        raw_total_frames = video_info["total_frames"]

        if max_frames and max_frames > 0:
            total_frames = min(raw_total_frames, max_frames) if raw_total_frames > 0 else max_frames
        else:
            total_frames = raw_total_frames


        # Determine output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_stem = video_path.stem
        model_stem = Path(model_name).stem

        if output_filename is None:
            final_filename = f"{video_stem}_{model_stem}_{timestamp}.mp4"
        else:
            final_filename = output_filename

        final_output_path = self.outputs_dir / final_filename
        final_output_path.parent.mkdir(parents=True, exist_ok=True)

        temp_raw_path = self.outputs_dir / f"temp_raw_{os.getpid()}_{timestamp}.mp4"

        # Initialize OpenCV VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_raw_path), fourcc, fps, (width, height))

        if not writer.isOpened():
            raise RuntimeError(
                f"Failed to initialize OpenCV VideoWriter for output: {temp_raw_path.name}"
            )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            writer.release()
            raise RuntimeError(f"Failed to open video source: {video_path.name}")

        start_time = time.time()
        frames_processed = 0
        total_head_detections = 0
        max_heads_in_frame = 0
        all_confidences = []

        # Determine class name handling
        # If single class in model, treat that class as 'head'
        # If model has class names dict, inspect
        model_names = getattr(model, "names", {})
        is_single_class = len(model_names) <= 1

        # Calculate dynamic text scaling for bounding box labels
        scale_factor = max(width, height) / 1000.0
        font_scale = max(0.45, min(1.2, 0.5 * scale_factor))
        thickness = max(1, int(1.8 * scale_factor))
        box_thickness = max(2, int(2.2 * scale_factor))

        # Color definition: High-visibility vibrant neon cyan / mint
        # BGR format: (B=235, G=206, R=0) or vibrant green (B=40, G=225, R=80)
        box_color = (40, 225, 80)
        text_bg_color = (25, 140, 50)
        text_color = (255, 255, 255)

        logger.info(
            f"Starting video processing: {video_path.name} with model {model_name} "
            f"(device={device_str}, conf={conf_threshold}, iou={iou_threshold})"
        )

        try:
            while True:
                if cancel_check and cancel_check():
                    logger.info("Video processing cancelled by user.")
                    raise InterruptedError("Video processing was cancelled.")

                if max_frames and max_frames > 0 and frames_processed >= max_frames:
                    logger.info(f"Reached user requested frame limit of {max_frames} frames.")
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    break


                # Run inference on current frame
                results = model(
                    frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    device=device_str,
                    verbose=False,
                )

                frame_heads = 0
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    xyxy_arr = boxes.xyxy.cpu().numpy()
                    conf_arr = boxes.conf.cpu().numpy()
                    cls_arr = boxes.cls.cpu().numpy().astype(int)

                    for idx in range(len(xyxy_arr)):
                        x1, y1, x2, y2 = xyxy_arr[idx].astype(int)
                        conf = float(conf_arr[idx])
                        cls_id = int(cls_arr[idx])

                        # Determine label
                        if is_single_class:
                            label_text = f"head {conf:.2f}"
                            is_head = True
                        else:
                            cls_name = model_names.get(cls_id, f"class_{cls_id}")
                            if str(cls_name).strip().lower() == "head":
                                label_text = f"head {conf:.2f}"
                                is_head = True
                            else:
                                label_text = f"{cls_name} {conf:.2f}"
                                is_head = ("head" in str(cls_name).lower())

                        if is_head:
                            frame_heads += 1
                            total_head_detections += 1
                            all_confidences.append(conf)

                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

                        # Draw text background banner with boundary safety
                        (text_w, text_h), baseline = cv2.getTextSize(
                            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                        )
                        if y1 - text_h - 8 > 0:
                            label_y1 = y1 - text_h - 8
                            label_y2 = y1
                            text_y = y1 - 4
                        else:
                            label_y1 = y1
                            label_y2 = y1 + text_h + 8
                            text_y = y1 + text_h + 2

                        label_x1 = max(0, x1)
                        label_x2 = min(width - 1, x1 + text_w + 6)
                        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), text_bg_color, -1)
                        cv2.putText(
                            frame,
                            label_text,
                            (label_x1 + 3, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            text_color,
                            thickness,
                            cv2.LINE_AA,
                        )

                if frame_heads > max_heads_in_frame:
                    max_heads_in_frame = frame_heads

                # Draw clean on-screen HUD banner on the video
                hud_text = f"Model: {model_stem} | Heads in Frame: {frame_heads} | Total: {total_head_detections}"
                (hud_w, hud_h), _ = cv2.getTextSize(
                    hud_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, max(1, thickness - 1)
                )
                cv2.rectangle(frame, (12, 10), (22 + hud_w, 18 + hud_h + 6), (15, 23, 42), -1)
                cv2.rectangle(frame, (12, 10), (22 + hud_w, 18 + hud_h + 6), (51, 65, 85), 1)
                cv2.putText(
                    frame,
                    hud_text,
                    (17, 13 + hud_h + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.85,
                    (245, 245, 245),
                    max(1, thickness - 1),
                    cv2.LINE_AA,
                )


                # Write processed frame
                writer.write(frame)
                frames_processed += 1

                # Update progress
                if progress_callback and (frames_processed % 3 == 0 or frames_processed == total_frames):
                    now = time.time()
                    elapsed = max(0.001, now - start_time)
                    progress_pct = (
                        min(99.0, round((frames_processed / max(1, total_frames)) * 100, 1))
                        if total_frames > 0
                        else 50.0
                    )
                    current_fps = round(frames_processed / elapsed, 1)

                    progress_callback({
                        "status": "processing",
                        "progress": progress_pct,
                        "current_frame": frames_processed,
                        "total_frames": total_frames,
                        "elapsed_seconds": round(elapsed, 1),
                        "inference_fps": current_fps,
                    })

        finally:
            cap.release()
            writer.release()

        end_time = time.time()
        elapsed_total = max(0.01, end_time - start_time)
        overall_inference_fps = round(frames_processed / elapsed_total, 1)
        avg_confidence = round(float(np.mean(all_confidences)), 2) if all_confidences else 0.0
        avg_heads_per_frame = (
            round(total_head_detections / max(1, frames_processed), 2)
        )

        logger.info(
            f"Inference finished: {frames_processed} frames in {elapsed_total:.2f}s "
            f"({overall_inference_fps} FPS). Total detections: {total_head_detections}"
        )

        # Transcode or move raw video to browser-streamable format
        self._finalize_output_video(temp_raw_path, final_output_path)

        # Report 100% progress
        if progress_callback:
            progress_callback({
                "status": "completed",
                "progress": 100,
                "current_frame": frames_processed,
                "total_frames": total_frames,
                "elapsed_seconds": round(elapsed_total, 1),
                "inference_fps": overall_inference_fps,
            })

        # Return comprehensive detection statistics
        return {
            "model": model_name,
            "video": video_path.name,
            "resolution": f"{width} × {height}",
            "fps": round(fps, 2),
            "duration": video_info["duration_formatted"],
            "total_frames": total_frames,
            "frames_processed": frames_processed,
            "total_head_detections": total_head_detections,
            "avg_heads_per_frame": avg_heads_per_frame,
            "max_heads_per_frame": max_heads_in_frame,
            "avg_confidence": avg_confidence,
            "processing_time": f"{round(elapsed_total, 1)} seconds",
            "processing_time_sec": round(elapsed_total, 1),
            "inference_fps": overall_inference_fps,
            "output_filename": final_filename,
            "output_relative_path": str(final_output_path.relative_to(self.outputs_dir.parent)),
            "output_url": f"/api/result/{final_filename}",
            "download_url": f"/api/result/{final_filename}?download=1",
            "device_used": device_str,
        }

    def _finalize_output_video(self, temp_raw: Path, final_target: Path) -> None:
        """
        Converts the raw OpenCV output video to web-streamable H.264 MP4 with faststart,
        or falls back to moving the file if FFmpeg is unavailable.
        """
        if not temp_raw.exists() or temp_raw.stat().st_size == 0:
            raise RuntimeError("Processed video file was empty or missing.")

        if self.ffmpeg_exe:
            try:
                logger.info(f"Encoding web-compatible H.264 video to: {final_target.name}")
                cmd = [
                    self.ffmpeg_exe,
                    "-y",
                    "-i",
                    str(temp_raw),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(final_target),
                ]
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    timeout=300,
                )
                # Remove temporary raw file upon successful transcode
                if temp_raw.exists():
                    temp_raw.unlink()
                return
            except Exception as ffmpeg_err:
                logger.warning(
                    f"FFmpeg transcode failed ({ffmpeg_err}). Falling back to raw OpenCV video."
                )

        # Fallback: rename/move raw file directly to final target
        if final_target.exists():
            final_target.unlink()
        shutil.move(str(temp_raw), str(final_target))
