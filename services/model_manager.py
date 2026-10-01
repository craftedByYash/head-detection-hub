import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages discovery, device selection, caching, and inference-ready loading
    of Ultralytics YOLO head detection models (.pt files).
    """

    def __init__(self, models_dir: Optional[Path] = None):
        if models_dir is None:
            self.models_dir = Path(__file__).resolve().parent.parent / "models"
        else:
            self.models_dir = Path(models_dir)

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._model_cache: Dict[str, Any] = {}

    def get_system_device_info(self) -> Dict[str, Any]:
        """Returns details about hardware acceleration availability."""
        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "Not available"
        device_count = torch.cuda.device_count() if cuda_available else 0

        return {
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "device_count": device_count,
            "recommended_device": "cuda" if cuda_available else "cpu",
            "torch_version": torch.__version__,
        }

    def resolve_device(self, requested_device: str = "auto") -> Tuple[str, str]:
        """
        Determines the actual PyTorch device to use.
        Returns:
            Tuple of (device_string, display_description)
            e.g. ("cuda:0", "CUDA (NVIDIA GeForce RTX ...)") or ("cpu", "CPU")
        """
        cuda_available = torch.cuda.is_available()
        requested = (requested_device or "auto").strip().lower()

        if requested == "cuda":
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                return "cuda", f"CUDA GPU ({gpu_name})"
            else:
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return "cpu", "CPU (CUDA unavailable)"
        elif requested == "cpu":
            return "cpu", "CPU"
        else:  # auto
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                return "cuda", f"CUDA GPU ({gpu_name})"
            else:
                return "cpu", "CPU"

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Scans the models/ directory for all available .pt model weights.
        Returns a list of model information dicts.
        """
        models = []
        if not self.models_dir.exists():
            return models

        for file_path in sorted(self.models_dir.glob("*.pt")):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    size_mb = round(stat.st_size / (1024 * 1024), 2)
                    models.append({
                        "filename": file_path.name,
                        "size_mb": size_mb,
                        "modified": stat.st_mtime
                    })
                except Exception as e:
                    logger.warning(f"Could not read stat for {file_path.name}: {e}")
                    models.append({
                        "filename": file_path.name,
                        "size_mb": 0.0,
                        "modified": 0
                    })

        return models

    def load_model(self, model_name: str, device: str = "auto") -> Tuple[YOLO, str]:
        """
        Loads a YOLO model once, configuring the target execution device.
        Caches models in memory to avoid redundant disk I/O and weight initializations.
        
        Returns:
            Tuple of (YOLO model instance, resolved_device_string)
        """
        if not model_name:
            raise ValueError("Model name must be specified.")

        # Sanitize model name to prevent path traversal
        clean_name = Path(model_name).name
        model_path = self.models_dir / clean_name

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file '{clean_name}' was not found in {self.models_dir}. "
                f"Please ensure your .pt weights are saved inside the 'models/' directory."
            )

        target_device, device_desc = self.resolve_device(device)
        cache_key = f"{clean_name}_{target_device}"

        if cache_key in self._model_cache:
            logger.info(f"Reusing cached model '{clean_name}' on {target_device}")
            return self._model_cache[cache_key], target_device

        logger.info(f"Loading YOLO model from {model_path} onto {target_device} ({device_desc})")
        try:
            model = YOLO(str(model_path))
            
            # Explicitly set device if supported
            if target_device.startswith("cuda"):
                try:
                    model.to("cuda")
                except Exception as cuda_err:
                    logger.warning(f"Failed to move model to CUDA: {cuda_err}. Falling back to CPU.")
                    target_device = "cpu"
                    model.to("cpu")
            else:
                model.to("cpu")

            self._model_cache[cache_key] = model
            return model, target_device

        except Exception as e:
            logger.error(f"Error loading model '{clean_name}': {e}", exc_info=True)
            raise RuntimeError(
                f"Unable to load '{clean_name}'. "
                f"Please verify that this is a valid Ultralytics YOLO .pt model. "
                f"Underlying error: {str(e)}"
            )

    def clear_cache(self):
        """Clears cached models from memory."""
        self._model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
