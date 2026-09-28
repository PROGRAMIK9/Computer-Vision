"""Environment-backed configuration for the edge prototype."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


@dataclass(frozen=True)
class Settings:
    camera_source: str = os.getenv("CAMERA_SOURCE", "webcam").strip().lower()
    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    phone_stream_url: str = os.getenv("PHONE_STREAM_URL", "http://127.0.0.1:8081/video")
    camera_id: str = os.getenv("CAMERA_ID", "webcam-01")
    camera_name: str = os.getenv("CAMERA_NAME", "Laptop webcam")
    fps: int = max(1, int(os.getenv("FPS", "30")))
    buffer_seconds: float = max(0.1, float(os.getenv("BUFFER_SECONDS", "5")))
    motion_threshold: int = max(1, int(os.getenv("MOTION_THRESHOLD", "25")))
    min_contour_area: float = max(0.0, float(os.getenv("MIN_CONTOUR_AREA", "800")))
    blur_size: int = max(1, int(os.getenv("BLUR_SIZE", "21")))
    no_motion_timeout: float = max(0.1, float(os.getenv("NO_MOTION_TIMEOUT", "3.0")))
    resize_width: int = max(0, int(os.getenv("RESIZE_WIDTH", "640")))
    events_dir: Path = Path(os.getenv("EVENTS_DIR", "storage/events"))
    video_fourcc: str = os.getenv("VIDEO_FOURCC", "mp4v")
    show_preview: bool = _bool("SHOW_PREVIEW", True)
    inference_enabled: bool = _bool("INFERENCE_ENABLED", False)
    inference_interval: float = max(0.1, float(os.getenv("INFERENCE_INTERVAL", "0.2")))
    anomaly_threshold: float = float(os.getenv("ANOMALY_THRESHOLD", "0.5"))
    yolo_confidence: float = float(os.getenv("YOLO_CONFIDENCE", "0.35"))
    anomaly_model_path: Path = Path(os.getenv(
        "ANOMALY_MODEL_PATH", "../tested_models/7_train_7_finetune/model_final.keras"))
    yolo_model_path: Path = Path(os.getenv("YOLO_MODEL_PATH", "../weights/yolo26s.pt"))
    backend_url: str = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
    backend_upload_enabled: bool = _bool("BACKEND_UPLOAD_ENABLED", True)
    upload_queue_dir: Path = _path("UPLOAD_QUEUE_DIR", "storage/upload_queue")
    backend_db_path: Path = _path("BACKEND_DB_PATH", "storage/backend.db")
    backend_media_dir: Path = _path("BACKEND_MEDIA_DIR", "storage/backend_media")

    @property
    def resolved_events_dir(self) -> Path:
        return self.events_dir if self.events_dir.is_absolute() else PROJECT_ROOT / self.events_dir

    def resolve_project_path(self, path: Path) -> Path:
        """Resolve model paths from this project, independent of shell cwd.

        Paths may be written relative to edge-ai-camera (e.g. ../tested_models)
        or relative to the Computer-Vision repo root (e.g. tested_models).
        """
        if path.is_absolute():
            return path.resolve()
        project_relative = (PROJECT_ROOT / path).resolve()
        if project_relative.exists():
            return project_relative
        return (WORKSPACE_ROOT / path).resolve()


settings = Settings()
