"""Persist event-only video and metadata."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class EventStorage:
    def __init__(self, events_dir: Path, fps: int = 30, fourcc: str = "mp4v") -> None:
        self.events_dir = events_dir
        self.fps = fps
        self.fourcc = fourcc

    def save(self, frames: list[np.ndarray], camera_id: str, camera_name: str,
             started_at: datetime, ended_at: datetime,
             detections: list[dict[str, Any]] | None = None,
             anomaly: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not frames:
            logger.warning("Skipping event with no frames")
            return None

        event_id = str(uuid4())
        event_dir = self.events_dir / f"event_{event_id}"
        event_dir.mkdir(parents=True, exist_ok=False)
        height, width = frames[0].shape[:2]
        video_path = event_dir / "video.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*self.fourcc),
                                 self.fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {video_path}")
        try:
            for frame in frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
        finally:
            writer.release()

        duration = max(0.0, (ended_at - started_at).total_seconds())
        metadata = {
            "event_id": event_id,
            "camera_id": camera_id,
            "camera_name": camera_name,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration": round(duration, 3),
            "video_path": str(video_path),
            "frame_count": len(frames),
            "detections": detections or [],
            "anomaly": anomaly,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (event_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        logger.info("Saved event %s (%d frames, %.1fs)", event_id, len(frames), duration)
        return metadata
