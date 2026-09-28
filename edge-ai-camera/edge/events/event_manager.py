"""Motion event lifecycle: preserve pre-roll, record, and close after a quiet period."""

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

from edge.storage.event_storage import EventStorage

logger = logging.getLogger(__name__)


class EventManager:
    def __init__(self, storage: EventStorage, camera_id: str, camera_name: str,
                 no_motion_timeout: float, on_complete=None) -> None:
        self.storage = storage
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.no_motion_timeout = no_motion_timeout
        self.on_complete = on_complete
        self._frames: list[np.ndarray] | None = None
        self._started_at: datetime | None = None
        self._last_motion: datetime | None = None
        self._detections: list[dict[str, Any]] = []
        self._latest_anomaly: dict[str, Any] | None = None

    @property
    def recording(self) -> bool:
        return self._frames is not None

    @property
    def latest_anomaly(self) -> dict[str, Any] | None:
        return self._latest_anomaly

    @property
    def event_token(self) -> str | None:
        return self._started_at.isoformat() if self._started_at else None

    def update(self, frame: np.ndarray, motion: bool, pre_roll: list[np.ndarray]) -> None:
        now = datetime.now(timezone.utc)
        if motion:
            if self._frames is None:
                self._frames = [item.copy() for item in pre_roll]
                self._started_at = now
                logger.info("Motion detected; event recording started")
            self._frames.append(frame.copy())
            self._last_motion = now
        elif self._frames is not None:
            self._frames.append(frame.copy())
            if self._last_motion and (now - self._last_motion).total_seconds() >= self.no_motion_timeout:
                self._complete(now)

    def add_inference(self, result: dict[str, Any], timestamp: datetime | None = None) -> None:
        """Attach one throttled model result to the active motion event."""
        if not self.recording:
            return
        timestamp = timestamp or datetime.now(timezone.utc)
        self._latest_anomaly = {
            "anomaly_probability": result["anomaly_probability"],
            "normal_probability": result["normal_probability"],
            "label": result["label"],
            "threshold": result["threshold"],
            "timestamp": timestamp.isoformat(),
        }
        for detection in result.get("detections", []):
            self._detections.append({**detection, "timestamp": timestamp.isoformat()})

    def _complete(self, ended_at: datetime) -> None:
        assert self._frames is not None and self._started_at is not None
        frames = self._frames
        started_at = self._started_at
        self._frames = None
        self._started_at = None
        self._last_motion = None
        metadata = self.storage.save(frames, self.camera_id, self.camera_name, started_at, ended_at,
                                     detections=self._detections, anomaly=self._latest_anomaly)
        if metadata is not None and self.on_complete is not None:
            self.on_complete(metadata)
        self._detections = []
        self._latest_anomaly = None

    def close(self) -> None:
        """Save an active event on a clean shutdown."""
        if self._frames is not None:
            self._complete(datetime.now(timezone.utc))
