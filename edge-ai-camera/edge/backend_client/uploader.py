"""Persist completed events locally until the FastAPI backend accepts them."""

import json
import logging
import threading
import time
import time
from pathlib import Path
from typing import Any

import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


class EventUploader:
    def __init__(self, backend_url: str, queue_dir: Path, enabled: bool = True):
        self.backend_url = backend_url.rstrip("/")
        self.queue_dir = queue_dir
        self.enabled = enabled
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self.camera_id: str | None = None

    def start(self) -> None:
        if not self.enabled or (self._worker and self._worker.is_alive()):
            return
        self._worker = threading.Thread(target=self._run, name="event-uploader", daemon=True)
        self._worker.start()

    def set_camera(self, camera_id: str) -> None:
        self.camera_id = camera_id

    def stop(self) -> None:
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=1)

    def _run(self) -> None:
        last_heartbeat = 0.0
        while not self._stop.is_set():
            self.flush()
            if self.camera_id and time.monotonic() - last_heartbeat >= 15:
                self.heartbeat()
                last_heartbeat = time.monotonic()
            self._stop.wait(5)

    def heartbeat(self) -> None:
        if not self.enabled or not self.camera_id:
            return
        request = urllib.request.Request(
            f"{self.backend_url}/api/cameras/{self.camera_id}/heartbeat",
            data=b"{}", headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=3):
                pass
        except (urllib.error.URLError, TimeoutError, OSError):
            logger.debug("Camera heartbeat could not reach backend")

    def enqueue(self, metadata: dict[str, Any]) -> None:
        event_id = metadata["event_id"]
        queue_file = self.queue_dir / f"{event_id}.json"
        queue_file.write_text(json.dumps(metadata, indent=2) + "\n")
        logger.info("Queued completed event %s for backend upload", event_id)

    def flush(self) -> None:
        if not self.enabled:
            return
        for queue_file in sorted(self.queue_dir.glob("*.json")):
            self.flush_one(queue_file)

    def flush_one(self, queue_file: Path) -> bool:
        if not self.enabled:
            return False
        try:
            payload = queue_file.read_bytes()
            request = urllib.request.Request(
                f"{self.backend_url}/api/events", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(request, timeout=4) as response:
                if 200 <= response.status < 300:
                    event = json.loads(payload)
                    video_path = Path(event.get("video_path", ""))
                    if video_path.is_file():
                        upload = urllib.request.Request(
                            f"{self.backend_url}/api/events/{event['event_id']}/video",
                            data=video_path.read_bytes(),
                            headers={"Content-Type": "video/mp4"}, method="PUT",
                        )
                        with urllib.request.urlopen(upload, timeout=30) as video_response:
                            if not 200 <= video_response.status < 300:
                                return False
                    queue_file.unlink(missing_ok=True)
                    logger.info("Backend accepted event %s", queue_file.stem)
                    return True
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("Event %s queued for backend retry: %s", queue_file.stem, exc)
        return False
