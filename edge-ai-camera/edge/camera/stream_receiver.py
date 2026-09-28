"""OpenCV frame source for a local MJPEG phone stream or laptop webcam."""

import logging
import time

import cv2

from edge.config import settings

logger = logging.getLogger(__name__)


class StreamReceiver:
    def __init__(self) -> None:
        if settings.camera_source not in {"webcam", "phone"}:
            raise ValueError("CAMERA_SOURCE must be 'webcam' or 'phone'")
        source = settings.camera_index if settings.camera_source == "webcam" else settings.phone_stream_url
        self.source = source
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera source {source!r}")
        if settings.camera_source == "webcam":
            self.capture.set(cv2.CAP_PROP_FPS, settings.fps)

    def read(self):
        return self.capture.read()

    def reconnect(self, delay: float = 1.0) -> bool:
        self.release()
        time.sleep(delay)
        self.capture = cv2.VideoCapture(self.source)
        connected = self.capture.isOpened()
        if connected:
            logger.info("Reconnected to camera source %s", self.source)
        return connected

    def release(self) -> None:
        self.capture.release()
