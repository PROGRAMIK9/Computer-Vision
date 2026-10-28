"""Picamera2 frame source for Raspberry Pi camera modules."""

import logging
import time

import cv2

logger = logging.getLogger(__name__)


class PiCamera:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 15) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 is not installed. On Raspberry Pi OS install it with "
                "'sudo apt install python3-picamera2'."
            ) from exc

        self._camera = Picamera2()
        config = self._camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps},
            buffer_count=4,
        )
        self._camera.configure(config)
        self._camera.start()
        time.sleep(1.0)
        logger.info("Picamera2 started at %dx%d, target %d FPS", width, height, fps)

    def read(self):
        try:
            rgb = self._camera.capture_array("main")
            return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            logger.exception("Could not capture a Pi camera frame")
            return False, None

    def reconnect(self, delay: float = 1.0) -> bool:
        self.release()
        time.sleep(delay)
        try:
            self._camera.start()
            return True
        except Exception:
            logger.exception("Could not restart Pi camera")
            return False

    def release(self) -> None:
        if getattr(self, "_camera", None) is not None:
            try:
                self._camera.stop()
            except Exception:
                logger.debug("Pi camera was already stopped")

