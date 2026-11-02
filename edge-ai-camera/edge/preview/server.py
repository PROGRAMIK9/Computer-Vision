"""Small MJPEG server for viewing the edge camera in a LAN browser."""

import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PreviewServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._last_encode = 0.0
        self._http: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in {"/", "/index.html"}:
                    body = ("<!doctype html><title>Edge camera preview</title>"
                            "<style>body{margin:0;background:#111;display:grid;place-items:center;height:100vh}"
                            "img{max-width:100vw;max-height:100vh}</style>"
                            "<img src='/stream.mjpg' alt='Live camera preview'>").encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path != "/stream.mjpg":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Connection", "close")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while preview._http is not None:
                        with preview._lock:
                            jpeg = preview._jpeg
                        if jpeg is None:
                            time.sleep(0.1)
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg + b"\r\n")
                        self.wfile.flush()
                        time.sleep(0.08)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

            def log_message(self, fmt, *args):
                logger.debug("Preview HTTP: " + fmt, *args)

        self._http = ThreadingHTTPServer((self.host, self.port), Handler)
        self._http.daemon_threads = True
        self._thread = threading.Thread(target=self._http.serve_forever,
                                        name="mjpeg-preview", daemon=True)
        self._thread.start()
        logger.info("Browser preview ready at http://<pi-ip>:%d/", self.port)

    def update(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last_encode < 0.2:
            return
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self._lock:
                self._jpeg = encoded.tobytes()
                self._last_encode = now

    def stop(self) -> None:
        if self._http is not None:
            self._http.shutdown()
            self._http.server_close()
            self._http = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
