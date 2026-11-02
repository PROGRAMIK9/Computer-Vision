"""Run TensorFlow and PyTorch inference in isolated worker processes.

Some ARM/Linux builds of TensorFlow and PyTorch each load LLVM command-line
registries into the process. Isolating the runtimes prevents duplicate LLVM
option registration while preserving the existing model files and outputs.
"""

import logging
import multiprocessing as mp
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


def _anomaly_worker(connection, model_path: str, threshold: float) -> None:
    """Child process entry point; only this process imports TensorFlow."""
    try:
        import cv2
        import tensorflow as tf

        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                logger.debug("TensorFlow GPU memory growth already initialized")

        model = tf.keras.models.load_model(model_path, compile=False)
        connection.send(("ready", None))
        while True:
            frame = connection.recv()
            if frame is None:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            batch = np.expand_dims(resized.astype(np.float32), axis=0)
            prediction = model.predict(batch, verbose=0)
            probability = float(np.asarray(prediction).reshape(-1)[0])
            connection.send(("result", {
                "anomaly_probability": probability,
                "normal_probability": 1.0 - probability,
                "label": "ANOMALY" if probability > threshold else "NORMAL",
                "threshold": threshold,
            }))
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _yolo_worker(connection, model_path: str, confidence: float) -> None:
    """Child process entry point; only this process imports PyTorch/Ultralytics."""
    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
        connection.send(("ready", None))
        while True:
            frame = connection.recv()
            if frame is None:
                break
            prediction = model.predict(source=frame, conf=confidence, verbose=False)[0]
            detections = []
            if prediction.boxes is not None:
                boxes = prediction.boxes
                for index in range(len(boxes)):
                    xyxy = boxes.xyxy[index].cpu().numpy().astype(float).tolist()
                    class_id = int(boxes.cls[index].cpu().item())
                    detections.append({
                        "class_name": model.names[class_id],
                        "confidence": float(boxes.conf[index].cpu().item()),
                        "bbox": xyxy,
                    })
            connection.send(("result", {"detections": detections}))
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class _ModelWorker:
    def __init__(self, name: str, target: Callable, args: tuple[Any, ...]) -> None:
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self.name = name
        self.connection = parent
        self.process = context.Process(target=target, args=(child, *args), name=f"{name}-inference")
        self.process.start()
        child.close()
        try:
            self._wait_for_message(expected="ready", timeout=180)
        except BaseException:
            self.close()
            raise
        logger.info("%s worker ready (pid=%s)", name, self.process.pid)

    def _wait_for_message(self, expected: str, timeout: float = 120.0) -> Any:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.connection.poll(0.2):
                try:
                    status, payload = self.connection.recv()
                except (EOFError, OSError) as exc:
                    raise RuntimeError(f"{self.name} worker exited unexpectedly") from exc
                if status == "error":
                    raise RuntimeError(f"{self.name} worker failed:\n{payload}")
                if status != expected:
                    raise RuntimeError(f"{self.name} worker returned unexpected status {status!r}")
                return payload
            if not self.process.is_alive():
                self.process.join(timeout=0.1)
                raise RuntimeError(
                    f"{self.name} worker exited with code {self.process.exitcode}; "
                    "see the worker's startup/inference error above"
                )
        raise TimeoutError(f"{self.name} inference timed out after {timeout:g}s")

    def predict(self, frame: np.ndarray) -> dict[str, Any]:
        if not self.process.is_alive():
            raise RuntimeError(f"{self.name} worker is not running (exit={self.process.exitcode})")
        try:
            self.connection.send(frame)
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise RuntimeError(f"Could not send frame to {self.name} worker") from exc
        return self._wait_for_message(expected="result")

    def close(self) -> None:
        if self.process.is_alive():
            try:
                self.connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        self.connection.close()


class ModelInference:
    """Coordinate isolated model processes and merge their per-frame results."""

    def __init__(self, anomaly_model_path: Path, yolo_model_path: Path,
                 anomaly_threshold: float = 0.5, yolo_confidence: float = 0.35,
                 anomaly_enabled: bool = True, yolo_enabled: bool = True) -> None:
        self.anomaly_model_path = anomaly_model_path
        self.yolo_model_path = yolo_model_path
        self.anomaly_threshold = anomaly_threshold
        self.yolo_confidence = yolo_confidence
        self.anomaly_enabled = anomaly_enabled
        self.yolo_enabled = yolo_enabled
        self._workers: list[_ModelWorker] = []
        self._anomaly_worker: _ModelWorker | None = None
        self._yolo_worker: _ModelWorker | None = None

    def load(self) -> None:
        if not self.anomaly_enabled and not self.yolo_enabled:
            raise ValueError("At least one inference model must be enabled")
        if self.anomaly_enabled and not self.anomaly_model_path.is_file():
            raise FileNotFoundError(f"Anomaly model not found: {self.anomaly_model_path}")
        if self.yolo_enabled and not self.yolo_model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {self.yolo_model_path}")

        try:
            if self.anomaly_enabled:
                logger.info("Starting isolated anomaly worker: %s", self.anomaly_model_path)
                self._anomaly_worker = _ModelWorker(
                    "anomaly", _anomaly_worker,
                    (str(self.anomaly_model_path), self.anomaly_threshold),
                )
                self._workers.append(self._anomaly_worker)
            if self.yolo_enabled:
                logger.info("Starting isolated YOLO worker: %s", self.yolo_model_path)
                self._yolo_worker = _ModelWorker(
                    "YOLO", _yolo_worker,
                    (str(self.yolo_model_path), self.yolo_confidence),
                )
                self._workers.append(self._yolo_worker)
        except BaseException:
            self.close()
            raise

    def predict(self, frame: np.ndarray) -> dict[str, Any]:
        result: dict[str, Any] = {"detections": []}
        if self._anomaly_worker is not None:
            result.update(self._anomaly_worker.predict(frame))
        if self._yolo_worker is not None:
            result.update(self._yolo_worker.predict(frame))
        return result

    def close(self) -> None:
        for worker in reversed(self._workers):
            worker.close()
        self._workers.clear()
        self._anomaly_worker = None
        self._yolo_worker = None
