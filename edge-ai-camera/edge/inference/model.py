"""Adapters for the existing V1.1 anomaly and object detection models.

The anomaly model uses the V1.1 demo's preprocessing: BGR to RGB, bilinear
resize to 224x224, float32 pixels in [0, 255], then one sigmoid probability.
YOLO sees the original frame and supplies object labels and bounding boxes.
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ModelInference:
    """Load model assets on construction and infer on one BGR frame at a time."""

    def __init__(self, anomaly_model_path: Path, yolo_model_path: Path,
                 anomaly_threshold: float = 0.5, yolo_confidence: float = 0.35) -> None:
        self.anomaly_model_path = anomaly_model_path
        self.yolo_model_path = yolo_model_path
        self.anomaly_threshold = anomaly_threshold
        self.yolo_confidence = yolo_confidence
        self._anomaly_model: Any = None
        self._yolo_model: Any = None

    def load(self) -> None:
        """Load both models once; keep imports lazy so Phase 1 remains lightweight."""
        if not self.anomaly_model_path.is_file():
            raise FileNotFoundError(f"Anomaly model not found: {self.anomaly_model_path}")
        if not self.yolo_model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {self.yolo_model_path}")

        import tensorflow as tf
        from ultralytics import YOLO

        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                logger.debug("TensorFlow GPU memory growth already initialized")

        logger.info("Loading anomaly model: %s", self.anomaly_model_path)
        self._anomaly_model = tf.keras.models.load_model(self.anomaly_model_path, compile=False)
        logger.info("Loading YOLO model: %s", self.yolo_model_path)
        self._yolo_model = YOLO(str(self.yolo_model_path))

    def predict(self, frame: np.ndarray) -> dict[str, Any]:
        if self._anomaly_model is None or self._yolo_model is None:
            raise RuntimeError("Call load() before predict()")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        batch = np.expand_dims(resized.astype(np.float32), axis=0)
        prediction = self._anomaly_model.predict(batch, verbose=0)
        anomaly_probability = float(np.asarray(prediction).reshape(-1)[0])

        yolo_result = self._yolo_model.predict(
            source=frame, conf=self.yolo_confidence, verbose=False
        )[0]
        detections = []
        if yolo_result.boxes is not None:
            boxes = yolo_result.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(float).tolist()
                class_id = int(boxes.cls[i].cpu().item())
                detections.append({
                    "class_name": self._yolo_model.names[class_id],
                    "confidence": float(boxes.conf[i].cpu().item()),
                    "bbox": xyxy,
                })

        return {
            "anomaly_probability": anomaly_probability,
            "normal_probability": 1.0 - anomaly_probability,
            "label": "ANOMALY" if anomaly_probability > self.anomaly_threshold else "NORMAL",
            "threshold": self.anomaly_threshold,
            "detections": detections,
        }
