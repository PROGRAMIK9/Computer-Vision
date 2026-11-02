"""Raspberry Pi camera -> motion event -> optional on-device inference -> dashboard."""

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone

import cv2

from edge.backend_client.uploader import EventUploader
from edge.config import settings
from edge.events.event_manager import EventManager
from edge.inference.model import ModelInference
from edge.motion.detector import MotionDetector
from edge.storage.event_storage import EventStorage
from edge_iot.camera import PiCamera


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("edge-iot")
    capture = PiCamera(
        width=settings.pi_camera_width,
        height=settings.pi_camera_height,
        fps=settings.fps,
    )
    detector = MotionDetector(settings.motion_threshold, settings.min_contour_area, settings.blur_size)
    storage = EventStorage(settings.resolved_events_dir, settings.fps, settings.video_fourcc)
    uploader = EventUploader(settings.backend_url, settings.upload_queue_dir, settings.backend_upload_enabled)
    uploader.set_camera(settings.camera_id)
    uploader.start()
    manager = EventManager(
        storage, settings.camera_id, settings.camera_name,
        settings.no_motion_timeout, on_complete=uploader.enqueue,
    )

    inference = None
    if settings.inference_enabled:
        logger.info("Inference enabled: anomaly model=%s (threshold=%.2f), YOLO model=%s (confidence=%.2f)",
                    settings.resolve_project_path(settings.anomaly_model_path), settings.anomaly_threshold,
                    settings.resolve_project_path(settings.yolo_model_path), settings.yolo_confidence)
        inference = ModelInference(
            settings.resolve_project_path(settings.anomaly_model_path),
            settings.resolve_project_path(settings.yolo_model_path),
            settings.anomaly_threshold,
            settings.yolo_confidence,
            settings.anomaly_inference_enabled,
            settings.yolo_inference_enabled,
        )
        inference.load()

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi-inference") if inference else None
    future: Future | None = None
    future_event: str | None = None
    last_inference = 0.0
    logger.info("Camera %s ready; inference %s", settings.camera_id,
                "enabled during motion" if inference else "disabled")
    logger.info("Events start at detected motion and close after %.1fs without motion",
                settings.no_motion_timeout)
    if settings.show_preview:
        logger.info("Showing live camera preview; press q in the preview window or Ctrl+C to stop")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                logger.warning("Camera read failed; attempting restart")
                capture.reconnect()
                continue

            if settings.resize_width and frame.shape[1] > settings.resize_width:
                height = round(frame.shape[0] * settings.resize_width / frame.shape[1])
                frame = cv2.resize(frame, (settings.resize_width, height))

            motion, boxes = detector.detect(frame)
            now = time.monotonic()

            if future is not None and future.done():
                try:
                    result = future.result()
                    if manager.recording and manager.event_token == future_event:
                        manager.add_inference(result, datetime.now(timezone.utc))
                        anomaly = result.get("label")
                        score = result.get("anomaly_probability")
                        logger.info("Inference complete: anomaly=%s score=%s, YOLO detections=%d",
                                    anomaly if anomaly is not None else "disabled",
                                    f"{score:.3f}" if score is not None else "n/a",
                                    len(result.get("detections", [])))
                except Exception:
                    logger.exception("Inference failed for a sampled event frame")
                finally:
                    future = None
                    future_event = None

            # Keep the event open until its outstanding model result is attached.
            manager.update(frame, motion, [], defer_completion_seconds=22.0 if future is not None else 0.0)

            if (inference and executor and manager.recording and future is None
                    and now - last_inference >= settings.inference_interval):
                future = executor.submit(inference.predict, frame.copy())
                future_event = manager.event_token
                last_inference = now

            if settings.show_preview:
                preview = frame.copy()
                for x, y, width, height in boxes:
                    cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 220, 0), 2)
                status = "RECORDING" if manager.recording else ("MOTION" if motion else "IDLE")
                color = (0, 0, 255) if manager.recording else (0, 220, 0)
                cv2.putText(preview, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if inference and manager.recording:
                    anomaly = manager.latest_anomaly
                    status_text = "ML: warming up" if anomaly is None else (
                        f"ML: {anomaly['label']} ({anomaly['anomaly_probability']:.2f})"
                    )
                    cv2.putText(preview, status_text, (12, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (255, 200, 0), 2)
                    if anomaly is not None:
                        logger.info("Anomaly score %.3f -> %s (threshold %.2f)",
                                    anomaly["anomaly_probability"], anomaly["label"], anomaly["threshold"])
                cv2.imshow("Edge AI Camera - Raspberry Pi", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        logger.info("Stopping on keyboard interrupt")
    finally:
        manager.close()
        capture.release()
        uploader.stop()
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
        if inference:
            inference.close()
        if settings.show_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
