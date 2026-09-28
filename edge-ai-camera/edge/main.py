"""Phase 1: webcam -> in-memory pre-roll -> motion events saved as video."""

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone

import cv2

from edge.buffer.circular_buffer import CircularFrameBuffer
from edge.backend_client.uploader import EventUploader
from edge.camera.stream_receiver import StreamReceiver
from edge.config import settings
from edge.events.event_manager import EventManager
from edge.inference.model import ModelInference
from edge.motion.detector import MotionDetector
from edge.storage.event_storage import EventStorage


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    configure_logging()
    logger = logging.getLogger("edge")
    capture = StreamReceiver()
    frame_buffer = CircularFrameBuffer(round(settings.fps * settings.buffer_seconds))
    detector = MotionDetector(settings.motion_threshold, settings.min_contour_area, settings.blur_size)
    storage = EventStorage(settings.resolved_events_dir, settings.fps, settings.video_fourcc)
    uploader = EventUploader(settings.backend_url, settings.upload_queue_dir, settings.backend_upload_enabled)
    uploader.set_camera(settings.camera_id)
    uploader.start()
    manager = EventManager(storage, settings.camera_id, settings.camera_name,
                           settings.no_motion_timeout, on_complete=uploader.enqueue)
    inference = None
    if settings.inference_enabled:
        inference = ModelInference(
            settings.resolve_project_path(settings.anomaly_model_path),
            settings.resolve_project_path(settings.yolo_model_path),
            settings.anomaly_threshold,
            settings.yolo_confidence,
        )
        inference.load()
    last_inference = 0.0
    inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-inference") if inference else None
    inference_future: Future | None = None
    inference_event_token: str | None = None
    logger.info("Camera source=%s; keeping up to %d frames in memory", settings.camera_source,
                round(settings.fps * settings.buffer_seconds))
    if settings.camera_source == "phone":
        logger.info("Phone MJPEG source: %s", settings.phone_stream_url)
    logger.info("Press q in the preview window or Ctrl+C to stop")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                logger.warning("Camera frame read failed; attempting source reconnect")
                capture.reconnect(delay=1.0)
                continue

            if settings.resize_width and frame.shape[1] > settings.resize_width:
                height = round(frame.shape[0] * settings.resize_width / frame.shape[1])
                frame = cv2.resize(frame, (settings.resize_width, height))

            motion, boxes = detector.detect(frame)
            # Include this frame in the rolling window before taking the pre-roll snapshot.
            frame_buffer.append(frame)
            manager.update(frame, motion, frame_buffer.snapshot())

            # Only infer during an active motion event and at a bounded rate.
            now = time.monotonic()
            if inference_future is not None and inference_future.done():
                try:
                    result = inference_future.result()
                    if manager.recording and manager.event_token == inference_event_token:
                        manager.add_inference(result, datetime.now(timezone.utc))
                except Exception:
                    logger.exception("Inference failed for sampled event frame")
                finally:
                    inference_future = None
                    inference_event_token = None

            if (inference is not None and inference_executor is not None and manager.recording
                    and inference_future is None and now - last_inference >= settings.inference_interval):
                inference_future = inference_executor.submit(inference.predict, frame.copy())
                inference_event_token = manager.event_token
                last_inference = now

            if settings.show_preview:
                preview = frame.copy()
                for x, y, width, height in boxes:
                    cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 220, 0), 2)
                status = "RECORDING" if manager.recording else ("MOTION" if motion else "IDLE")
                cv2.putText(preview, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255) if manager.recording else (0, 220, 0), 2)
                if inference is not None and manager.recording:
                    anomaly = manager.latest_anomaly
                    status_text = "ML: warming up" if anomaly is None else (
                        f"ML: {anomaly['label']} ({anomaly['anomaly_probability']:.2f})"
                    )
                    cv2.putText(preview, status_text, (12, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (255, 200, 0), 2)
                cv2.imshow("Edge AI Camera - Phase 1", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        logger.info("Stopping on keyboard interrupt")
    finally:
        manager.close()
        capture.release()
        uploader.stop()
        if inference_executor is not None:
            inference_executor.shutdown(wait=False, cancel_futures=True)
        if settings.show_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
