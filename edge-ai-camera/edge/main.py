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
from edge.preview.server import PreviewServer
from edge.storage.event_storage import EventStorage


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def orient_phone_frame(frame):
    """Apply configured correction when a phone stream has a fixed rotation/flip."""
    if settings.camera_source != "phone":
        return frame
    rotations = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    rotation = settings.phone_rotation % 360
    if rotation not in rotations:
        raise ValueError("PHONE_ROTATION must be 0, 90, 180, or 270")
    if rotations[rotation] is not None:
        frame = cv2.rotate(frame, rotations[rotation])
    flips = {"none": None, "horizontal": 1, "vertical": 0, "both": -1}
    if settings.phone_flip not in flips:
        raise ValueError("PHONE_FLIP must be none, horizontal, vertical, or both")
    if flips[settings.phone_flip] is not None:
        frame = cv2.flip(frame, flips[settings.phone_flip])
    return frame


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
    inference_executor = None
    inference_load_future: Future | None = None
    inference_ready = False
    if settings.inference_enabled:
        inference = ModelInference(
            settings.resolve_project_path(settings.anomaly_model_path),
            settings.resolve_project_path(settings.yolo_model_path),
            settings.anomaly_threshold,
            settings.yolo_confidence,
            settings.anomaly_inference_enabled,
            settings.yolo_inference_enabled,
        )
        # Load model runtimes off the camera thread so motion capture remains live.
        inference_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="model-inference")
        inference_load_future = inference_executor.submit(inference.load)
        logger.info("Model workers loading in background; camera capture will continue")
    last_inference = 0.0
    inference_future: Future | None = None
    inference_event_token: str | None = None
    latest_inference_result: dict | None = None
    preview_server = PreviewServer(settings.preview_host, settings.preview_port) if settings.preview_server_enabled else None
    if preview_server:
        try:
            preview_server.start()
        except OSError:
            logger.exception("Could not start browser preview server; camera recording will continue")
            preview_server = None
    show_local_preview = settings.show_preview
    logger.info("Camera source=%s; keeping up to %d frames in memory", settings.camera_source,
                round(settings.fps * settings.buffer_seconds))
    if settings.camera_source == "phone":
        logger.info("Phone MJPEG source: %s", settings.phone_stream_url)
    logger.info("Press q in the local preview window or Ctrl+C to stop")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                logger.warning("Camera frame read failed; attempting source reconnect")
                capture.reconnect(delay=1.0)
                continue

            frame = orient_phone_frame(frame)

            if settings.resize_width and frame.shape[1] > settings.resize_width:
                height = round(frame.shape[0] * settings.resize_width / frame.shape[1])
                frame = cv2.resize(frame, (settings.resize_width, height))

            motion, boxes = detector.detect(frame)
            # Include this frame in the rolling window before taking the pre-roll snapshot.
            frame_buffer.append(frame)

            # Only infer during an active motion event and at a bounded rate.
            now = time.monotonic()
            if inference_load_future is not None and inference_load_future.done():
                try:
                    inference_load_future.result()
                    inference_ready = True
                    logger.info("All enabled model workers are ready")
                except Exception:
                    logger.exception("Model worker startup failed; camera recording will continue without inference")
                finally:
                    inference_load_future = None

            if inference_future is not None and inference_future.done():
                try:
                    result = inference_future.result()
                    if manager.recording and manager.event_token == inference_event_token:
                        latest_inference_result = result
                        manager.add_inference(result, datetime.now(timezone.utc))
                        for error in result.get("errors", []):
                            logger.error("Model worker error: %s", error)
                        anomaly = result.get("label")
                        score = result.get("anomaly_probability")
                        logger.info("Inference complete: anomaly=%s score=%s, YOLO detections=%d",
                                    anomaly if anomaly is not None else "disabled",
                                    f"{score:.3f}" if score is not None else "n/a",
                                    len(result.get("detections", [])))
                except Exception:
                    logger.exception("Inference failed for sampled event frame")
                finally:
                    inference_future = None
                    inference_event_token = None

            # Keep the event open until its outstanding model result is attached.
            manager.update(frame, motion, frame_buffer.snapshot(),
                           defer_completion_seconds=22.0 if inference_future is not None else 0.0)

            if (inference is not None and inference_executor is not None and inference_ready and manager.recording
                    and inference_future is None and now - last_inference >= settings.inference_interval):
                inference_future = inference_executor.submit(inference.predict, frame.copy())
                inference_event_token = manager.event_token
                last_inference = now

            if show_local_preview or preview_server is not None:
                preview = frame.copy()
                for x, y, width, height in boxes:
                    cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 220, 0), 2)
                if latest_inference_result is not None:
                    for detection in latest_inference_result.get("detections", []):
                        x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
                        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 120, 0), 2)
                        cv2.putText(preview, f"{detection['class_name']} {detection['confidence']:.2f}",
                                    (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 120, 0), 2)
                status = "RECORDING" if manager.recording else ("MOTION" if motion else "IDLE")
                cv2.putText(preview, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255) if manager.recording else (0, 220, 0), 2)
                if inference is not None:
                    anomaly = latest_inference_result
                    status_text = "ML: loading" if not inference_ready else (
                        "ML: waiting for motion" if anomaly is None else
                        (f"ML: {anomaly['label']} ({anomaly['anomaly_probability']:.2f})"
                         if "anomaly_probability" in anomaly else "ML: anomaly model unavailable")
                    )
                    cv2.putText(preview, status_text, (12, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (255, 200, 0), 2)
                if preview_server is not None:
                    preview_server.update(preview)
                if show_local_preview:
                    try:
                        cv2.imshow("Edge AI Camera - Phase 1", preview)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except cv2.error:
                        logger.exception("OpenCV GUI unavailable; disabling local window preview")
                        show_local_preview = False
    except KeyboardInterrupt:
        logger.info("Stopping on keyboard interrupt")
    finally:
        manager.close()
        capture.release()
        uploader.stop()
        if inference_executor is not None:
            inference_executor.shutdown(wait=True, cancel_futures=True)
        if inference is not None:
            inference.close()
        if preview_server is not None:
            preview_server.stop()
        if show_local_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
