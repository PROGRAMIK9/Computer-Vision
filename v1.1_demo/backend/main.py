import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Existing V1 anomaly model
MODEL_PATH = PROJECT_ROOT / "tested_models" / "7_train_7_finetune" / "model_final.keras"

# YOLO object detector
YOLO_MODEL_PATH = PROJECT_ROOT / "weights" / "yolo26s.pt"

# V1 test frames
TEST_DIR = PROJECT_ROOT / "data" / "ucf_crime" / "Test"

FRONTEND_DIR = PROJECT_ROOT / "v1.1_demo" / "frontend"


# ============================================================
# V1 configuration
# ============================================================

IMG_SIZE = 224
THRESHOLD = 0.5

# YOLO confidence threshold
YOLO_CONFIDENCE = 0.35

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
}


# ============================================================
# GPU
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print(f"GPU detected: {len(gpus)}")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(
                gpu,
                True,
            )
        except RuntimeError:
            pass

        print(f"  {gpu}")

else:
    print("No GPU detected. Using CPU.")


# ============================================================
# Load V1 anomaly model
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"V1 model not found: {MODEL_PATH}")

print()
print("Loading V1 anomaly model...")
print(MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

print("V1 model loaded.")
print(f"Input shape:  {model.input_shape}")
print(f"Output shape: {model.output_shape}")


# ============================================================
# Load YOLO
# ============================================================

if not YOLO_MODEL_PATH.exists():
    raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL_PATH}")

print()
print("Loading YOLO object detector...")
print(YOLO_MODEL_PATH)

yolo_model = YOLO(str(YOLO_MODEL_PATH))

print("YOLO loaded.")
print(f"YOLO task: {yolo_model.task}")
print(f"YOLO classes: {len(yolo_model.names)}")

print()


# ============================================================
# Dataset indexing
# ============================================================

VIDEO_INDEX = {}


def parse_filename(filename):
    """
    Extract:

        video_id
        source frame number

    Example:

        Assault006_x264_120.png

    becomes:

        ("Assault006_x264", 120)
    """

    match = re.match(
        r"^(.+)_(\d+)\.(png|jpg|jpeg|bmp|webp)$",
        filename,
        re.IGNORECASE,
    )

    if not match:
        return None

    video_id = match.group(1)

    source_frame = int(match.group(2))

    return video_id, source_frame


def build_video_index():

    print("Indexing test dataset...")

    for class_dir in sorted(TEST_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        videos = {}

        for path in class_dir.iterdir():
            if not path.is_file():
                continue

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            parsed = parse_filename(path.name)

            if parsed is None:
                continue

            video_id, source_frame = parsed

            videos.setdefault(
                video_id,
                [],
            ).append(
                {
                    "path": path,
                    "source_frame": source_frame,
                }
            )

        for video_id in videos:
            videos[video_id].sort(key=lambda item: item["source_frame"])

        VIDEO_INDEX[class_name] = videos

        print(f"  {class_name}: {len(videos)} videos")

    print("Dataset indexing complete.")
    print()


build_video_index()


# ============================================================
# Prediction cache
# ============================================================

prediction_cache = {}


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="V1.1 Object Detection + Anomaly Demo",
    version="1.1",
)


app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static",
)


@app.get("/")
def root():

    return FileResponse(FRONTEND_DIR / "index.html")


# ============================================================
# Dataset APIs
# ============================================================


@app.get("/api/classes")
def get_classes():

    result = []

    for class_name in sorted(VIDEO_INDEX):
        videos = VIDEO_INDEX[class_name]

        result.append(
            {
                "name": class_name,
                "video_count": len(videos),
            }
        )

    return result


@app.get("/api/videos/{class_name}")
def get_videos(class_name: str):

    if class_name not in VIDEO_INDEX:
        raise HTTPException(
            status_code=404,
            detail="Class not found",
        )

    videos = VIDEO_INDEX[class_name]

    result = []

    for video_id in sorted(videos):
        frames = videos[video_id]

        result.append(
            {
                "id": video_id,
                "frame_count": len(frames),
            }
        )

    return result


@app.get("/api/video/{class_name}/{video_id}")
def get_video(
    class_name: str,
    video_id: str,
):

    if class_name not in VIDEO_INDEX:
        raise HTTPException(
            status_code=404,
            detail="Class not found",
        )

    videos = VIDEO_INDEX[class_name]

    if video_id not in videos:
        raise HTTPException(
            status_code=404,
            detail="Video not found",
        )

    frames = videos[video_id]

    return {
        "class_name": class_name,
        "video_id": video_id,
        "total_frames": len(frames),
        "frames": [
            {
                "index": i,
                "source_frame": frame["source_frame"],
                "filename": frame["path"].name,
            }
            for i, frame in enumerate(frames)
        ],
    }


# ============================================================
# Image API
# ============================================================


@app.get("/api/image/{class_name}/{video_id}/{frame_index}")
def get_image(
    class_name: str,
    video_id: str,
    frame_index: int,
):

    try:
        frame = VIDEO_INDEX[class_name][video_id][frame_index]

    except (
        KeyError,
        IndexError,
    ):
        raise HTTPException(
            status_code=404,
            detail="Frame not found",
        )

    return FileResponse(frame["path"])


# ============================================================
# Prediction API
# ============================================================


@app.get("/api/predict/{class_name}/{video_id}/{frame_index}")
def predict_frame(
    class_name: str,
    video_id: str,
    frame_index: int,
):

    try:
        frame = VIDEO_INDEX[class_name][video_id][frame_index]

    except (
        KeyError,
        IndexError,
    ):
        raise HTTPException(
            status_code=404,
            detail="Frame not found",
        )

    cache_key = (
        class_name,
        video_id,
        frame_index,
    )

    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------

    if cache_key in prediction_cache:
        result = prediction_cache[cache_key].copy()

        result["cached"] = True

        return result

    try:
        # ====================================================
        # Load original image
        # ====================================================

        image = Image.open(frame["path"]).convert("RGB")

        # ====================================================
        # YOLO OBJECT DETECTION
        #
        # IMPORTANT:
        #
        # YOLO receives the ORIGINAL image.
        # We do not resize it to 224x224.
        # ====================================================

        yolo_results = yolo_model.predict(
            source=np.asarray(image),
            conf=YOLO_CONFIDENCE,
            verbose=False,
        )

        yolo_result = yolo_results[0]

        detections = []

        if yolo_result.boxes is not None:
            boxes = yolo_result.boxes

            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()

                confidence = float(boxes.conf[i].cpu().item())

                class_id = int(boxes.cls[i].cpu().item())

                class_name_yolo = yolo_model.names[class_id]

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name_yolo,
                        "confidence": confidence,
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                    }
                )

        # ====================================================
        # V1 ANOMALY MODEL
        #
        # Keep the exact V1 preprocessing.
        # ====================================================

        resized_image = image.resize(
            (IMG_SIZE, IMG_SIZE),
            Image.Resampling.BILINEAR,
        )

        image_array = np.asarray(
            resized_image,
            dtype=np.float32,
        )

        batch = np.expand_dims(
            image_array,
            axis=0,
        )

        prediction = model.predict(
            batch,
            verbose=0,
        )

        anomaly_probability = float(prediction[0][0])

        normal_probability = 1.0 - anomaly_probability

        label = "ANOMALY" if anomaly_probability > THRESHOLD else "NORMAL"

        # ====================================================
        # Combined result
        # ====================================================

        result = {
            "version": "V1.1",
            "class_name": class_name,
            "video_id": video_id,
            "frame_index": frame_index,
            "frame_number": frame_index + 1,
            "source_frame": frame["source_frame"],
            "filename": frame["path"].name,
            # --------------------------------------------
            # V1 anomaly output
            # --------------------------------------------
            "prediction": anomaly_probability,
            "anomaly_probability": anomaly_probability,
            "normal_probability": normal_probability,
            "threshold": THRESHOLD,
            "label": label,
            # --------------------------------------------
            # YOLO output
            # --------------------------------------------
            "object_count": len(detections),
            "detections": detections,
            "yolo_confidence": YOLO_CONFIDENCE,
            "cached": False,
        }

        prediction_cache[cache_key] = result

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# Model information
# ============================================================


@app.get("/api/info")
def get_info():

    return {
        "version": "V1.1",
        "anomaly_model": {
            "name": "MobileNetV3Small",
            "image_size": IMG_SIZE,
            "threshold": THRESHOLD,
        },
        "object_detector": {
            "name": "YOLO26n",
            "weights": str(YOLO_MODEL_PATH),
            "confidence": YOLO_CONFIDENCE,
            "classes": len(yolo_model.names),
        },
        "pipeline": [
            "Original frame",
            "YOLO object detection",
            "V1 MobileNetV3Small anomaly detection",
            "Combined result",
        ],
    }
