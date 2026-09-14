import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "tested_models"
    / "7_train_7_finetune"
    / "model_final.keras"
)

TEST_DIR = PROJECT_ROOT / "data" / "Test"

FRONTEND_DIR = PROJECT_ROOT / "v1_demo" / "frontend"


# ============================================================
# V1 configuration
# ============================================================

IMG_SIZE = 224
THRESHOLD = 0.5

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
# Model
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

print()
print("Loading V1 model...")
print(MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

print("V1 model loaded.")
print(f"Input shape:  {model.input_shape}")
print(f"Output shape: {model.output_shape}")
print()


# ============================================================
# Dataset indexing
# ============================================================

# Cache the discovered structure in memory.
#
# {
#     "Assault": {
#         "Assault006_x264": [
#             {
#                 "path": "...",
#                 "source_frame": 0
#             },
#             ...
#         ]
#     }
# }

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

        # Numerical ordering by actual source frame.
        for video_id in videos:

            videos[video_id].sort(
                key=lambda item: item["source_frame"]
            )

        VIDEO_INDEX[class_name] = videos

        print(
            f"  {class_name}: "
            f"{len(videos)} videos"
        )

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
    title="V1 Frame-by-Frame Demo",
    version="1.0",
)


app.mount(
    "/static",
    StaticFiles(
        directory=FRONTEND_DIR
    ),
    name="static",
)


@app.get("/")
def root():

    return FileResponse(
        FRONTEND_DIR / "index.html"
    )


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

@app.get(
    "/api/image/{class_name}/{video_id}/{frame_index}"
)
def get_image(
    class_name: str,
    video_id: str,
    frame_index: int,
):

    try:
        frame = VIDEO_INDEX[
            class_name
        ][video_id][frame_index]

    except (
        KeyError,
        IndexError,
    ):
        raise HTTPException(
            status_code=404,
            detail="Frame not found",
        )

    return FileResponse(
        frame["path"]
    )


# ============================================================
# Prediction API
# ============================================================

@app.get(
    "/api/predict/{class_name}/{video_id}/{frame_index}"
)
def predict_frame(
    class_name: str,
    video_id: str,
    frame_index: int,
):

    try:
        frame = VIDEO_INDEX[
            class_name
        ][video_id][frame_index]

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

    # Don't run TensorFlow again if we already
    # evaluated this frame.
    if cache_key in prediction_cache:

        result = prediction_cache[
            cache_key
        ].copy()

        result["cached"] = True

        return result

    try:

        image = Image.open(
            frame["path"]
        ).convert("RGB")

        # Same image resizing behavior as
        # image_dataset_from_directory.
        image = image.resize(
            (IMG_SIZE, IMG_SIZE),
            Image.Resampling.BILINEAR,
        )

        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

        # Same shape supplied to model.predict().
        batch = np.expand_dims(
            image_array,
            axis=0,
        )

        # IMPORTANT:
        #
        # Do NOT call mobilenet_v3.preprocess_input()
        # here.
        #
        # The V1 model itself contains:
        #
        #     mobilenet_v3.preprocess_input()
        #
        # before MobileNetV3Small.
        prediction = model.predict(
            batch,
            verbose=0,
        )

        anomaly_probability = float(
            prediction[0][0]
        )

        normal_probability = (
            1.0 -
            anomaly_probability
        )

        label = (
            "ANOMALY"
            if anomaly_probability > THRESHOLD
            else "NORMAL"
        )

        result = {
            "class_name": class_name,
            "video_id": video_id,

            "frame_index": frame_index,
            "frame_number": frame_index + 1,

            "source_frame": frame[
                "source_frame"
            ],

            "filename": frame[
                "path"
            ].name,

            "prediction": anomaly_probability,

            "anomaly_probability":
                anomaly_probability,

            "normal_probability":
                normal_probability,

            "threshold": THRESHOLD,

            "label": label,

            "cached": False,
        }

        prediction_cache[
            cache_key
        ] = result

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
        "model": "MobileNetV3Small",
        "version": "V1",
        "image_size": IMG_SIZE,
        "threshold": THRESHOLD,
        "output": "Binary anomaly probability",
        "architecture": [
            "MobileNetV3Small",
            "GlobalAveragePooling2D",
            "Dense(128, ReLU)",
            "Dropout(0.2)",
            "Dense(1, sigmoid)",
        ],
    }