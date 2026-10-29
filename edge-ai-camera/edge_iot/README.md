# Raspberry Pi 4 edge device setup

This guide deploys the existing edge camera application to a Raspberry Pi 4 with 8 GB RAM and a Logitech USB webcam. The Pi captures frames, detects motion, records event clips, and (when enabled) runs both existing ML models locally. It uploads completed events to the FastAPI backend; the backend does not process a live camera stream.

## What runs where

```text
Logitech USB camera → Raspberry Pi edge process → event video + metadata → backend/dashboard computer
                       motion + optional ML inference
```

Run these on the **Pi**:

- `python -m edge.main` — USB camera capture, motion detection, event recording, optional inference, and event upload.
- The `.keras` anomaly model and YOLO `.pt` model files — loaded into memory by `edge/inference/model.py` when `INFERENCE_ENABLED=true`.

Run these on a **separate computer on the same LAN** (recommended):

- FastAPI backend (`uvicorn backend.app.main:app ...`), SQLite storage, and the React dashboard (built and served by FastAPI).

You can run the backend on the Pi too, but then it shares CPU, memory, disk, and thermal headroom with capture and inference. The Pi does not need Node.js when using the built dashboard served from FastAPI.

## 1. Prepare Raspberry Pi OS

Use Raspberry Pi OS **64-bit** on the Pi 4. Connect the Logitech camera directly to a USB port, preferably a USB 3 port, and check that Linux sees it:

```bash
lsusb
ls -l /dev/video*
```

Install Git, Python, USB-video/OpenCV support, and build tools:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip python3-opencv \
  python3-numpy v4l-utils ffmpeg
```

Check the camera modes and capture a quick test frame (replace `/dev/video0` if needed):

```bash
v4l2-ctl --device=/dev/video0 --all
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

If the user cannot read `/dev/video0`, add the login to the `video` group and log out/in:

```bash
sudo usermod -aG video "$USER"
```

The code uses OpenCV `VideoCapture` and defaults to camera index 0 (`/dev/video0`). If multiple video devices exist, test indices 0, 1, etc. and set `CAMERA_INDEX` accordingly.

## 2. Get the code and create the Python environment

Clone the repository on the Pi (or copy the project there). From the repository root:

```bash
git clone <your-repository-url> Computer-Vision
cd Computer-Vision/edge-ai-camera
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-iot.txt
```

`--system-site-packages` lets this environment use Debian/Raspberry Pi OS OpenCV and NumPy while installing this app's small Python dependencies. This Pi camera app uses a USB camera, so it does **not** require Picamera2. Do not use `edge_iot.main` for the Logitech camera; that entry point is specifically for CSI Pi Camera modules.

## 3. Put the model files on the Pi

The defaults point outside `edge-ai-camera` to these files in the repository:

```text
Computer-Vision/tested_models/7_train_7_finetune/model_final.keras
Computer-Vision/weights/yolo26s.pt
```

Copy the repo's `tested_models/` and `weights/` folders to the Pi, preserving those paths. The Pi loads the models itself at startup; no inference is sent to the backend. The anomaly model uses TensorFlow/Keras; YOLO uses Ultralytics/PyTorch.

For a first camera-only run, skip the ML install and leave inference disabled. Model frameworks are large, and ARM64 wheel availability depends on your Raspberry Pi OS and Python version. If either framework cannot install on your image, the current Python runner cannot load that model as-is; use a compatible 64-bit OS/Python environment or export/convert and add a runtime such as TFLite/ONNX. Merely copying model files does not install the model runtime.

After confirming the OS/Python combination supports the project dependencies, install them in the same venv:

```bash
source .venv/bin/activate
python -m pip install -r requirements-ml.txt
```

Keep enough free disk for the packages and models. On a Pi 4, running TensorFlow and PyTorch inference together may be slow even with 8 GB RAM; try the smaller `yolo26n.pt`, lower inference frequency, and a 320–640 pixel processing width. The code schedules no more than one outstanding inference at a time and only samples during motion events.

## 4. Configure the Pi edge process

Create `edge-ai-camera/.env` if needed:

```bash
cd ~/Computer-Vision/edge-ai-camera
cp .env.example .env
nano .env
```

Set the following values (replace the backend address with the backend computer's LAN IP):

```env
CAMERA_SOURCE=webcam
CAMERA_INDEX=0
CAMERA_ID=pi-usb-01
CAMERA_NAME=Logitech entrance camera
BACKEND_URL=http://192.168.1.20:8000
BACKEND_UPLOAD_ENABLED=true

FPS=15
BUFFER_SECONDS=5
RESIZE_WIDTH=640
MOTION_THRESHOLD=25
MIN_CONTOUR_AREA=800
NO_MOTION_TIMEOUT=3.0
SHOW_PREVIEW=false

INFERENCE_ENABLED=false
INFERENCE_INTERVAL=0.5
ANOMALY_THRESHOLD=0.5
YOLO_CONFIDENCE=0.35
ANOMALY_MODEL_PATH=../tested_models/7_train_7_finetune/model_final.keras
YOLO_MODEL_PATH=../weights/yolo26n.pt
```

Keep `INFERENCE_ENABLED=false` until the camera-only flow works and you have installed compatible TensorFlow and Ultralytics dependencies. Then switch it to `true`. To use the `s` model instead, set `YOLO_MODEL_PATH=../weights/yolo26s.pt`.

The settings file is loaded by `edge/config.py`. Relative model paths are resolved from `edge-ai-camera`, with a fallback to the repository root, so these defaults work from the cloned repository layout.

## 5. Start the backend and dashboard on another computer

On a Linux backend computer on the same trusted LAN, install/run the backend according to [the main Edge AI Camera README](../README.md). In brief, build the frontend once and start FastAPI:

```bash
cd Computer-Vision/edge-ai-camera
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd frontend
npm ci
npm run build
cd ..
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Open `http://192.168.1.20:8000/` in a browser, replacing the example address. Allow TCP port 8000 through the backend machine's LAN firewall. The Pi's `BACKEND_URL` must point at this LAN address, never `127.0.0.1` (that would mean the Pi itself).

## 6. Run the Pi camera application

Connect the Logitech camera, then on the Pi:

```bash
cd ~/Computer-Vision/edge-ai-camera
source .venv/bin/activate
python -m edge.main
```

The app opens camera index 0, analyzes frames for motion, and retains the last five seconds in memory. When motion begins it writes an event video; after three quiet seconds it closes the event and queues its metadata/video for upload. Local event files are in `storage/events/`; uploads that fail are retained in `storage/upload_queue/` and retried. With inference enabled, both models are loaded once at process startup and predictions are sampled during active events. Press `Ctrl+C` to stop cleanly.

Check `http://<backend-ip>:8000/api/health` from the Pi if uploads are not reaching the dashboard. Confirm the camera index and that `BACKEND_URL` is reachable. A headless Pi should use `SHOW_PREVIEW=false`.

## Which code loads and runs the models?

The Pi entry point is:

```text
edge-ai-camera/edge/main.py
```

When `INFERENCE_ENABLED=true`, it creates `ModelInference` and calls `load()` before entering the camera loop. [The inference adapter](../edge/inference/model.py) loads the Keras model with `tf.keras.models.load_model(...)` and the YOLO model with `YOLO(...)`. During motion events, `predict()` sends sampled frames to both models. Results are added to event metadata and uploaded with the event. The default is disabled, so set it explicitly to true to run model inference on the Pi.

## Optional: start automatically after reboot

Once manual startup works, create a systemd service (replace `pi` and the path if your account differs):

```ini
[Unit]
Description=Edge AI USB Camera
After=network-online.target
Wants=network-online.target

[Service]
User=pi
WorkingDirectory=/home/pi/Computer-Vision/edge-ai-camera
ExecStart=/home/pi/Computer-Vision/edge-ai-camera/.venv/bin/python -m edge.main
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Save as `/etc/systemd/system/edge-camera.service`, then run:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now edge-camera
sudo systemctl status edge-camera
journalctl -u edge-camera -f
```

## Troubleshooting

- **`Could not open camera source 0`:** inspect `/dev/video*`, check `v4l2-ctl`, permissions/group membership, and try another `CAMERA_INDEX`.
- **No event video:** verify that motion fills enough pixels to exceed `MIN_CONTOUR_AREA`; lower it for a distant/small subject. Adjust `MOTION_THRESHOLD` for lighting changes.
- **No dashboard event:** check `BACKEND_URL`, backend health endpoint, LAN firewall, and local pending files in `storage/upload_queue/`.
- **Model file not found:** make sure the model folders are copied to the expected repository-relative paths or set absolute paths in `.env`.
- **Model packages fail to install or inference is too slow:** leave inference disabled for functional motion recording, then use a compatible ARM64 Python/runtime and benchmark one model at a time. 8 GB RAM does not add a GPU/NPU accelerator to the Pi 4.
- **Video playback fails:** the backend machine needs `ffmpeg`/`ffprobe` to prepare browser-compatible event video.

## Pi Camera Module alternative

For a CSI Pi Camera Module rather than a Logitech USB webcam, install Picamera2 from Raspberry Pi OS packages and run:

```bash
python -m edge_iot.main
```

That adapter is in `edge_iot/camera.py`; it is not used by the USB webcam setup above.
