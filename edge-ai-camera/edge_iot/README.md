# Raspberry Pi Edge IoT

This folder contains a Raspberry Pi camera runner that uses the existing event API and dashboard. Each Pi processes its own camera locally. The backend receives a video and event metadata after motion has stopped; it does not receive a continuous camera stream.

## Event behavior

```text
Pi camera frames → visual motion detector → event starts
                                      └── sample frames for inference while motion event is active
Motion stops → wait NO_MOTION_TIMEOUT → save event video + metadata → upload to dashboard
```

The current detector is frame-difference motion detection. Inference runs on the Pi at `INFERENCE_INTERVAL` while an event is active. The event video retains frames from the motion period and the configured quiet timeout. There is no OLED or GPIO sensor dependency.

## Raspberry Pi setup

Use Raspberry Pi OS 64-bit on a Pi with a supported camera module. Connect and enable the camera, then verify it with `rpicam-hello`.

Install the system camera and computer-vision libraries:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-numpy python3-venv git
```

Clone this repository on the Pi, then create a virtual environment that can see the Raspberry Pi OS Picamera2 and OpenCV packages:

```bash
cd Computer-Vision/edge-ai-camera
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-iot.txt
```

Do not install `requirements.txt` wholesale in this environment: it also installs server packages and pip OpenCV, while the Pi runner uses the OS Picamera2/OpenCV packages. Picamera2 itself comes from `apt`.

## Configure both Pis

Run the backend and dashboard on a separate computer on the same trusted LAN. If the backend computer's LAN address is `192.168.1.20`, the Pi must be able to reach `http://192.168.1.20:8000`.

In `edge-ai-camera/.env` on each Pi, set a different ID and display name:

```env
CAMERA_ID=pi-camera-01
CAMERA_NAME=Entrance camera
BACKEND_URL=http://192.168.1.20:8000
BACKEND_UPLOAD_ENABLED=true
FPS=15
PI_CAMERA_WIDTH=640
PI_CAMERA_HEIGHT=480
RESIZE_WIDTH=640
MOTION_THRESHOLD=25
MIN_CONTOUR_AREA=800
NO_MOTION_TIMEOUT=3.0
SHOW_PREVIEW=false
INFERENCE_ENABLED=false
```

Use `pi-camera-02` and a unique name on the second Pi. `SHOW_PREVIEW=false` is appropriate for a headless Pi. If `.env` does not exist, copy `.env.example` and then edit it. The backend URL must use the backend machine's LAN IP, not `127.0.0.1`.

Register the same camera IDs in the dashboard, or let the first event create camera entries automatically. The dashboard displays completed events and supports viewing their uploaded videos.

## Run

From `edge-ai-camera/` with the virtual environment active:

```bash
python -m edge_iot.main
```

Press `Ctrl+C` to stop cleanly. The current event is saved on shutdown. Events are saved under `storage/events/`. If the backend is unavailable, metadata and video remain queued locally and are retried when connectivity returns.

## Enable the existing models on the Pi

The runner calls the existing anomaly classifier and YOLO detector only during an active event. Set `INFERENCE_ENABLED=true`, install the optional packages from `requirements-ml.txt` only after checking compatibility with your Pi OS/Python version, and configure model paths:

```env
INFERENCE_ENABLED=true
INFERENCE_INTERVAL=0.5
ANOMALY_MODEL_PATH=../tested_models/7_train_7_finetune/model_final.keras
YOLO_MODEL_PATH=../weights/yolo26n.pt
```

Use `yolo26n.pt` as the smaller detector. The model files must be present at those paths on each Pi. TensorFlow and PyTorch support depends on the Pi model, OS, and Python version, and the supplied models may be too slow or memory-intensive on some Raspberry Pi hardware. If inference dependencies or models are not installed, leave inference disabled: motion event recording and dashboard uploads still work.

`INFERENCE_INTERVAL=0.5` samples up to two frames per second during motion. Increase it to reduce CPU load. A single inference worker prevents model calls from piling up if one sample takes longer than the interval.

## Dashboard on the LAN

On the backend computer, follow the main README to start FastAPI and build/serve the frontend. For a same-origin dashboard accessible on your LAN, build the frontend and run FastAPI on all interfaces:

```bash
cd edge-ai-camera/frontend
npm install
npm run build
cd ..
source .venv/bin/activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Open `http://192.168.1.20:8000/` from a LAN browser (replace the IP). Ensure the backend machine's firewall allows port 8000 on the local network. Avoid exposing the camera API directly to the public internet.

## Current limitations

- Motion start/end comes from image changes, not a PIR sensor.
- Inference summaries and object detections are attached to the event; the event video itself is not annotated with model overlays.
- The backend/dashboard must be reachable on the LAN for uploads; offline uploads are retried from the Pi's local queue.
