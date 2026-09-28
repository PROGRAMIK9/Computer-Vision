# Edge AI Camera Monitoring

MVP for monitoring a laptop webcam or an Android phone camera. The laptop is the **edge processor**: it receives frames, detects motion, keeps a short in-memory frame buffer, runs the existing models only during motion, and saves only event video. FastAPI stores camera and event records in SQLite and receives completed event metadata/video. A React dashboard registers cameras and displays events.

## System flow

```text
Android phone camera app (local MJPEG over Wi-Fi)
        │ frames stay on the local network
        ▼
Laptop edge process
  frame buffer → motion detector → event recording
                   └─ during motion: MobileNet anomaly score + YOLO objects
        │ completed metadata + event video only
        ▼
FastAPI → SQLite + stored event video → React dashboard
```

The phone is the camera, not the inference device. The laptop performs edge processing. The backend does not receive the continuous stream. If the backend is unavailable, event uploads wait in `storage/upload_queue/` and retry in the background.

## 1. Install and configure

Use Python 3.11 or 3.12 for the broadest compatibility with TensorFlow/model dependencies. From a terminal:

```bash
cd /home/aaryamann/Projects/Computer-Vision/edge-ai-camera
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env` only if it does not exist:

```bash
test -f .env || cp .env.example .env
```

The default `CAMERA_SOURCE=webcam` tests the laptop camera. Inference defaults to disabled so the first camera/backend setup stays lightweight. The local ML files are not copied into this folder.

## 2. Start the FastAPI backend

In terminal 1:

```bash
cd /home/aaryamann/Projects/Computer-Vision/edge-ai-camera
source .venv/bin/activate
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Open the React dashboard in terminal 2 (next section), or inspect the API at `http://127.0.0.1:8000/docs`. The backend creates `storage/backend.db` automatically.

## 3. Start the React dashboard

In terminal 2:

```bash
cd /home/aaryamann/Projects/Computer-Vision/edge-ai-camera/frontend
npm install
npm run dev
```

Open the URL Vite prints, usually `http://localhost:5173`. Use **Register a phone camera** to choose a camera ID and display name. The ID must contain only letters, numbers, `_`, or `-`; save the ID for the edge configuration. The dashboard refreshes camera/event data every four seconds.

To serve a built dashboard from FastAPI instead:

```bash
cd /home/aaryamann/Projects/Computer-Vision/edge-ai-camera/frontend
npm run build
```

Then open `http://127.0.0.1:8000/` (restart FastAPI if it was already running).

## 4. Run the laptop webcam edge processor

In terminal 3, leave these settings in `.env`:

```env
CAMERA_SOURCE=webcam
CAMERA_INDEX=0
INFERENCE_ENABLED=false
```

Run:

```bash
cd /home/aaryamann/Projects/Computer-Vision/edge-ai-camera
source .venv/bin/activate
python -m edge.main
```

Move in front of the camera. After motion stops for `NO_MOTION_TIMEOUT` seconds (3 seconds by default), an event folder is created under `storage/events/`. The uploader sends its metadata and video to FastAPI in the background. Press `q` to stop; `Ctrl+C` also works.

## 5. Connect an Android phone camera

The MVP expects an MJPEG camera endpoint. Install an Android IP-camera app that provides an MJPEG `/video` URL (for example, IP Webcam). Keep the phone and laptop on the same trusted Wi-Fi; guest Wi-Fi/client isolation can prevent them from reaching each other.

1. Start the camera server in the phone app and note its LAN address and stream port. The app usually displays a URL similar to `http://192.168.1.50:8080/video`.
2. Register the phone in the React dashboard. For example, use camera ID `phone-01` and name `Front door phone`.
3. On the laptop, edit `edge-ai-camera/.env`:

   ```env
   CAMERA_SOURCE=phone
   CAMERA_ID=phone-01
   CAMERA_NAME=Front door phone
   PHONE_STREAM_URL=http://192.168.1.50:8080/video
   FPS=15
   SHOW_PREVIEW=true
   ```

   Replace the example phone IP/port with the address shown in the Android camera app. `CAMERA_ID` must exactly match the ID registered on the website. Set `FPS` near the stream rate shown in the app so the five-second circular buffer represents about five seconds of footage.

4. Keep the phone app streaming, then run `python -m edge.main` from the project directory. The laptop opens the MJPEG URL, performs motion detection, and writes event video locally. The edge process sends completed events to FastAPI; the dashboard will show them after upload.

The phone side currently uses an existing Android IP-camera app; this repository does not yet build a custom Android APK. The phone app may expose its stream to any device on the same Wi-Fi. This MVP has no pairing/authentication yet; use a trusted network and do not port-forward the phone camera or backend to the public internet.

## 6. Enable the existing models

The models are the same assets used by `v1.1_demo`:

```text
../tested_models/7_train_7_finetune/model_final.keras
../weights/yolo26s.pt
```

MobileNetV3Small outputs an `ANOMALY` probability (0.5 threshold); YOLO26s reports object labels, confidence, and bounding boxes. Both run only after motion starts, by default at up to five samples per second.

Install model requirements separately after webcam/phone streaming works:

```bash
cd /home/aaryamann/Projects/Computer-Vision/edge-ai-camera
source .venv/bin/activate
pip install -r requirements-ml.txt
```

TensorFlow and PyTorch can download large packages; PyTorch may select CUDA/NVIDIA dependencies depending on Python/platform/package index. Then set in `.env`:

```env
INFERENCE_ENABLED=true
INFERENCE_INTERVAL=0.2
ANOMALY_THRESHOLD=0.5
YOLO_CONFIDENCE=0.35
ANOMALY_MODEL_PATH=../tested_models/7_train_7_finetune/model_final.keras
YOLO_MODEL_PATH=../weights/yolo26s.pt
```

Restart the edge process. Inference results and YOLO detections are included in event metadata and displayed in the dashboard. If the models are too slow for the laptop, increase `INFERENCE_INTERVAL` (for example, `0.5` for 2 samples/sec).

## API overview

- `GET /api/health`
- `POST /api/cameras` — register camera (`id`, `name`, optional `device_id`)
- `GET /api/cameras`, `GET /api/cameras/{camera_id}`
- `POST /api/cameras/{camera_id}/heartbeat`
- `POST /api/events` — edge uploads completed event JSON
- `GET /api/events`, `GET /api/events/{event_id}`
- `PUT /api/events/{event_id}/video`, `GET /api/events/{event_id}/video`

SQLite stores camera, event, and detection rows. Event videos are saved under `storage/backend_media/`. Edge copies remain under `storage/events/`.

## Configuration reference

| Variable | Default | Description |
| --- | --- | --- |
| `CAMERA_SOURCE` | `webcam` | `webcam` or `phone` |
| `CAMERA_INDEX` | `0` | Laptop webcam index |
| `CAMERA_ID` | `webcam-01` | Must match a dashboard registration for phone events |
| `CAMERA_NAME` | `Laptop webcam` | Display name attached to events |
| `PHONE_STREAM_URL` | example URL | Android app's local MJPEG `/video` URL |
| `FPS` | `30` | Webcam FPS hint and circular buffer sizing |
| `BUFFER_SECONDS` | `5` | Pre-motion frame retention |
| `MOTION_THRESHOLD` | `25` | Pixel-difference threshold |
| `MIN_CONTOUR_AREA` | `800` | Minimum changed region area |
| `BLUR_SIZE` | `21` | Blur kernel, rounded up to odd |
| `NO_MOTION_TIMEOUT` | `3.0` | Quiet time before event closes |
| `RESIZE_WIDTH` | `640` | Edge processing width; 0 disables resize |
| `SHOW_PREVIEW` | `true` | Local OpenCV preview window |
| `INFERENCE_ENABLED` | `false` | Enable existing TensorFlow + YOLO models |
| `INFERENCE_INTERVAL` | `0.2` | Minimum seconds between model samples |
| `BACKEND_URL` | `http://127.0.0.1:8000` | FastAPI URL receiving completed events |
| `BACKEND_UPLOAD_ENABLED` | `true` | Enable background event upload/retry |
| `UPLOAD_QUEUE_DIR` | `storage/upload_queue` | Durable pending-upload queue |
| `BACKEND_DB_PATH` | `storage/backend.db` | SQLite database file |
| `BACKEND_MEDIA_DIR` | `storage/backend_media` | Backend copy of event video |

## Project layout

```text
edge-ai-camera/
├── edge/                 # camera receiver, motion, buffer, inference, storage, uploader
├── backend/app/          # FastAPI + SQLite API
├── frontend/             # React + Vite dashboard
├── storage/              # local events, upload queue, SQLite, backend media
├── requirements.txt      # small edge/backend install
└── requirements-ml.txt   # optional TensorFlow + Ultralytics
```
