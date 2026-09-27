# V1.1 Object Detection + Anomaly Detection Demo

Web-based demo for the V1.1 computer-vision pipeline.

V1.1 extends the existing V1 frame-level anomaly detection model by adding **YOLO26s object detection**. Both models process the same video frame independently.

```text
                     Original Video Frame
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
             YOLO26s                 MobileNetV3Small
          Object Detection            Anomaly Detection
                │                           │
                ▼                           ▼
        Objects + Bounding Box       Anomaly Probability
                │                           │
                └─────────────┬─────────────┘
                              ▼
                       Combined V1.1 Result
```

The existing V1 MobileNetV3Small model is **not modified or retrained**.

YOLO26s is used only for object detection, while MobileNetV3Small continues to perform the frame-level **NORMAL / ANOMALY** classification.

---

## 1. How It Works

For each selected frame:

1. The original frame is loaded from the UCF-Crime test dataset.
2. The original-resolution frame is passed to **YOLO26s** for object detection.
3. YOLO returns detected objects, confidence scores, and bounding boxes.
4. A separate copy of the frame is resized to **224 × 224**.
5. The resized frame is passed to the existing **MobileNetV3Small** model.
6. The model produces an anomaly probability.
7. The probability is compared against the `0.5` threshold.
8. The frontend displays the object detections and anomaly result together.

The two models operate independently. Detecting an object does **not** automatically mean that the frame is anomalous.

---

## 2. Dataset and Models

The demo uses the following existing project files.

### Dataset

Test frames:

```text
data/ucf_crime/Test/
```

The test dataset contains the 14 UCF-Crime classes:

```text
Test/
├── Abuse/
├── Arrest/
├── Arson/
├── Assault/
├── Burglary/
├── Explosion/
├── Fighting/
├── NormalVideos/
├── RoadAccidents/
├── Robbery/
├── Shooting/
├── Shoplifting/
├── Stealing/
└── Vandalism/
```

The demo reads the extracted frames directly from this directory.

### V1 Anomaly Model

```text
tested_models/7_train_7_finetune/model_final.keras
```

Model:

```text
MobileNetV3Small
Input:     224 × 224 × 3
Output:    Anomaly probability
Threshold: 0.5
```

### YOLO Object Detection Model

```text
weights/yolo26s.pt
```

Model:

```text
YOLO26s
Task:       Object Detection
Classes:    80 COCO classes
Confidence: 0.35
```

---

## 3. Project Structure

The relevant V1.1 files are:

```text
Computer-Vision/
├── data/
│   └── ucf_crime/
│       └── Test/
│           ├── Abuse/
│           ├── Arrest/
│           ├── ...
│           └── Vandalism/
│
├── tested_models/
│   └── 7_train_7_finetune/
│       └── model_final.keras
│
├── weights/
│   └── yolo26s.pt
│
└── v1.1_demo/
    ├── backend/
    │   └── main.py
    └── frontend/
        ├── app.js
        ├── index.html
        └── style.css
```

Before running the demo, verify the required files:

```bash
ls data/ucf_crime/Test
ls -lh tested_models/7_train_7_finetune/model_final.keras
ls -lh weights/yolo26s.pt
```

---

## 4. Requirements

The demo uses:

- Python
- FastAPI
- Uvicorn
- TensorFlow
- Ultralytics
- NumPy
- Pillow

The project dependencies are listed in:

```text
model/requirements.txt
```

Install them with:

```bash
pip install -r model/requirements.txt
```

---

## 5. Run the Demo

From the project directory:

```bash
cd ~/Dev/Computer-Vision
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Install dependencies if required:

```bash
pip install -r model/requirements.txt
```

Go to the V1.1 backend:

```bash
cd v1.1_demo/backend
```

Start the server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

For development with automatic reload:

```bash
uvicorn main:app --reload
```

---

## 6. Open the Demo

Once the server is running, open:

```text
http://127.0.0.1:8000
```

The V1.1 demo provides:

- UCF-Crime class selection
- Video selection
- Frame-by-frame playback
- Frame timeline
- YOLO bounding boxes
- Detected object labels and confidence
- Anomaly probability
- Normal probability
- NORMAL / ANOMALY decision

The YOLO detection area has a fixed layout so that the interface does not shift when the number of detected objects changes.

---

## 7. Model Behavior

### MobileNetV3Small

The V1 anomaly model evaluates each frame independently.

```text
Probability > 0.5  → ANOMALY
Probability ≤ 0.5  → NORMAL
```

No previous or future frames are provided to the model.

### YOLO26s

YOLO26s receives the **original-resolution frame** and detects objects using the COCO object-detection classes.

Examples include:

```text
person
car
bicycle
motorcycle
bus
truck
dog
cat
chair
laptop
cell phone
backpack
bottle
```

YOLO is an **object detector**, not an activity classifier. It does not directly classify UCF-Crime activities such as Assault, Burglary, Fighting, Robbery, Shooting, Stealing, or Vandalism.

---

## 8. V1.1 Inference Pipeline

```text
Original Frame
      │
      ├──────────────────────┐
      │                      │
      ▼                      ▼
   YOLO26s             Resize → 224 × 224
      │                      │
      ▼                      ▼
Object Detection      MobileNetV3Small
      │                      │
      ▼                      ▼
Bounding Boxes         Anomaly Probability
      │                      │
      │                 Threshold = 0.5
      │                      │
      └──────────┬───────────┘
                 ▼
          Combined V1.1 Output
```

V1.1 therefore adds object-level visual information to the existing V1 anomaly-detection result without changing the original anomaly model.
