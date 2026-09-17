Yes. For the **current V2 work**, I'd remove all the V1/demo material and keep the README focused on the new pipeline.

# Computer Vision — Anomaly Detection System

Computer-vision subsystem for a surveillance-based anomaly detection system.

The system is designed as a modular pipeline that detects objects, tracks them across frames, analyzes their movement and interactions, and classifies the observed behaviour as normal or anomalous.

---

## System Pipeline

```text
Video Input
    ↓
Frame Preprocessing
    ↓
Object Detection
    ↓
Object Tracking
    ↓
Behaviour Analysis
    ↓
Behaviour Classification
    ↓
Normal / Anomaly
```

The system is intended for edge deployment, with model training and experimentation performed on a GPU-equipped development system before optimization for an edge device such as a Raspberry Pi.

---

## Datasets

### UCF-Crime

UCF-Crime is used as the primary surveillance anomaly dataset for behaviour and anomaly classification.

The current processed dataset contains:

- 14 classes
- 1,377,653 extracted frames
- 1,610 training videos
- 290 test videos
- No overlap between training and test source videos
- Frames extracted at regular intervals from the source videos

The extracted frames preserve the original source-video identifier and frame number, allowing temporal sequences to be reconstructed.

---

### UCF-Crime2Local

UCF-Crime2Local is used for the object-detection component of the system.

It provides ground-truth bounding-box annotations that can be used to train and evaluate the YOLO object detector.

---

## Dataset Setup and Download

The project uses **KaggleHub** to download the Kaggle datasets.

### 1. Install Dependencies

From the project directory:

```bash
cd ~/Dev/Computer-Vision
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r model/requirements.txt
```

The requirements include:

- `kagglehub` — Kaggle dataset downloads
- `python-dotenv` — loads Kaggle credentials from `.env`
- TensorFlow and related libraries — model development
- OpenCV and Pillow — image/video processing
- NumPy and Pandas — data processing
- Scikit-learn — model evaluation
- Matplotlib and Seaborn — visualization

---

## Kaggle Authentication

The dataset downloader uses **KaggleHub** with Kaggle credentials stored in the project's `.env` file.

### 1. Create the `.env` File

Create a `.env` file in the project root:

```text
Computer-Vision/
├── .env
├── data/
├── model/
└── ...
```

Add your Kaggle credentials:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

Replace the values with your actual Kaggle username and API key.

---

## Downloading the Datasets

The download script is:

```text
model/01_download_data.py
```

The script supports downloading either dataset individually or both datasets.

## Downloader Options

Display the available options with:

```bash
python model/01_download_data.py --help
```

---

## Dataset Directory Structure

After downloading the datasets, the structure will be:

```text
Computer-Vision/
├── data/
│   ├── Train/
│   ├── Test/
│   ├── manifests/
│   │   ├── train_manifest.csv
│   │   ├── test_manifest.csv
│   │   └── ...
│   │
│   ├── ucf_crime/
│   │   └── ...
│   │
│   └── ucfcrime2local/
│       └── ...
│
├── model/
│   ├── 01_download_data.py
│   ├── 02_explore_data.py
│   ├── 03_build_manifest.py
│   └── ...
│
├── .env
└── ...
```

The `.env` file remains in the project root but is excluded from Git.

---

## Current Dataset Usage

The two datasets serve different purposes in the system:

```text
UCF-Crime
    ↓
Behaviour / Anomaly Classification


UCF-Crime2Local
    ↓
Ground-Truth Bounding Boxes
    ↓
YOLO Object Detection
```

The UCF-Crime2Local annotations will first be inspected and mapped to their corresponding frames before being converted into the format required for YOLO training.

````

---

## Dataset Manifests

The UCF-Crime frame dataset is organized into manifests containing information such as:

- Source video
- Class
- Frame number
- Image path
- Dataset split

The manifests are stored in:

```text
data/manifests/
````

The source-video-level train/test separation is maintained to prevent frames from the same source video appearing in both splits.

---

## Object Detection

The first stage of the V2 pipeline is object detection.

```text
Video Frame
    ↓
YOLO
    ↓
Bounding Boxes
    ↓
Object Class
    ↓
Confidence Score
```

The detector is primarily intended to identify relevant objects such as people and other entities involved in the observed scene.

UCF-Crime2Local provides the ground-truth bounding boxes required for supervised detector training and evaluation.

---

## Object Tracking

After object detection, detected objects will be tracked across consecutive frames.

```text
Frame 1 → Detection → Person #1
Frame 2 → Detection → Person #1
Frame 3 → Detection → Person #1
...
```

The tracking stage provides persistent object identities and allows the system to construct trajectories over time.

A multi-object tracking algorithm such as **ByteTrack** or **BoT-SORT** will be evaluated.

---

## Behaviour Analysis

The tracking information will be used to derive behaviour-related features, including:

- Object position
- Movement direction
- Trajectory
- Velocity
- Acceleration
- Distance between objects
- Object interactions
- Movement patterns

These temporal features provide information that cannot be obtained reliably from a single video frame.

---

## Behaviour Classification

The extracted spatial and temporal information will be used for behaviour classification.

```text
Tracked Objects
      ↓
Trajectories
      ↓
Motion / Interaction Features
      ↓
Behaviour Classification
      ↓
Normal / Anomaly
```

The goal is to separate normal activities from abnormal activities while retaining information about the behaviour that produced the decision.

---

## Development Strategy

The models will initially be trained and evaluated on the available NVIDIA RTX 4050 GPU.

After achieving suitable detection and classification performance, the models will be optimized for edge deployment.

Potential optimization techniques include:

- Reduced input resolution
- Reduced frame rate
- Lightweight model architectures
- Model quantization
- TensorFlow Lite / ONNX-based deployment
- Reduced tracking frequency

The final objective is to achieve a practical balance between:

```text
Accuracy
    ↕
Inference Speed
    ↕
Memory Usage
    ↕
Edge Hardware Constraints
```

---

## Current Development Status

### Completed

- UCF-Crime dataset exploration
- Frame dataset validation
- Source-video analysis
- Train/test source-video separation verification
- Temporal frame-spacing verification
- UCF-Crime frame manifests
- UCF-Crime2Local dataset selection
- KaggleHub-based dataset downloader

### In Progress

- UCF-Crime2Local annotation inspection
- Bounding-box format verification
- Mapping annotations to source videos/frames
- YOLO dataset preparation

### Planned

- YOLO object detector training
- Object tracking
- Trajectory extraction
- Behaviour feature extraction
- Behaviour classification
- End-to-end anomaly detection pipeline
- Edge-device optimization
- Raspberry Pi deployment

```

This version treats the project as the **new V2 system from the beginning** rather than carrying the old CNN/demo implementation forward.
```
