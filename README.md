# V1 Anomaly Detection Demo

Web-based demo for the V1 anomaly detection model.

he V1 demo processes the selected video frame by frame. Each frame is extracted from the test dataset, resized to 224 × 224 pixels, and passed independently to the trained MobileNetV3Small model. The model outputs an anomaly probability, which is compared against a threshold of 0.5 to classify the frame as either NORMAL or ANOMALY.

---


## How to Run the Demo

From the project directory:

```bash
cd ~/Dev/Computer-Vision
````

### Option 1 — Using the Virtual Environment

If you are using the project's Python virtual environment, activate it with:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r model/requirements.txt
```

The requirements include **FastAPI** and **Uvicorn**, which are required to run the demo backend.

Then go to the demo directory:

```bash
cd v1_demo
```

Start the backend:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Option 2 — Using uv

If you are using `uv`, go to the demo directory:

```bash
cd v1_demo
```

Install the project dependencies:

```bash
uv sync
```

If FastAPI and Uvicorn are not already included in the project dependencies, install them with:

```bash
uv add fastapi uvicorn
```

Then start the backend:

```bash
uv run python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Open the Demo

Once the server is running, open the following URL in a browser:

```text
http://127.0.0.1:8000
```

The V1 Anomaly Detection Demo will then be available in the browser.


## Demo

The demo is divided into two sections.

### Left — Video / Input

* Category selection
* Video selection
* Current frame
* Source frame
* Filename
* Video frame display
* Frame timeline
* Previous / Next frame
* Play / Pause
* Playback speed

### Right — Model / Output

* Model information
* Model architecture
* Current prediction
* Anomaly probability
* Normal probability
* Raw model output
* Decision threshold
* Final decision

The model classifies each frame as:

```text
NORMAL
```

or:

```text
ANOMALY
```

The decision threshold is:

```text
0.5
```

---

# Performance

V1 model performance on the test dataset:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **72.92%** |
| Precision | **65.48%** |
| Recall    | **73.99%** |
| F1 Score  | **69.48%** |

---


## Best Classes to Show

For a short presentation/demo, the recommended classes are:

### Strong Anomaly Detection

* **Burglary — 91.67%**
* **Vandalism — 86.14%**
* **Explosion — 84.02%**

### Normal Example

* **Normal Videos — 72.16%**

### More Challenging Examples

* **Arrest — 62.23%**
* **Shooting — 57.35%**
* **Road Accidents — 52.76%**

These classes provide a good selection for manually demonstrating both successful and challenging predictions.

---

## Demo Model

**Model:** MobileNetV3Small

**Input:** `224 × 224 × 3`

**Output:** Anomaly probability

**Threshold:** `0.5`

### Classification

```text
Probability > 0.5  → ANOMALY
Probability ≤ 0.5  → NORMAL
```