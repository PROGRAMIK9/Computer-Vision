const classSelect = document.getElementById("classSelect");

const videoSelect = document.getElementById("videoSelect");

const speedSelect = document.getElementById("speedSelect");

const frameImage = document.getElementById("frameImage");

const detectionCanvas = document.getElementById("detectionCanvas");

const loading = document.getElementById("loading");

const frameNumber = document.getElementById("frameNumber");

const sourceFrame = document.getElementById("sourceFrame");

const filename = document.getElementById("filename");

const predictionCard = document.getElementById("predictionCard");

const predictionLabel = document.getElementById("predictionLabel");

const confidence = document.getElementById("confidence");

const rawPrediction = document.getElementById("rawPrediction");

const decision = document.getElementById("decision");

const frameSlider = document.getElementById("frameSlider");

const previousButton = document.getElementById("previousButton");

const nextButton = document.getElementById("nextButton");

const playButton = document.getElementById("playButton");

const sideClass = document.getElementById("sideClass");

const sideVideo = document.getElementById("sideVideo");

const sideFrames = document.getElementById("sideFrames");

const normalProbability = document.getElementById("normalProbability");

const anomalyProbability = document.getElementById("anomalyProbability");

const normalBar = document.getElementById("normalBar");

const anomalyBar = document.getElementById("anomalyBar");

const timelinePosition = document.getElementById("timelinePosition");

const objectCount = document.getElementById("objectCount");

const detectionList = document.getElementById("detectionList");

let currentClass = null;
let currentVideo = null;

let frames = [];
let currentFrame = 0;

let playing = false;
let playbackTimer = null;

/* ============================================================
API helper
============================================================ */

async function getJSON(url) {
  const response = await fetch(url);

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }

  return data;
}

/* ============================================================
Canvas helpers
============================================================ */

/*

* Clear all YOLO bounding boxes.
  */

function clearDetections() {
  const context = detectionCanvas.getContext("2d");

  context.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
}

/*

* Draw YOLO bounding boxes.
*
* YOLO coordinates are based on the original image.
*
* The canvas is displayed at the same size as the
* image, so the original coordinates are scaled
* to the displayed dimensions.
  */

function drawDetections(detections) {
  const context = detectionCanvas.getContext("2d");

  const imageWidth = frameImage.naturalWidth;

  const imageHeight = frameImage.naturalHeight;

  if (!imageWidth || !imageHeight) {
    return;
  }

  /*
   * Get the actual displayed image dimensions.
   */

  const displayedWidth = frameImage.clientWidth;

  const displayedHeight = frameImage.clientHeight;

  /*
   * Set canvas resolution to the displayed
   * image dimensions.
   */

  detectionCanvas.width = displayedWidth;

  detectionCanvas.height = displayedHeight;

  /*
   * Because object-fit: contain is used,
   * the actual image may not occupy the entire
   * image element.
   *
   * Calculate the contained image dimensions.
   */

  const imageRatio = imageWidth / imageHeight;

  const containerRatio = displayedWidth / displayedHeight;

  let renderedWidth;
  let renderedHeight;
  let offsetX;
  let offsetY;

  if (imageRatio > containerRatio) {
    renderedWidth = displayedWidth;

    renderedHeight = displayedWidth / imageRatio;

    offsetX = 0;

    offsetY = (displayedHeight - renderedHeight) / 2;
  } else {
    renderedHeight = displayedHeight;

    renderedWidth = displayedHeight * imageRatio;

    offsetX = (displayedWidth - renderedWidth) / 2;

    offsetY = 0;
  }

  const scaleX = renderedWidth / imageWidth;

  const scaleY = renderedHeight / imageHeight;

  /*
   * Clear previous boxes.
   */

  context.clearRect(0, 0, displayedWidth, displayedHeight);

  /*
   * Draw every YOLO detection.
   */

  for (const detection of detections) {
    const x = offsetX + detection.x1 * scaleX;

    const y = offsetY + detection.y1 * scaleY;

    const width = (detection.x2 - detection.x1) * scaleX;

    const height = (detection.y2 - detection.y1) * scaleY;

    /*
     * Bounding box.
     */

    context.strokeStyle = "#ffffff";

    context.lineWidth = 2;

    context.strokeRect(x, y, width, height);

    /*
     * Label.
     */

    const label = `${detection.class_name} ${(detection.confidence * 100).toFixed(0)}%`;

    context.font = "bold 12px Arial";

    const textWidth = context.measureText(label).width;

    const labelHeight = 18;

    /*
     * Keep label inside the image.
     */

    const labelY = Math.max(labelHeight, y);

    /*
     * Label background.
     */

    context.fillStyle = "rgba(0, 0, 0, 0.75)";

    context.fillRect(x, labelY - labelHeight, textWidth + 8, labelHeight);

    /*
     * Label text.
     */

    context.fillStyle = "#ffffff";

    context.fillText(label, x + 4, labelY - 5);
  }
}

/*

* Update the object detection list below the image.
  */

function updateDetectionList(detections) {
  detectionList.innerHTML = "";

  objectCount.textContent = `${detections.length} ${
    detections.length === 1 ? "object" : "objects"
  }`;

  if (!detections.length) {
    const empty = document.createElement("span");

    empty.className = "no-detections";

    empty.textContent = "No objects detected";

    detectionList.appendChild(empty);

    return;
  }

  for (const detection of detections) {
    const item = document.createElement("div");

    item.className = "detection-item";

    const name = document.createElement("span");

    name.className = "detection-name";

    name.textContent = detection.class_name;

    const probability = document.createElement("strong");

    probability.className = "detection-confidence";

    probability.textContent = `${(detection.confidence * 100).toFixed(1)}%`;

    item.appendChild(name);

    item.appendChild(probability);

    detectionList.appendChild(item);
  }
}

/*

* Clear the object detection UI.
  */

function resetDetections() {
  clearDetections();

  objectCount.textContent = "0 objects";

  detectionList.innerHTML = `<span class="no-detections">
        Analyzing...
    </span>`;
}

/* ============================================================
Load classes
============================================================ */

async function loadClasses() {
  const classes = await getJSON("/api/classes");

  classSelect.innerHTML = "";

  for (const cls of classes) {
    const option = document.createElement("option");

    option.value = cls.name;

    option.textContent = `${cls.name} (${cls.video_count} videos)`;

    classSelect.appendChild(option);
  }

  if (classes.length > 0) {
    await loadClass(classes[0].name);
  }
}

/* ============================================================
Load class
============================================================ */

async function loadClass(className) {
  stopPlayback();

  currentClass = className;

  sideClass.textContent = className;

  const videos = await getJSON(`/api/videos/${encodeURIComponent(className)}`);

  videoSelect.innerHTML = "";

  for (const video of videos) {
    const option = document.createElement("option");

    option.value = video.id;

    option.textContent = `${video.id} (${video.frame_count} frames)`;

    videoSelect.appendChild(option);
  }

  if (videos.length > 0) {
    await loadVideo(videos[0].id);
  }
}

/* ============================================================
Load video
============================================================ */

async function loadVideo(videoId) {
  stopPlayback();

  currentVideo = videoId;

  sideVideo.textContent = videoId;

  const data = await getJSON(
    `/api/video/${encodeURIComponent(currentClass)}/${encodeURIComponent(videoId)}`,
  );

  frames = data.frames;

  currentFrame = 0;

  sideFrames.textContent = frames.length;

  frameSlider.min = 0;

  frameSlider.max = Math.max(0, frames.length - 1);

  frameSlider.value = 0;

  await showFrame(currentFrame);
}

/* ============================================================
Show frame
============================================================ */

async function showFrame(frameIndex) {
  if (!frames.length || frameIndex < 0 || frameIndex >= frames.length) {
    return;
  }

  currentFrame = frameIndex;

  frameSlider.value = currentFrame;

  const frame = frames[currentFrame];

  /* --------------------------------------------------------
   Show image
   -------------------------------------------------------- */

  loading.style.display = "block";

  resetDetections();

  /*
   * Remove previous onload handler.
   */

  frameImage.onload = null;

  /*
   * Set image.
   */

  frameImage.src = `/api/image/${encodeURIComponent(currentClass)}/${encodeURIComponent(currentVideo)}/${currentFrame}?t=${Date.now()}`;

  frameImage.onload = () => {
    loading.style.display = "none";
  };

  /* --------------------------------------------------------
   Frame metadata
   -------------------------------------------------------- */

  frameNumber.textContent = `Frame ${currentFrame + 1} / ${frames.length}`;

  sourceFrame.textContent = `Source frame: ${frame.source_frame}`;

  filename.textContent = frame.filename;

  /* --------------------------------------------------------
   Reset prediction
   -------------------------------------------------------- */

  predictionLabel.textContent = "ANALYZING";

  confidence.textContent = "...";

  rawPrediction.textContent = "...";

  decision.textContent = "...";

  try {
    /*
     * Run both V1 anomaly detection and
     * YOLO object detection.
     */

    const result = await getJSON(
      `/api/predict/${encodeURIComponent(currentClass)}/${encodeURIComponent(currentVideo)}/${currentFrame}`,
    );

    /* ====================================================
       V1 ANOMALY RESULT
       ==================================================== */

    const probability = result.anomaly_probability;

    const normal = result.normal_probability;

    const anomaly = result.anomaly_probability;

    normalProbability.textContent = `${(normal * 100).toFixed(2)}%`;

    anomalyProbability.textContent = `${(anomaly * 100).toFixed(2)}%`;

    normalBar.style.width = `${normal * 100}%`;

    anomalyBar.style.width = `${anomaly * 100}%`;

    timelinePosition.textContent = `${currentFrame + 1} / ${frames.length}`;

    predictionLabel.textContent = result.label;

    confidence.textContent = `${(probability * 100).toFixed(2)}%`;

    rawPrediction.textContent = probability.toFixed(4);

    decision.textContent = result.label;

    predictionCard.classList.remove("normal", "anomaly");

    predictionCard.classList.add(result.label.toLowerCase());

    /* ====================================================
       YOLO RESULT
       ==================================================== */

    const detections = result.detections || [];

    updateDetectionList(detections);

    /*
     * Wait until the image has finished loading
     * before drawing the boxes.
     */

    if (frameImage.complete) {
      drawDetections(detections);
    } else {
      frameImage.onload = () => {
        loading.style.display = "none";

        drawDetections(detections);
      };
    }
  } catch (error) {
    console.error(error);

    predictionLabel.textContent = "ERROR";

    confidence.textContent = "-";

    rawPrediction.textContent = "-";

    decision.textContent = error.message;

    objectCount.textContent = "Error";

    detectionList.innerHTML = `<span class="no-detections">
            Object detection failed
        </span>`;
  }
}

/* ============================================================
Redraw detections when browser window changes size
============================================================ */

window.addEventListener("resize", () => {
  /*
   * The current detections are not stored separately,
   * so the current frame is simply re-requested when
   * necessary.
   *
   * Cached backend predictions make this inexpensive.
   */

  if (currentClass !== null && currentVideo !== null && frames.length) {
    const frame = frames[currentFrame];

    if (frame) {
      getJSON(
        `/api/predict/${encodeURIComponent(currentClass)}/${encodeURIComponent(currentVideo)}/${currentFrame}`,
      )
        .then((result) => {
          drawDetections(result.detections || []);
        })
        .catch((error) => {
          console.error("Failed to redraw detections:", error);
        });
    }
  }
});

/* ============================================================
Class changed
============================================================ */

classSelect.addEventListener("change", async () => {
  await loadClass(classSelect.value);
});

/* ============================================================
Video changed
============================================================ */

videoSelect.addEventListener("change", async () => {
  await loadVideo(videoSelect.value);
});

/* ============================================================
Previous
============================================================ */

previousButton.addEventListener("click", async () => {
  stopPlayback();

  if (currentFrame > 0) {
    await showFrame(currentFrame - 1);
  }
});

/* ============================================================
Next
============================================================ */

nextButton.addEventListener("click", async () => {
  stopPlayback();

  if (currentFrame < frames.length - 1) {
    await showFrame(currentFrame + 1);
  }
});

/* ============================================================
Slider
============================================================ */

frameSlider.addEventListener("input", async () => {
  stopPlayback();

  await showFrame(Number(frameSlider.value));
});

/* ============================================================
Playback
============================================================ */

function startPlayback() {
  if (playing || !frames.length) {
    return;
  }

  playing = true;

  playButton.textContent = "⏸ Pause";

  playNext();
}

async function playNext() {
  if (!playing) {
    return;
  }

  if (currentFrame >= frames.length - 1) {
    stopPlayback();

    return;
  }

  await showFrame(currentFrame + 1);

  if (!playing) {
    return;
  }

  playbackTimer = setTimeout(playNext, Number(speedSelect.value));
}

function stopPlayback() {
  playing = false;

  playButton.textContent = "▶ Play";

  if (playbackTimer !== null) {
    clearTimeout(playbackTimer);

    playbackTimer = null;
  }
}

playButton.addEventListener("click", () => {
  if (playing) {
    stopPlayback();
  } else {
    startPlayback();
  }
});

/* ============================================================
Keyboard controls
============================================================ */

document.addEventListener("keydown", async (event) => {
  if (event.target.tagName === "SELECT") {
    return;
  }

  if (event.code === "Space") {
    event.preventDefault();

    if (playing) {
      stopPlayback();
    } else {
      startPlayback();
    }
  }

  if (event.code === "ArrowLeft") {
    stopPlayback();

    if (currentFrame > 0) {
      await showFrame(currentFrame - 1);
    }
  }

  if (event.code === "ArrowRight") {
    stopPlayback();

    if (currentFrame < frames.length - 1) {
      await showFrame(currentFrame + 1);
    }
  }
});

/* ============================================================
Start
============================================================ */

loadClasses();
