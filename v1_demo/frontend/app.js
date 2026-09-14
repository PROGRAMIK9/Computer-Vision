const classSelect =
    document.getElementById("classSelect");

const videoSelect =
    document.getElementById("videoSelect");

const speedSelect =
    document.getElementById("speedSelect");

const frameImage =
    document.getElementById("frameImage");

const loading =
    document.getElementById("loading");

const frameNumber =
    document.getElementById("frameNumber");

const sourceFrame =
    document.getElementById("sourceFrame");

const filename =
    document.getElementById("filename");

const predictionCard =
    document.getElementById("predictionCard");

const predictionLabel =
    document.getElementById("predictionLabel");

const confidence =
    document.getElementById("confidence");

const rawPrediction =
    document.getElementById("rawPrediction");

const decision =
    document.getElementById("decision");

const frameSlider =
    document.getElementById("frameSlider");

const previousButton =
    document.getElementById("previousButton");

const nextButton =
    document.getElementById("nextButton");

const playButton =
    document.getElementById("playButton");

const sideClass =
    document.getElementById("sideClass");

const sideVideo =
    document.getElementById("sideVideo");

const sideFrames =
    document.getElementById("sideFrames");

const normalProbability =
    document.getElementById("normalProbability");

const anomalyProbability =
    document.getElementById("anomalyProbability");

const normalBar =
    document.getElementById("normalBar");

const anomalyBar =
    document.getElementById("anomalyBar");

const timelinePosition =
    document.getElementById("timelinePosition");


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

    const response =
        await fetch(url);

    const data =
        await response.json();

    if (!response.ok) {

        throw new Error(
            data.detail ||
            "Request failed"
        );
    }

    return data;
}


/* ============================================================
   Load classes
   ============================================================ */

async function loadClasses() {

    const classes =
        await getJSON(
            "/api/classes"
        );

    classSelect.innerHTML = "";

    for (const cls of classes) {

        const option =
            document.createElement(
                "option"
            );

        option.value =
            cls.name;

        option.textContent =
            `${cls.name} (${cls.video_count} videos)`;

        classSelect.appendChild(
            option
        );
    }

    if (classes.length > 0) {

        await loadClass(
            classes[0].name
        );
    }
}


/* ============================================================
   Load class
   ============================================================ */

async function loadClass(
    className
) {

    stopPlayback();

    currentClass =
        className;

    sideClass.textContent =
        className;

    const videos =
        await getJSON(
            `/api/videos/${encodeURIComponent(className)}`
        );

    videoSelect.innerHTML = "";

    for (const video of videos) {

        const option =
            document.createElement(
                "option"
            );

        option.value =
            video.id;

        option.textContent =
            `${video.id} (${video.frame_count} frames)`;

        videoSelect.appendChild(
            option
        );
    }

    if (videos.length > 0) {

        await loadVideo(
            videos[0].id
        );
    }
}


/* ============================================================
   Load video
   ============================================================ */

async function loadVideo(
    videoId
) {

    stopPlayback();

    currentVideo =
        videoId;

    sideVideo.textContent =
        videoId;

    const data =
        await getJSON(
            `/api/video/${encodeURIComponent(currentClass)}/${encodeURIComponent(videoId)}`
        );

    frames =
        data.frames;

    currentFrame = 0;

    sideFrames.textContent =
        frames.length;

    frameSlider.min = 0;

    frameSlider.max =
        Math.max(
            0,
            frames.length - 1
        );

    frameSlider.value = 0;

    await showFrame(
        currentFrame
    );
}


/* ============================================================
   Show frame
   ============================================================ */

async function showFrame(
    frameIndex
) {

    if (
        !frames.length ||
        frameIndex < 0 ||
        frameIndex >= frames.length
    ) {
        return;
    }

    currentFrame =
        frameIndex;

    frameSlider.value =
        currentFrame;

    const frame =
        frames[currentFrame];


    /* Show image */

    loading.style.display =
        "block";

    frameImage.src =
        `/api/image/${encodeURIComponent(currentClass)}/${encodeURIComponent(currentVideo)}/${currentFrame}?t=${Date.now()}`;

    frameImage.onload =
        () => {
            loading.style.display =
                "none";
        };


    /* Frame metadata */

    frameNumber.textContent =
        `Frame ${currentFrame + 1} / ${frames.length}`;

    sourceFrame.textContent =
        `Source frame: ${frame.source_frame}`;

    filename.textContent =
        frame.filename;


    /* Reset prediction while loading */

    predictionLabel.textContent =
        "ANALYZING";

    confidence.textContent =
        "...";

    rawPrediction.textContent =
        "...";

    decision.textContent =
        "...";


    try {

        /*
         * This request causes exactly ONE frame
         * to be passed through the V1 model.
         */

        const result =
            await getJSON(
                `/api/predict/${encodeURIComponent(currentClass)}/${encodeURIComponent(currentVideo)}/${currentFrame}`
            );


        const probability =
            result.anomaly_probability;

        const normal =
            result.normal_probability;

        const anomaly =
            result.anomaly_probability;

        normalProbability.textContent =
            `${(normal * 100).toFixed(2)}%`;

        anomalyProbability.textContent =
            `${(anomaly * 100).toFixed(2)}%`;

        normalBar.style.width =
            `${normal * 100}%`;

        anomalyBar.style.width =
            `${anomaly * 100}%`;

        timelinePosition.textContent =
            `${currentFrame + 1} / ${frames.length}`;

        predictionLabel.textContent =
            result.label;

        confidence.textContent =
            `${(probability * 100).toFixed(2)}%`;

        rawPrediction.textContent =
            probability.toFixed(4);

        decision.textContent =
            result.label;


        predictionCard.classList.remove(
            "normal",
            "anomaly"
        );

        predictionCard.classList.add(
            result.label.toLowerCase()
        );

    } catch (error) {

        console.error(error);

        predictionLabel.textContent =
            "ERROR";

        confidence.textContent =
            "-";

        rawPrediction.textContent =
            "-";

        decision.textContent =
            error.message;
    }
}


/* ============================================================
   Class changed
   ============================================================ */

classSelect.addEventListener(
    "change",
    async () => {

        await loadClass(
            classSelect.value
        );
    }
);


/* ============================================================
   Video changed
   ============================================================ */

videoSelect.addEventListener(
    "change",
    async () => {

        await loadVideo(
            videoSelect.value
        );
    }
);


/* ============================================================
   Previous
   ============================================================ */

previousButton.addEventListener(
    "click",
    async () => {

        stopPlayback();

        if (currentFrame > 0) {

            await showFrame(
                currentFrame - 1
            );
        }
    }
);


/* ============================================================
   Next
   ============================================================ */

nextButton.addEventListener(
    "click",
    async () => {

        stopPlayback();

        if (
            currentFrame <
            frames.length - 1
        ) {

            await showFrame(
                currentFrame + 1
            );
        }
    }
);


/* ============================================================
   Slider
   ============================================================ */

frameSlider.addEventListener(
    "input",
    async () => {

        stopPlayback();

        await showFrame(
            Number(
                frameSlider.value
            )
        );
    }
);


/* ============================================================
   Playback
   ============================================================ */

function startPlayback() {

    if (
        playing ||
        !frames.length
    ) {
        return;
    }

    playing = true;

    playButton.textContent =
        "⏸ Pause";

    playNext();
}


async function playNext() {

    if (!playing) {
        return;
    }

    if (
        currentFrame >=
        frames.length - 1
    ) {

        stopPlayback();

        return;
    }

    await showFrame(
        currentFrame + 1
    );

    if (!playing) {
        return;
    }

    playbackTimer =
        setTimeout(
            playNext,
            Number(
                speedSelect.value
            )
        );
}


function stopPlayback() {

    playing = false;

    playButton.textContent =
        "▶ Play";

    if (playbackTimer !== null) {

        clearTimeout(
            playbackTimer
        );

        playbackTimer = null;
    }
}


playButton.addEventListener(
    "click",
    () => {

        if (playing) {

            stopPlayback();

        } else {

            startPlayback();
        }
    }
);


/* ============================================================
   Keyboard controls
   ============================================================ */

document.addEventListener(
    "keydown",
    async event => {

        if (
            event.target.tagName ===
            "SELECT"
        ) {
            return;
        }

        if (
            event.code ===
            "Space"
        ) {

            event.preventDefault();

            if (playing) {
                stopPlayback();
            } else {
                startPlayback();
            }
        }

        if (
            event.code ===
            "ArrowLeft"
        ) {

            stopPlayback();

            if (currentFrame > 0) {

                await showFrame(
                    currentFrame - 1
                );
            }
        }

        if (
            event.code ===
            "ArrowRight"
        ) {

            stopPlayback();

            if (
                currentFrame <
                frames.length - 1
            ) {

                await showFrame(
                    currentFrame + 1
                );
            }
        }
    }
);


/* ============================================================
   Start
   ============================================================ */

loadClasses();