import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data" / "ucf_crime"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "yolo_tracking"

MODEL_NAME = "yolo26s.pt"

# Process only relevant surveillance objects.
# COCO IDs:
# person = 0
# bicycle = 1
# car = 2
# motorcycle = 3
# bus = 5
# truck = 7
TRACK_CLASSES = [
    0,
    1,
    2,
    3,
    5,
    7,
]

CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

CONFIDENCE_THRESHOLD = 0.25

# UCF-Crime extracted frames are normally spaced
# by 10 original video frames.
SOURCE_FRAME_STEP = 10

# Output video FPS.
# This is a visualization value, NOT used for
# behaviour calculations.
OUTPUT_FPS = 10


# ============================================================
# Dataset discovery
# ============================================================


def extract_source_video(image_path):
    """
    Extract the source video identifier.

    Example:

        Abuse001_x264_120.png

    becomes:

        Abuse001_x264
    """

    stem = image_path.stem

    match = re.match(
        r"(.+)_(\d+)$",
        stem,
    )

    if not match:
        return None

    return match.group(1)


def extract_source_frame(image_path):
    """
    Extract the original frame number.

    Example:

        Abuse001_x264_120.png

    returns:

        120
    """

    stem = image_path.stem

    match = re.search(
        r"_(\d+)$",
        stem,
    )

    if not match:
        return None

    return int(match.group(1))


def find_source_videos(split):

    split_root = DATA_ROOT / split

    if not split_root.exists():
        raise FileNotFoundError(f"Dataset split not found:\n{split_root}")

    videos = defaultdict(list)

    for image_path in split_root.rglob("*"):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in {
            ".png",
            ".jpg",
            ".jpeg",
        }:
            continue

        relative = image_path.relative_to(split_root)

        if len(relative.parts) < 2:
            continue

        class_name = relative.parts[0]

        source_video = extract_source_video(image_path)
        source_frame = extract_source_frame(image_path)

        if source_video is None or source_frame is None:
            continue

        key = (
            class_name,
            source_video,
        )

        videos[key].append(
            (
                source_frame,
                image_path,
            )
        )

    for key in videos:
        videos[key].sort(key=lambda item: item[0])

    return videos


# ============================================================
# Sequence validation
# ============================================================


def get_valid_sequences(videos):

    valid = []

    for (
        class_name,
        source_video,
    ), frames in videos.items():
        if len(frames) < 2:
            continue

        # Verify temporal ordering.
        frame_numbers = [frame_number for frame_number, _ in frames]

        gaps = [
            frame_numbers[i + 1] - frame_numbers[i]
            for i in range(len(frame_numbers) - 1)
        ]

        # We don't require every gap to be 10,
        # but report unusual gaps.
        unusual_gaps = [gap for gap in gaps if gap != SOURCE_FRAME_STEP]

        valid.append(
            {
                "class": class_name,
                "video": source_video,
                "frames": frames,
                "unusual_gaps": unusual_gaps,
            }
        )

    return valid


def select_videos(
    valid_sequences,
    videos_per_class=None,
    seed=42,
):

    by_class = defaultdict(list)

    for sequence in valid_sequences:
        by_class[sequence["class"]].append(sequence)

    selected = []

    random.seed(seed)

    for class_name in sorted(by_class):
        candidates = by_class[class_name]

        if videos_per_class is None:
            selected.extend(candidates)
        else:
            random.shuffle(candidates)
            selected.extend(candidates[:videos_per_class])

    return selected


# ============================================================
# Track history
# ============================================================


def update_track_history(
    track_history,
    result,
    source_frame,
    sequence_index,
):

    if result.boxes is None:
        return

    if result.boxes.id is None:
        return

    boxes = result.boxes.xyxy.detach().cpu().tolist()

    track_ids = result.boxes.id.detach().cpu().tolist()

    classes = result.boxes.cls.detach().cpu().tolist()

    confidences = result.boxes.conf.detach().cpu().tolist()

    for box, track_id, class_id, confidence in zip(
        boxes,
        track_ids,
        classes,
        confidences,
    ):
        track_id = int(track_id)
        class_id = int(class_id)

        x1, y1, x2, y2 = box

        center_x = (x1 + x2) / 2

        center_y = (y1 + y2) / 2

        track_history.append(
            {
                "sequence_index": sequence_index,
                "source_frame": source_frame,
                "track_id": track_id,
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(
                    class_id,
                    str(class_id),
                ),
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": center_x,
                "center_y": center_y,
            }
        )


# ============================================================
# Track statistics
# ============================================================


def calculate_track_statistics(
    track_history,
):

    tracks = defaultdict(list)

    for row in track_history:
        key = (
            row["class_name"],
            row["track_id"],
        )

        tracks[key].append(row)

    statistics = []

    for (
        class_name,
        track_id,
    ), rows in tracks.items():
        rows.sort(key=lambda row: row["sequence_index"])

        source_frames = [row["source_frame"] for row in rows]

        centers = [
            (
                row["center_x"],
                row["center_y"],
            )
            for row in rows
        ]

        # Total pixel displacement.
        displacement = 0.0

        for i in range(len(centers) - 1):
            x1, y1 = centers[i]
            x2, y2 = centers[i + 1]

            dx = x2 - x1
            dy = y2 - y1

            displacement += (dx * dx + dy * dy) ** 0.5

        # Actual source-frame span.
        if len(source_frames) >= 2:
            frame_span = source_frames[-1] - source_frames[0]

        else:
            frame_span = 0

        statistics.append(
            {
                "class_name": class_name,
                "track_id": track_id,
                "frames_tracked": len(rows),
                "first_source_frame": source_frames[0],
                "last_source_frame": source_frames[-1],
                "source_frame_span": frame_span,
                "total_pixel_displacement": displacement,
            }
        )

    return statistics


# ============================================================
# Save CSV
# ============================================================


def save_csv(
    path,
    rows,
):

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# Tracking visualization
# ============================================================


def draw_tracking_boxes(
    frame,
    result,
):
    """
    Draw only bounding boxes and track IDs.

    No class names or confidence scores are displayed.
    """

    if result.boxes is None:
        return frame

    if result.boxes.id is None:
        return frame

    boxes = result.boxes.xyxy.detach().cpu().tolist()

    track_ids = result.boxes.id.detach().cpu().tolist()

    for box, track_id in zip(
        boxes,
        track_ids,
    ):
        x1, y1, x2, y2 = map(
            int,
            box,
        )

        track_id = int(track_id)

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        # Track ID
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return frame


# ============================================================
# Process one video
# ============================================================


def process_video(
    model,
    sequence,
    output_root,
    video_index,
    split,
    save_video,
):
    class_name = sequence["class"]
    source_video = sequence["video"]
    frames = sequence["frames"]

    print("\n" + "=" * 70)
    print(f"{class_name} / {source_video}")
    print("=" * 70)

    print(f"Frames: {len(frames):,}")

    if sequence["unusual_gaps"]:
        print(
            "Unusual frame gaps:",
            sorted(set(sequence["unusual_gaps"])),
        )

    # --------------------------------------------------------
    # Read first frame to determine video size
    # --------------------------------------------------------

    first_image = cv2.imread(str(frames[0][1]))

    if first_image is None:
        raise RuntimeError(f"Could not read:\n{frames[0][1]}")

    height, width = first_image.shape[:2]

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    video_dir = output_root / f"{video_index:04d}_{class_name}_{source_video}"

    video_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Tracking visualization is optional because saving an MP4
    # for every UCF-Crime video is unnecessary for training.
    writer = None
    output_video = None

    if save_video:
        output_video = video_dir / "tracking.mp4"

        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            OUTPUT_FPS,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Could not create video:\n{output_video}")

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    track_history = []

    total_detections = 0
    frames_with_detections = 0

    for sequence_index, (
        source_frame,
        image_path,
    ) in enumerate(frames):
        # Read frame once
        frame = cv2.imread(str(image_path))

        if frame is None:
            raise RuntimeError(f"Could not read:\n{image_path}")

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE_THRESHOLD,
            classes=TRACK_CLASSES,
            device=0 if torch.cuda.is_available() else "cpu",
            verbose=False,
        )

        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:
            detections = len(result.boxes.id)

            total_detections += detections

            if detections > 0:
                frames_with_detections += 1

        update_track_history(
            track_history,
            result,
            source_frame,
            sequence_index,
        )

        # Draw and save visualization only when requested.
        if save_video:
            annotated = draw_tracking_boxes(
                frame,
                result,
            )

            writer.write(annotated)

        if sequence_index % 100 == 0 or sequence_index == len(frames) - 1:
            print(
                f"Processed {sequence_index + 1:,}/{len(frames):,}",
                end="\r",
            )

    print()

    if writer is not None:
        writer.release()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    track_statistics = calculate_track_statistics(track_history)

    unique_tracks = len(track_statistics)

    average_track_length = (
        sum(item["frames_tracked"] for item in track_statistics) / unique_tracks
        if unique_tracks
        else 0
    )

    longest_track = max(
        (item["frames_tracked"] for item in track_statistics),
        default=0,
    )

    print("\nTracking results:")

    print(f"  Frames:               {len(frames):,}")

    print(f"  Frames with tracks:   {frames_with_detections:,}")

    print(f"  Total detections:     {total_detections:,}")

    print(f"  Unique tracks:        {unique_tracks:,}")

    print(f"  Average track length: {average_track_length:.2f}")

    print(f"  Longest track:        {longest_track:,} frames")

    if save_video:
        print(f"\nTracking video:\n  {output_video}")
    else:
        print("\nTracking video: not saved")

    # --------------------------------------------------------
    # Save track history
    # --------------------------------------------------------

    save_csv(
        video_dir / "track_history.csv",
        track_history,
    )

    save_csv(
        video_dir / "track_statistics.csv",
        track_statistics,
    )

    return {
        "split": split,
        "class": class_name,
        "video": source_video,
        "frames": len(frames),
        "frames_with_tracks": frames_with_detections,
        "total_detections": total_detections,
        "unique_tracks": unique_tracks,
        "average_track_length": average_track_length,
        "longest_track": longest_track,
        "output_video": str(output_video) if output_video else "",
    }


# ============================================================
# Command-line arguments
# ============================================================


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=("Run YOLO26s + ByteTrack on UCF-Crime source videos.")
    )

    parser.add_argument(
        "--split",
        choices=["Train", "Test", "Both"],
        default="Both",
        help="Original UCF-Crime split to process. Default: Both.",
    )

    parser.add_argument(
        "--videos_per_class",
        type=int,
        default=None,
        help="Number of source videos per class. If omitted, all valid source videos are processed.",
    )

    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save tracking visualization videos.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when selecting source videos.",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================


def main():

    args = parse_arguments()

    if args.videos_per_class is not None and args.videos_per_class < 1:
        raise ValueError("--videos_per_class must be at least 1.")

    print("=" * 70)
    print("YOLO26s + BYTE TRACK TRACKING TEST")
    print("=" * 70)

    if args.videos_per_class is None:
        print("\nVideo selection: ALL valid source videos")
    else:
        print(f"\nVideos per class: {args.videos_per_class}")
        print(f"Random seed:      {args.seed}")

    # --------------------------------------------------------
    # Determine which splits to process
    # --------------------------------------------------------

    splits = ["Train", "Test"] if args.split == "Both" else [args.split]

    # --------------------------------------------------------
    # Load model once
    # --------------------------------------------------------

    print(f"\nLoading {MODEL_NAME}...")

    model = YOLO(MODEL_NAME)

    # --------------------------------------------------------
    # Process each split
    # --------------------------------------------------------

    for split in splits:
        print("\n" + "=" * 70)
        print(f"PROCESSING {split.upper()} SPLIT")
        print("=" * 70)

        # ----------------------------------------------------
        # Find source videos
        # ----------------------------------------------------

        videos = find_source_videos(split)

        print(f"\nSource videos found: {len(videos):,}")

        valid_sequences = get_valid_sequences(videos)

        print(f"Valid source videos: {len(valid_sequences):,}")

        # ----------------------------------------------------
        # Select videos
        # ----------------------------------------------------

        selected = select_videos(
            valid_sequences,
            videos_per_class=args.videos_per_class,
            seed=args.seed,
        )

        print(f"Selected source videos: {len(selected):,}")

        if not selected:
            raise RuntimeError(f"No source videos selected for {split}.")

        # ----------------------------------------------------
        # Output directory
        # ----------------------------------------------------

        split_output_root = OUTPUT_ROOT / split

        split_output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        results = []

        # ----------------------------------------------------
        # Process videos
        # ----------------------------------------------------

        for index, sequence in enumerate(
            selected,
            start=1,
        ):
            result = process_video(
                model,
                sequence,
                split_output_root,
                index,
                split,
                args.save_video,
            )

            results.append(result)

        # ----------------------------------------------------
        # Save split summary
        # ----------------------------------------------------

        save_csv(
            split_output_root / "tracking_summary.csv",
            results,
        )

        # ----------------------------------------------------
        # Split statistics
        # ----------------------------------------------------

        total_frames = sum(item["frames"] for item in results)

        total_tracks = sum(item["unique_tracks"] for item in results)

        total_detections = sum(item["total_detections"] for item in results)

        print("\n" + "-" * 70)
        print(f"{split} SPLIT COMPLETE")
        print("-" * 70)

        print(f"Videos processed:    {len(results):,}")
        print(f"Frames processed:    {total_frames:,}")
        print(f"Total detections:    {total_detections:,}")
        print(f"Total unique tracks: {total_tracks:,}")

        print("\nResults:")
        print(f"  {split_output_root}")

    # --------------------------------------------------------
    # Overall completion
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRACKING TEST COMPLETE")
    print("=" * 70)

    print("\nResults:")
    print(f"  {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
