import argparse
import os
from collections import Counter
from itertools import pairwise
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".webm",
}


# ============================================================================
# General Utilities
# ============================================================================


def get_files(directory, extensions):
    """Recursively find files with the specified extensions."""

    directory = Path(directory)

    if not directory.exists():
        return []

    return [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]


def get_text_files(directory):
    """Recursively find text files."""

    directory = Path(directory)

    if not directory.exists():
        return []

    return [path for path in directory.rglob("*.txt") if path.is_file()]


def get_image_properties(image_files, sample_size=100):
    """
    Inspect image dimensions and channels using a sample of images.
    """

    if not image_files:
        return None

    sample = image_files[:sample_size]

    dimensions = Counter()
    channels = Counter()
    unreadable = []

    for image_path in sample:
        image = cv2.imread(str(image_path))

        if image is None:
            unreadable.append(str(image_path))
            continue

        height, width = image.shape[:2]

        dimensions[(width, height)] += 1

        if len(image.shape) == 2:
            channels[1] += 1
        else:
            channels[image.shape[2]] += 1

    return {
        "dimensions": dimensions,
        "channels": channels,
        "unreadable": unreadable,
        "sample_size": len(sample),
    }


def calculate_directory_size(files):
    """Calculate total size of a collection of files."""

    total_size = 0

    for path in files:
        try:
            total_size += path.stat().st_size
        except OSError:
            pass

    return total_size


# ============================================================================
# UCF-Crime
# ============================================================================


def get_class_counts(directory):
    """
    Count image files belonging to each class.

    Expected structure:

        Train/
            Abuse/
                image1.png
            NormalVideos/
                image2.png
    """

    counts = {}

    directory = Path(directory)

    if not directory.exists():
        return counts

    for class_path in directory.iterdir():
        if not class_path.is_dir():
            continue

        files = get_files(
            class_path,
            IMAGE_EXTENSIONS,
        )

        counts[class_path.name] = len(files)

    return counts


def get_directory_info(directory):
    """Check for nested directories inside class directories."""

    nested = []

    directory = Path(directory)

    if not directory.exists():
        return nested

    for class_path in directory.iterdir():
        if not class_path.is_dir():
            continue

        for path in class_path.rglob("*"):
            if path.is_dir() and path != class_path:
                nested.append(path)

    return nested


def analyze_split(split_name, split_dir):
    """Analyze one UCF-Crime dataset split."""

    print("\n" + "=" * 80)
    print(f"{split_name.upper()} SET")
    print("=" * 80)

    if not os.path.exists(split_dir):
        print(f"ERROR: {split_dir} does not exist.")
        return None

    print(f"Path: {split_dir}")

    # ------------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------------

    image_files = get_files(
        split_dir,
        IMAGE_EXTENSIONS,
    )

    video_files = get_files(
        split_dir,
        VIDEO_EXTENSIONS,
    )

    print(f"\nTotal images : {len(image_files):,}")
    print(f"Total videos : {len(video_files):,}")

    # ------------------------------------------------------------------------
    # Class distribution
    # ------------------------------------------------------------------------

    counts = get_class_counts(split_dir)

    print("\nClass distribution:")

    for class_name, count in sorted(counts.items()):
        print(f"  {class_name:<30} {count:>12,} frames")

    # ------------------------------------------------------------------------
    # Dataset size
    # ------------------------------------------------------------------------

    total_size = calculate_directory_size(image_files)

    print(f"\nImage storage: {total_size / (1024**3):.2f} GB")

    # ------------------------------------------------------------------------
    # Image properties
    # ------------------------------------------------------------------------

    properties = get_image_properties(
        image_files,
        sample_size=100,
    )

    if properties:
        print(f"\nImage inspection sample: {properties['sample_size']} images")

        print("\nImage dimensions:")

        for (width, height), count in properties["dimensions"].most_common():
            print(f"  {width}x{height:<10} {count:>5} images")

        print("\nChannels:")

        for channel_count, count in properties["channels"].most_common():
            print(f"  {channel_count} channels: {count} images")

        if properties["unreadable"]:
            print("\nUnreadable images in sample:")

            for path in properties["unreadable"][:10]:
                print(f"  {path}")

        else:
            print("\nUnreadable images in sample: 0")

    # ------------------------------------------------------------------------
    # Nested directory check
    # ------------------------------------------------------------------------

    nested = get_directory_info(split_dir)

    if nested:
        print(f"\nNested directories found: {len(nested)}")

        for path in nested[:20]:
            print(f"  {path}")

        if len(nested) > 20:
            print(f"  ... and {len(nested) - 20} more")

    else:
        print("\nNested directories inside classes: 0")

    # ------------------------------------------------------------------------
    # Sample filenames
    # ------------------------------------------------------------------------

    print("\nSample files:")

    for path in image_files[:10]:
        try:
            relative = path.relative_to(split_dir)
        except ValueError:
            relative = path

        print(f"  {relative}")

    return {
        "name": split_name,
        "image_files": image_files,
        "video_files": video_files,
        "class_counts": counts,
    }


def analyze_source_videos(split_name, split_dir):
    """
    Analyze source videos encoded in extracted frame filenames.

    Expected examples:

        Abuse001_x264_0.png
        Abuse001_x264_10.png
        Abuse001_x264_20.png
    """

    print("\n" + "=" * 80)
    print(f"{split_name.upper()} SOURCE VIDEO ANALYSIS")
    print("=" * 80)

    image_files = get_files(
        split_dir,
        IMAGE_EXTENSIONS,
    )

    source_info = {}

    for image_path in image_files:
        stem = image_path.stem

        parts = stem.rsplit("_", 1)

        if len(parts) != 2:
            continue

        source_id = parts[0]
        frame_string = parts[1]

        try:
            frame_number = int(frame_string)
        except ValueError:
            continue

        source_info.setdefault(
            source_id,
            [],
        ).append(frame_number)

    print(f"\nUnique source videos: {len(source_info):,}")

    # ------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------

    frame_counts = [len(frames) for frames in source_info.values()]

    if frame_counts:
        frame_series = pd.Series(frame_counts)

        print("\nFrames per source video:")

        print(f"  Minimum : {frame_series.min():,.0f}")

        print(f"  Mean    : {frame_series.mean():,.2f}")

        print(f"  Median  : {frame_series.median():,.0f}")

        print(f"  Maximum : {frame_series.max():,.0f}")

    # ------------------------------------------------------------------------
    # Frame continuity
    # ------------------------------------------------------------------------

    continuous = 0
    discontinuous = 0

    for frames in source_info.values():
        frames = sorted(frames)

        if len(frames) < 2:
            continue

        differences = [b - a for a, b in pairwise(frames)]

        if all(difference == 10 for difference in differences):
            continuous += 1
        else:
            discontinuous += 1

    print("\nFrame sequence structure:")

    print(f"  Regular 10-frame spacing : {continuous:,} videos")

    print(f"  Irregular spacing        : {discontinuous:,} videos")

    # ------------------------------------------------------------------------
    # Sample source videos
    # ------------------------------------------------------------------------

    print("\nSample source videos:")

    for source_id in sorted(source_info)[:20]:
        frames = sorted(source_info[source_id])

        print(
            f"  {source_id:<30} {len(frames):>6,} frames [{frames[0]} → {frames[-1]}]"
        )

    return source_info


def check_source_overlap(
    train_sources,
    test_sources,
):

    print("\n" + "=" * 80)
    print("TRAIN / TEST SOURCE VIDEO OVERLAP")
    print("=" * 80)

    train_ids = set(train_sources.keys())

    test_ids = set(test_sources.keys())

    overlap = train_ids & test_ids

    print(f"Training source videos : {len(train_ids):,}")

    print(f"Testing source videos  : {len(test_ids):,}")

    print(f"Source videos in both  : {len(overlap):,}")

    if overlap:
        print("\nWARNING: Source-video overlap detected!")

        for source_id in sorted(overlap)[:50]:
            print(f"  {source_id}")

    else:
        print("\nNo source-video overlap detected.")

        print("Train/Test separation appears to be video-level.")


def create_distribution_plot(
    train_counts,
    test_counts,
    data_dir,
):

    df_train = pd.DataFrame(
        list(train_counts.items()),
        columns=["Class", "Count"],
    )

    df_train["Split"] = "Train"

    df_test = pd.DataFrame(
        list(test_counts.items()),
        columns=["Class", "Count"],
    )

    df_test["Split"] = "Test"

    df_all = pd.concat(
        [df_train, df_test],
        ignore_index=True,
    )

    plt.figure(figsize=(14, 7))

    sns.barplot(
        x="Class",
        y="Count",
        hue="Split",
        data=df_all,
    )

    plt.title("Class Distribution in UCF-Crime Dataset")

    plt.xlabel("Class")
    plt.ylabel("Number of Frames")

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    plot_path = os.path.join(
        data_dir,
        "class_distribution.png",
    )

    plt.savefig(
        plot_path,
        dpi=150,
    )

    plt.close()

    print(f"\nSaved class distribution plot:\n  {plot_path}")


def print_comparison(
    train_counts,
    test_counts,
):

    print("\n" + "=" * 80)
    print("TRAIN / TEST COMPARISON")
    print("=" * 80)

    classes = sorted(set(train_counts) | set(test_counts))

    rows = []

    for class_name in classes:
        train_count = train_counts.get(
            class_name,
            0,
        )

        test_count = test_counts.get(
            class_name,
            0,
        )

        rows.append(
            {
                "Class": class_name,
                "Train": train_count,
                "Test": test_count,
                "Total": (train_count + test_count),
            }
        )

    df = pd.DataFrame(rows)

    print(df.to_string(index=False))

    print(f"\nTotal training frames: {df['Train'].sum():,}")

    print(f"Total testing frames: {df['Test'].sum():,}")

    print(f"Total frames: {df['Total'].sum():,}")


# ============================================================================
# UCF-Crime2Local
# ============================================================================


def read_split_file(split_file):
    """
    Read a UCF-Crime2Local train/test split file.

    The exact path format is preserved rather than assuming
    a particular naming convention.
    """

    entries = []

    if not split_file.exists():
        return entries

    with open(
        split_file,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            entries.append(line)

    return entries


def analyze_split_file(
    split_name,
    split_file,
):

    entries = read_split_file(split_file)

    print(f"\n{split_name} split file:")

    print(f"  Path    : {split_file}")

    print(f"  Entries : {len(entries):,}")

    if entries:
        print("  Samples:")

        for entry in entries[:10]:
            print(f"    {entry}")

    return entries


def analyze_crime2local(
    dataset_dir,
):

    print("\n\n" + "=" * 80)
    print("UCF-CRIME2LOCAL DATASET EXPLORATION")
    print("=" * 80)

    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory does not exist:\n  {dataset_dir}")

        return

    print(f"\nDataset directory:\n  {dataset_dir}")

    # ------------------------------------------------------------------------
    # Expected structure
    # ------------------------------------------------------------------------

    rgb_dir = dataset_dir / "rgb-images"
    labels_dir = dataset_dir / "labels"

    train_file = dataset_dir / "train.txt"
    test_file = dataset_dir / "test.txt"
    labels_file = dataset_dir / "labels.txt"

    print("\nExpected dataset components:")

    components = {
        "RGB images": rgb_dir,
        "Labels": labels_dir,
        "Class labels": labels_file,
        "Train split": train_file,
        "Test split": test_file,
    }

    for name, path in components.items():
        status = "FOUND" if path.exists() else "MISSING"

        print(f"  {name:<15} {status:<8} {path}")

    # ------------------------------------------------------------------------
    # RGB images
    # ------------------------------------------------------------------------

    image_files = get_files(
        rgb_dir,
        IMAGE_EXTENSIONS,
    )

    print("\nRGB image analysis:")

    print(f"  Total images : {len(image_files):,}")

    image_size = calculate_directory_size(image_files)

    print(f"  Storage      : {image_size / (1024**3):.2f} GB")

    properties = get_image_properties(
        image_files,
        sample_size=100,
    )

    if properties:
        print(f"\n  Inspection sample: {properties['sample_size']} images")

        print("\n  Dimensions:")

        for (
            dimensions,
            count,
        ) in properties["dimensions"].most_common():
            width, height = dimensions

            print(f"    {width}x{height}: {count}")

        print("\n  Channels:")

        for (
            channel_count,
            count,
        ) in properties["channels"].most_common():
            print(f"    {channel_count}: {count} images")

        print(f"\n  Unreadable images in sample: {len(properties['unreadable'])}")

        if properties["unreadable"]:
            for path in properties["unreadable"][:10]:
                print(f"    {path}")

    # ------------------------------------------------------------------------
    # Label files
    # ------------------------------------------------------------------------

    label_files = get_text_files(labels_dir)

    print("\nAnnotation analysis:")

    print(f"  Label files : {len(label_files):,}")

    label_size = calculate_directory_size(label_files)

    print(f"  Storage     : {label_size / (1024**2):.2f} MB")

    if label_files:
        print("\n  Sample annotation files:")

        for path in label_files[:10]:
            try:
                relative = path.relative_to(labels_dir)
            except ValueError:
                relative = path

            print(f"    {relative}")

    # ------------------------------------------------------------------------
    # Annotation content inspection
    # ------------------------------------------------------------------------
    #
    # We intentionally do not assume that the labels are already
    # in YOLO format. The first few annotation files are inspected
    # as raw text so the exact format can be verified before
    # conversion/training.
    # ------------------------------------------------------------------------

    if label_files:
        print("\nAnnotation format samples:")

        for label_path in label_files[:5]:
            print(f"\n  {label_path.name}:")

            try:
                with open(
                    label_path,
                    "r",
                    encoding="utf-8",
                    errors="ignore",
                ) as file:
                    lines = [line.strip() for line in file if line.strip()]

                if not lines:
                    print("    [empty annotation]")

                else:
                    for line in lines[:5]:
                        print(f"    {line}")

                    if len(lines) > 5:
                        print(f"    ... {len(lines) - 5} more lines")

            except OSError as error:
                print(f"    Could not read: {error}")

    # ------------------------------------------------------------------------
    # Split files
    # ------------------------------------------------------------------------

    train_entries = analyze_split_file(
        "Train",
        train_file,
    )

    test_entries = analyze_split_file(
        "Test",
        test_file,
    )

    # ------------------------------------------------------------------------
    # Split overlap
    # ------------------------------------------------------------------------

    if train_entries and test_entries:
        train_set = set(train_entries)

        test_set = set(test_entries)

        overlap = train_set & test_set

        print("\nTrain/Test split overlap:")

        print(f"  Train entries : {len(train_set):,}")

        print(f"  Test entries  : {len(test_set):,}")

        print(f"  Overlap       : {len(overlap):,}")

        if overlap:
            print("\n  WARNING: overlapping entries detected.")

            for entry in sorted(overlap)[:20]:
                print(f"    {entry}")

        else:
            print("\n  No train/test entry overlap detected.")

    # ------------------------------------------------------------------------
    # Dataset consistency
    # ------------------------------------------------------------------------

    print("\nImage/annotation consistency:")

    # Compare relative paths rather than only filenames.
    #
    # Example:
    #
    #   rgb-images/Arrest/Arrest002/00256.jpg
    #   labels/Arrest/Arrest002/00256.txt
    #
    # Both become:
    #
    #   Arrest/Arrest002/00256
    #
    # This is necessary because frame numbers such as 00256
    # can occur in multiple source videos.

    image_keys = {path.relative_to(rgb_dir).with_suffix("") for path in image_files}

    label_keys = {path.relative_to(labels_dir).with_suffix("") for path in label_files}

    images_with_labels = image_keys & label_keys

    images_without_labels = image_keys - label_keys

    labels_without_images = label_keys - image_keys

    print(f"  Images                : {len(image_keys):,}")

    print(f"  Labels                : {len(label_keys):,}")

    print(f"  Images with labels    : {len(images_with_labels):,}")

    print(f"  Images without labels : {len(images_without_labels):,}")

    print(f"  Labels without images : {len(labels_without_images):,}")

    if images_without_labels:
        print("\n  Sample images without labels:")

        for name in sorted(images_without_labels)[:20]:
            print(f"    {name}")

    if labels_without_images:
        print("\n  Sample labels without images:")

        for name in sorted(labels_without_images)[:20]:
            print(f"    {name}")

    # ------------------------------------------------------------------------
    # Class list
    # ------------------------------------------------------------------------

    if labels_file.exists():
        print("\nClass label file:")

        try:
            with open(
                labels_file,
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as file:
                class_lines = [line.strip() for line in file if line.strip()]

            print(f"  Number of entries: {len(class_lines):,}")

            for index, line in enumerate(class_lines[:50]):
                print(f"  {index}: {line}")

            if len(class_lines) > 50:
                print(f"  ... {len(class_lines) - 50} more entries")

        except OSError as error:
            print(f"  Could not read {labels_file}: {error}")

    print("\nUCF-Crime2Local exploration complete.")


# ============================================================================
# Main
# ============================================================================


def main():

    parser = argparse.ArgumentParser(
        description=("Explore and analyze the Computer Vision datasets.")
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help=("Directory containing the dataset directories (default: ./data)."),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=[
            "all",
            "ucf-crime",
            "ucf-crime2local",
        ],
        help=("Dataset to analyze (default: all)."),
    )

    args = parser.parse_args()

    data_dir = Path(os.path.abspath(os.path.expanduser(args.data_dir)))

    print("=" * 80)
    print("COMPUTER VISION DATASET EXPLORATION")
    print("=" * 80)

    print(f"\nData directory:\n  {data_dir}")

    # =========================================================================
    # UCF-Crime
    # =========================================================================

    if args.dataset in {
        "all",
        "ucf-crime",
    }:
        ucf_dir = data_dir / "ucf_crime"

        # The downloaded UCF-Crime dataset has:
        #
        # data/ucf_crime/
        # ├── Train/
        # └── Test/

        train_dir = ucf_dir / "Train"
        test_dir = ucf_dir / "Test"

        if train_dir.exists() and test_dir.exists():
            print("\n\n" + "#" * 80)

            print("# UCF-CRIME")

            print("#" * 80)

            train_info = analyze_split(
                "Train",
                train_dir,
            )

            train_sources = analyze_source_videos(
                "Train",
                train_dir,
            )

            test_info = analyze_split(
                "Test",
                test_dir,
            )

            test_sources = analyze_source_videos(
                "Test",
                test_dir,
            )

            if train_info and test_info:
                print_comparison(
                    train_info["class_counts"],
                    test_info["class_counts"],
                )

                create_distribution_plot(
                    train_info["class_counts"],
                    test_info["class_counts"],
                    data_dir,
                )

                check_source_overlap(
                    train_sources,
                    test_sources,
                )

        else:
            print("\nUCF-Crime dataset not found at:")

            print(f"  {ucf_dir}")

    # =========================================================================
    # UCF-Crime2Local
    # =========================================================================

    if args.dataset in {
        "all",
        "ucf-crime2local",
    }:
        crime2local_dir = data_dir / "ucfcrime2local" / "ucfcrime2local"

        if crime2local_dir.exists():
            print("\n\n" + "#" * 80)

            print("# UCF-CRIME2LOCAL")

            print("#" * 80)

            analyze_crime2local(crime2local_dir)

        else:
            print("\nUCF-Crime2Local dataset not found at:")

            print(f"  {crime2local_dir}")

    print("\n\n" + "=" * 80)

    print("DATASET EXPLORATION COMPLETE")

    print("=" * 80)


if __name__ == "__main__":
    main()
