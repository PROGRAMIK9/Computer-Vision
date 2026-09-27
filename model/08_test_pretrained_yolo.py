import csv
import random
import time
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data" / "detection"

DATA_YAML = DATA_ROOT / "data.yaml"

VAL_IMAGE_ROOT = DATA_ROOT / "images" / "val"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "yolo_benchmark"

SAMPLE_ROOT = OUTPUT_ROOT / "benchmark_samples"

MODELS = [
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
]

IMG_SIZE = 320

BATCH_SIZE = 16

CONFIDENCE_THRESHOLD = 0.25

NUM_VISUAL_SAMPLES = 30

RANDOM_SEED = 42


# ============================================================
# Dataset
# ============================================================


def find_validation_images():
    """
    Find all images in the prepared validation split.
    """

    if not VAL_IMAGE_ROOT.exists():
        raise FileNotFoundError(
            f"Validation image directory not found:\n{VAL_IMAGE_ROOT}"
        )

    images = []

    for image_path in VAL_IMAGE_ROOT.rglob("*"):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
        }:
            continue

        images.append(image_path)

    images.sort()

    return images


# ============================================================
# Visual sample selection
# ============================================================


def select_visual_samples(images):
    """
    Select a fixed set of images so that every model is tested
    on exactly the same visual examples.
    """

    random.seed(RANDOM_SEED)

    if len(images) <= NUM_VISUAL_SAMPLES:
        return images

    return random.sample(
        images,
        NUM_VISUAL_SAMPLES,
    )


# ============================================================
# GPU helpers
# ============================================================


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_gpu_stats():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_mb():
    if not torch.cuda.is_available():
        return 0.0

    synchronize()

    return torch.cuda.max_memory_allocated() / (1024**2)


# ============================================================
# Model warm-up
# ============================================================


def warmup_model(
    model,
    image_path,
    device,
):
    """
    Warm up the model before timing inference.
    """

    print("  Warming up...")

    for _ in range(10):
        model.predict(
            source=str(image_path),
            imgsz=IMG_SIZE,
            device=device,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

    synchronize()


# ============================================================
# Detection speed test
# ============================================================


def benchmark_inference(
    model,
    images,
    device,
):
    """
    Measure inference latency on the validation images.

    This is separate from mAP evaluation.
    """

    print("\n  Measuring inference speed...")

    inference_times = []

    reset_gpu_stats()

    for index, image_path in enumerate(images, start=1):
        synchronize()

        start = time.perf_counter()

        model.predict(
            source=str(image_path),
            imgsz=IMG_SIZE,
            device=device,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        synchronize()

        elapsed = time.perf_counter() - start

        inference_times.append(elapsed)

        if index % 500 == 0 or index == len(images):
            print(
                f"    Processed {index:,}/{len(images):,}",
                end="\r",
            )

    print()

    average_ms = sum(inference_times) / len(inference_times) * 1000

    sorted_times = sorted(inference_times)

    median_ms = sorted_times[len(sorted_times) // 2] * 1000

    fps = 1000 / average_ms if average_ms > 0 else 0

    peak_memory = get_peak_memory_mb()

    return {
        "average_ms": average_ms,
        "median_ms": median_ms,
        "fps": fps,
        "peak_gpu_memory_mb": peak_memory,
    }


# ============================================================
# Visual predictions
# ============================================================


def save_visual_predictions(
    model,
    images,
    model_name,
    device,
):
    """
    Save annotated predictions for manual inspection.
    """

    output_dir = SAMPLE_ROOT / Path(model_name).stem

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"\n  Saving {len(images)} visual predictions...")

    for index, image_path in enumerate(
        images,
        start=1,
    ):
        results = model.predict(
            source=str(image_path),
            imgsz=IMG_SIZE,
            device=device,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        result = results[0]

        annotated = result.plot()

        output_name = f"{index:03d}_{image_path.name}"

        Image.fromarray(annotated).save(output_dir / output_name)

    print(f"  Saved to:\n  {output_dir}")


# ============================================================
# Evaluate one model
# ============================================================


def evaluate_model(
    model_name,
    validation_images,
    visual_samples,
    device,
):
    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = YOLO(model_name)

    print("Model loaded.")

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    warmup_model(
        model,
        validation_images[0],
        device,
    )

    # --------------------------------------------------------
    # Detection evaluation
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("GROUND-TRUTH DETECTION EVALUATION")
    print("-" * 70)

    start = time.perf_counter()

    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        plots=True,
        project=str(OUTPUT_ROOT),
        name=f"{Path(model_name).stem}_validation",
        exist_ok=True,
        verbose=True,
    )

    synchronize()

    evaluation_time = time.perf_counter() - start

    box = metrics.box

    precision = float(box.mp)
    recall = float(box.mr)
    map50 = float(box.map50)
    map75 = float(box.map75)
    map5095 = float(box.map)

    # --------------------------------------------------------
    # Speed benchmark
    # --------------------------------------------------------

    speed = benchmark_inference(
        model,
        validation_images,
        device,
    )

    # --------------------------------------------------------
    # Visual predictions
    # --------------------------------------------------------

    save_visual_predictions(
        model,
        visual_samples,
        model_name,
        device,
    )

    # --------------------------------------------------------
    # Overall results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(f"\nPrecision       : {precision:.4f}")

    print(f"Recall          : {recall:.4f}")

    print(f"mAP@50          : {map50:.4f}")

    print(f"mAP@75          : {map75:.4f}")

    print(f"mAP@50:95       : {map5095:.4f}")

    print("\nInference:")

    print(f"Average latency : {speed['average_ms']:.2f} ms")

    print(f"Median latency  : {speed['median_ms']:.2f} ms")

    print(f"Approx. FPS     : {speed['fps']:.2f}")

    print(f"Peak GPU memory : {speed['peak_gpu_memory_mb']:.1f} MB")

    print(f"\nEvaluation time : {evaluation_time:.2f} seconds")

    # --------------------------------------------------------
    # Per-class results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-CLASS RESULTS")
    print("=" * 70)

    names = metrics.names

    per_class = []

    for class_id in range(len(names)):
        try:
            class_metrics = box.class_result(class_id)

            class_precision = float(class_metrics[0])

            class_recall = float(class_metrics[1])

            class_map50 = float(class_metrics[2])

            class_map5095 = float(class_metrics[3])

        except Exception:
            class_precision = 0.0
            class_recall = 0.0
            class_map50 = 0.0
            class_map5095 = 0.0

        class_name = names[class_id]

        print(
            f"{class_id:2d} "
            f"{class_name:<12} "
            f"P={class_precision:.4f}  "
            f"R={class_recall:.4f}  "
            f"mAP50={class_map50:.4f}  "
            f"mAP50:95={class_map5095:.4f}"
        )

        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": class_precision,
                "recall": class_recall,
                "map50": class_map50,
                "map50_95": class_map5095,
            }
        )

    return {
        "model": model_name,
        "images": len(validation_images),
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map75": map75,
        "map50_95": map5095,
        "average_ms": speed["average_ms"],
        "median_ms": speed["median_ms"],
        "fps": speed["fps"],
        "peak_gpu_memory_mb": speed["peak_gpu_memory_mb"],
        "evaluation_time": evaluation_time,
        "per_class": per_class,
    }


# ============================================================
# Save results
# ============================================================


def save_results(results):
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Overall CSV
    # --------------------------------------------------------

    overall_path = OUTPUT_ROOT / "pretrained_yolo_results.csv"

    fieldnames = [
        "model",
        "images",
        "precision",
        "recall",
        "map50",
        "map75",
        "map50_95",
        "average_ms",
        "median_ms",
        "fps",
        "peak_gpu_memory_mb",
        "evaluation_time",
    ]

    with overall_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow({key: result[key] for key in fieldnames})

    # --------------------------------------------------------
    # Per-class CSV
    # --------------------------------------------------------

    class_path = OUTPUT_ROOT / "pretrained_yolo_per_class.csv"

    with class_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "class_id",
                "class_name",
                "precision",
                "recall",
                "map50",
                "map50_95",
            ],
        )

        writer.writeheader()

        for result in results:
            for row in result["per_class"]:
                writer.writerow(
                    {
                        "model": result["model"],
                        **row,
                    }
                )

    print("\nResults saved:")
    print(f"  {overall_path}")
    print(f"  {class_path}")


# ============================================================
# Main
# ============================================================


def main():

    print("=" * 70)
    print("PRETRAINED YOLO - UCF-CRIME2LOCAL")
    print("=" * 70)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset configuration not found:\n{DATA_YAML}")

    validation_images = find_validation_images()

    if not validation_images:
        raise RuntimeError("No validation images found.")

    print("\nDataset:")
    print(f"  Configuration : {DATA_YAML}")
    print(f"  Validation images : {len(validation_images):,}")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Visual samples
    # --------------------------------------------------------

    visual_samples = select_visual_samples(validation_images)

    print(f"\nVisual samples: {len(visual_samples)}")

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Evaluate models
    # --------------------------------------------------------

    results = []

    for model_name in MODELS:
        try:
            result = evaluate_model(
                model_name,
                validation_images,
                visual_samples,
                device,
            )

            results.append(result)

        except Exception as e:
            print("\n" + "!" * 70)
            print(f"ERROR while evaluating {model_name}")
            print("!" * 70)

            print(e)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    if results:
        save_results(results)

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        f"\n{'Model':<15}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'mAP50':>12}"
        f"{'mAP50:95':>14}"
        f"{'FPS':>10}"
    )

    print("-" * 75)

    for result in results:
        print(
            f"{result['model']:<15}"
            f"{result['precision']:>12.4f}"
            f"{result['recall']:>12.4f}"
            f"{result['map50']:>12.4f}"
            f"{result['map50_95']:>14.4f}"
            f"{result['fps']:>10.2f}"
        )

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    print(f"\nResults directory:\n  {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
