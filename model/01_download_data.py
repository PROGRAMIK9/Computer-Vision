import argparse
import os

import kagglehub
from dotenv import load_dotenv

load_dotenv()


DATASETS = {
    "ucf-crime": {
        "name": "odins0n/ucf-crime-dataset",
        "path": "ucf_crime",
    },
    "ucf-crime2local": {
        "name": "vulamnguyen/ucfcrime2local-with-ground-truth-bounding-boxes",
        "path": "ucfcrime2local",
    },
}


def download_dataset(dataset_name, download_path):
    os.makedirs(download_path, exist_ok=True)

    print(f"\nDownloading: {dataset_name}")
    print(f"Destination: {download_path}")

    path = kagglehub.dataset_download(
        dataset_name,
        output_dir=download_path,
    )

    print(f"Downloaded to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Download UCF-Crime datasets")

    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "ucf-crime", "ucf-crime2local"],
        help="Dataset to download (default: all)",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Base directory for datasets (default: ./data)",
    )

    args = parser.parse_args()

    if args.dataset == "all":
        datasets = DATASETS.values()
    else:
        datasets = [DATASETS[args.dataset]]

    for dataset in datasets:
        download_path = os.path.join(
            args.data_dir,
            dataset["path"],
        )

        download_dataset(
            dataset["name"],
            download_path,
        )

    print("\nAll requested downloads completed.")


if __name__ == "__main__":
    main()
