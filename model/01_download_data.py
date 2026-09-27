import argparse
import os

import kagglehub
import requests
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


UCA_FILES = {
    "UCFCrime_Train.json": "PASTE_TRAIN_URL_HERE",
    "UCFCrime_Val.json": "PASTE_VAL_URL_HERE",
    "UCFCrime_Test.json": "PASTE_TEST_URL_HERE",
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


def download_uca(download_path):
    os.makedirs(download_path, exist_ok=True)

    print("\nDownloading: UCA annotations")
    print(f"Destination: {download_path}")

    for filename, url in UCA_FILES.items():
        output_file = os.path.join(
            download_path,
            filename,
        )

        if os.path.exists(output_file):
            print(f"\nAlready exists: {filename}")
            continue

        print(f"\nDownloading: {filename}")

        response = requests.get(
            url,
            stream=True,
            timeout=60,
        )

        response.raise_for_status()

        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        print(f"Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download UCF-Crime datasets and UCA annotations"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=[
            "all",
            "ucf-crime",
            "ucf-crime2local",
            "uca",
        ],
        help="Dataset to download (default: all)",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Base directory for datasets (default: ./data)",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Kaggle datasets
    # --------------------------------------------------------

    if args.dataset == "all":
        datasets = DATASETS.values()

    elif args.dataset in DATASETS:
        datasets = [DATASETS[args.dataset]]

    else:
        datasets = []

    for dataset in datasets:
        download_path = os.path.join(
            args.data_dir,
            dataset["path"],
        )

        download_dataset(
            dataset["name"],
            download_path,
        )

    # --------------------------------------------------------
    # UCA annotations
    # --------------------------------------------------------

    if args.dataset in ["all", "uca"]:
        uca_path = os.path.join(
            args.data_dir,
            "UCA(UCF Crime Annotation) Dataset",
        )

        download_uca(uca_path)

    print("\nAll requested downloads completed.")


if __name__ == "__main__":
    main()
