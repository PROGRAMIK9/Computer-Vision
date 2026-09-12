import os
import argparse
import subprocess
import zipfile
import sys
from dotenv import load_dotenv

load_dotenv()

def download_dataset(dataset_name, download_path):
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    print(f"Downloading {dataset_name} to {download_path}...")
    try:
        # Requires kaggle API credentials configured (~/.kaggle/kaggle.json or KAGGLE_API_TOKEN)
        subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download", "-d", dataset_name, "-p", download_path], check=True)
        print("Download complete.")
        
        # Unzip the downloaded file
        zip_file = dataset_name.split("/")[-1] + ".zip"
        zip_path = os.path.join(download_path, zip_file)
        
        if os.path.exists(zip_path):
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            print("Extraction complete.")
            os.remove(zip_path) # Clean up zip file
        else:
            print(f"Expected zip file not found at {zip_path}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset. Ensure Kaggle API is configured. {e}")
    except FileNotFoundError:
        print("Kaggle CLI not found. Please install it using 'pip install kaggle'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kaggle dataset.")
    parser.add_argument("--dataset", type=str, default="odins0n/ucf-crime-dataset", help="Kaggle dataset name")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save the dataset")
    args = parser.parse_args()
    
    download_dataset(args.dataset, args.data_dir)
