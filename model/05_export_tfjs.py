import os
import argparse
import subprocess

def export_tfjs(model_path, output_dir):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Exporting model from {model_path} to TFJS format in {output_dir}...")
    
    try:
        subprocess.run([
            "tensorflowjs_converter",
            "--input_format=keras",
            model_path,
            output_dir
        ], check=True)
        print("TFJS Export complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error exporting to TFJS: {e}")
    except FileNotFoundError:
        print("tensorflowjs_converter not found. Please install it using 'pip install tensorflowjs'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to TFJS format.")
    parser.add_argument("--model_path", type=str, default="./saved_models/model_final.keras", help="Path to the saved Keras model")
    parser.add_argument("--output_dir", type=str, default="./tfjs_model", help="Directory to save the TFJS model")
    args = parser.parse_args()
    
    export_tfjs(args.model_path, args.output_dir)
