import os
import shutil
import subprocess
from huggingface_hub import hf_hub_download, HfApi

# Constants
OUTPUT_DIR = "public/models/paddle/structure_v3"
TEMP_DIR = "_temp_structure_v3_conversion"

# Model Definitions
MODELS = {
    "layout": {
        "repo_id": "PaddlePaddle/PicoDet-S_layout_17cls",
        "files": ["inference.json", "inference.pdiparams"],
        "onnx_name": "layout.onnx",
    },
    "table": {
        "repo_id": "PaddlePaddle/SLANet",
        "files": ["inference.json", "inference.pdiparams"],
        "onnx_name": "table_structure.onnx",
    },
}


def download_model(repo_id, files, dest_dir):
    print(f"Downloading {repo_id} to {dest_dir}...")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for filename in files:
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id, filename=filename, local_dir=dest_dir
            )
            print(f"Downloaded {filename} to {downloaded}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            # Try inspection if failed
            api = HfApi()
            print(f"Available files in {repo_id}:")
            repo_files = api.list_repo_files(repo_id)
            for rf in repo_files:
                print(f" - {rf}")
            raise e


def convert_to_onnx(
    model_dir,
    save_file,
    model_filename="inference.json",
    params_filename="inference.pdiparams",
):
    print(f"Converting model in {model_dir} to {save_file}...")

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    cmd = [
        "paddle2onnx",
        "--model_dir",
        model_dir,
        "--model_filename",
        model_filename,
        "--params_filename",
        params_filename,
        "--save_file",
        save_file,
        "--opset_version",
        "16",
        "--enable_onnx_checker",
        "True",
    ]

    print(f"Running command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully converted to {save_file}")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")
        raise e


def main():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for key, config in MODELS.items():
        print(f"\nProcessing {key}...")
        repo_id = config["repo_id"]
        model_temp_dir = os.path.join(TEMP_DIR, key)

        # Download
        try:
            download_model(repo_id, config["files"], model_temp_dir)
        except Exception as e:
            print(f"Skipping {key} due to download error: {e}")
            continue

        # Convert
        save_path = os.path.join(OUTPUT_DIR, config["onnx_name"])
        try:
            convert_to_onnx(
                model_dir=model_temp_dir,
                save_file=save_path,
                model_filename=config["files"][0],
                params_filename=config["files"][1],
            )
        except Exception as e:
            print(f"Skipping {key} due to conversion error: {e}")

    # Cleanup
    # shutil.rmtree(TEMP_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
