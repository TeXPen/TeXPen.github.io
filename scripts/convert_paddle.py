import os
import tarfile
import shutil
import subprocess

OUTPUT_DIR = "public/models/paddle"
TEMP_DIR = "_temp_paddle_conversion"

# Official Mobile/Lightweight Models
# V5 Mobile is available on HuggingFace as raw files (PIR format: inference.json + inference.pdiparams)
MOBILE_MODELS = {
    "rec.onnx": {
        "type": "hf_raw",
        "repo": "PaddlePaddle/PP-OCRv5_mobile_rec",
        "files": ["inference.pdiparams", "inference.json"],  # PIR format
    },
    "det.onnx": {
        "type": "hf_raw",
        "repo": "PaddlePaddle/PP-OCRv5_mobile_det",
        "files": ["inference.pdiparams", "inference.json"],  # PIR format
    },
    "layout.onnx": {
        "type": "tar",
        "url": "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar",
    },
}

KEYS_URL = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ppocr_keys_v1.txt"


def download_file(url, save_path):
    print(f"Downloading {url}...")
    try:
        # Use curl for reliability on Windows/Generic
        subprocess.run(["curl", "-L", url, "-o", save_path], check=True)
    except Exception:
        # Fallback to requests (or just fail)
        print("  curl failed, trying requests...")
        import requests

        r = requests.get(url, stream=True)
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def convert_model_entry(target_connx_name, config):
    print(f"--- Processing {target_connx_name} ---")
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    final_path = os.path.join(OUTPUT_DIR, target_connx_name)
    if os.path.exists(final_path):
        print(f"  {target_connx_name} already exists. Skipping.")
        return

    # Create a specific temp dir for this model
    model_friendly_name = target_connx_name.replace(".onnx", "")
    model_temp_dir = os.path.join(TEMP_DIR, model_friendly_name)
    if os.path.exists(model_temp_dir):
        shutil.rmtree(model_temp_dir)
    os.makedirs(model_temp_dir)

    model_file = "inference.pdmodel"
    params_file = "inference.pdiparams"

    if config["type"] == "tar":
        url = config["url"]
        tar_name = url.split("/")[-1]
        tar_path = os.path.join(TEMP_DIR, tar_name)
        if not os.path.exists(tar_path):
            download_file(url, tar_path)

        print("  Extracting...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=model_temp_dir)

        # Find the logical root inside
        # Usually tar contains one folder
        subdirs = [
            d
            for d in os.listdir(model_temp_dir)
            if os.path.isdir(os.path.join(model_temp_dir, d))
        ]
        if subdirs:
            real_root = os.path.join(model_temp_dir, subdirs[0])
        else:
            real_root = model_temp_dir

        model_dir = real_root

        # Detect files
        if os.path.exists(os.path.join(model_dir, "inference.json")):
            model_file = "inference.json"
            # params usually match

    elif config["type"] == "hf_raw":
        repo = config["repo"]
        base_url = f"https://huggingface.co/{repo}/resolve/main"
        model_dir = model_temp_dir

        for fname in config["files"]:
            url = f"{base_url}/{fname}"
            save_dest = os.path.join(model_dir, fname)
            download_file(url, save_dest)

            if fname.endswith(".json"):
                model_file = fname
            elif fname.endswith(".pdiparams"):
                params_file = fname
            elif fname.endswith(".pdmodel"):
                model_file = fname

    print(f"  Converting {model_dir}...")
    print(f"  Model File: {model_file}")
    print(f"  Params File: {params_file}")

    cmd = [
        "paddle2onnx",
        "--model_dir",
        model_dir,
        "--model_filename",
        model_file,
        "--params_filename",
        params_file,
        "--save_file",
        final_path,
        "--opset_version",
        "11",
        "--enable_onnx_checker",
        "True",
    ]

    try:
        subprocess.run(cmd, check=True)
        print("  Success!")
    except subprocess.CalledProcessError as e:
        print(f"  Conversion failed: {e}")


def download_keys():
    print("--- Processing Keys ---")
    keys_path = os.path.join(OUTPUT_DIR, "ppocr_keys_v1.txt")
    if not os.path.exists(keys_path):
        download_file(KEYS_URL, keys_path)
        print("  Success!")
    else:
        print("  Keys already exist.")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Convert All
    for name, config in MOBILE_MODELS.items():
        convert_model_entry(name, config)

    # 2. Keys
    download_keys()

    print("Done! All official mobile models ready.")


if __name__ == "__main__":
    main()
