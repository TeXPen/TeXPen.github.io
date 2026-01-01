import os
import requests
import tarfile
import shutil
import subprocess

OUTPUT_DIR = "public/models/paddle"
TEMP_DIR = "_temp_paddle_conversion"

# Official Mobile/Lightweight Models (Tarballs with Legacy pdmodel -> Convertible)
MOBILE_URLS = {
    # Reverting to V4 Mobile because V5 Mobile uses PIR (json) format which paddle2onnx 1.3.1 cannot parse.
    "rec.onnx": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar",  # V4 Mobile Rec
    "det.onnx": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar",  # V4 Mobile Det
    "layout.onnx": "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar",  # PicoDet Layout (Mobile-class)
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


def convert_tar_model(url, target_connx_name):
    print(f"--- Processing {target_connx_name} ---")
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    final_path = os.path.join(OUTPUT_DIR, target_connx_name)
    abs_path = os.path.abspath(final_path)
    print(f"  Checking {abs_path}")
    if os.path.exists(final_path):
        print(f"  {target_connx_name} already exists. Skipping.")
        return

    # Download Tar
    tar_name = url.split("/")[-1]
    tar_path = os.path.join(TEMP_DIR, tar_name)
    if not os.path.exists(tar_path):
        download_file(url, tar_path)

    # Extract
    print("  Extracting...")
    # Extract to a unique subdir named after the tar file (without extension)
    extract_subdir_name = tar_name.replace(".tar", "")
    unique_extract_dir = os.path.join(TEMP_DIR, extract_subdir_name)

    # Clean previous extraction if exists
    if os.path.exists(unique_extract_dir):
        shutil.rmtree(unique_extract_dir)

    # Python's tarfile extractall extracts into the current directory with the folder structure inside the tar
    # So if we extract to TEMP_DIR, it will create TEMP_DIR/ch_PP-OCRv4_rec_infer/
    # We want to be careful about what the internal folder name is.
    with tarfile.open(tar_path, "r") as tar:
        # Debug: list members
        members = tar.getnames()
        print(f"  Tar members (first 5): {members[:5]}")
        tar.extractall(path=TEMP_DIR)

    # We need to find *where* it extracted.
    # Use members to determine root
    extract_subdir_name = (
        members[0].split("/")[0] if members else tar_name.replace(".tar", "")
    )
    print(f"  Detected extract root from tar: {extract_subdir_name}")
    # Usually it's `extract_subdir_name` but let's verify.
    model_dir = os.path.join(TEMP_DIR, extract_subdir_name)
    if not os.path.exists(model_dir):
        # Try finding the directory in TEMP_DIR that isn't a file
        for d in os.listdir(TEMP_DIR):
            full_d = os.path.join(TEMP_DIR, d)
            if (
                os.path.isdir(full_d) and d != "paddle_model_temp_dir"
            ):  # Avoid paddle internal dir
                # rudimentary check: does it look like our model?
                # Assuming 1 extraction happens at a time mostly or names differ.
                if extract_subdir_name in d or d in extract_subdir_name:
                    model_dir = full_d
                    break

    print(f"  Located model dir: {model_dir}")

    # Convert
    print(f"  Converting {model_dir}...")

    # PicoDet layout models sometimes need specific opset or input shapes?
    # paddle2onnx usually handles it.

    # Detect model filename
    model_file = "inference.pdmodel"
    params_file = "inference.pdiparams"
    if not os.path.exists(os.path.join(model_dir, model_file)):
        if os.path.exists(os.path.join(model_dir, "model.pdmodel")):
            model_file = "model.pdmodel"
            params_file = "model.pdiparams"
            print(f"  Detected {model_file}")
        elif os.path.exists(os.path.join(model_dir, "inference.json")):
            model_file = "inference.json"
            # params file matches json? usually inference.pdiparams
            print(f"  Detected {model_file} (PIR Format)")

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
    for name, url in MOBILE_URLS.items():
        convert_tar_model(url, name)

    # 2. Keys
    download_keys()

    # Cleanup
    # if os.path.exists(TEMP_DIR):
    #     shutil.rmtree(TEMP_DIR)

    print("Done! All official mobile models ready.")


if __name__ == "__main__":
    main()
