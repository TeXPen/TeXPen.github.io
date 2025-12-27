import os
from huggingface_hub import snapshot_download


def download_models():
    models_dir = os.path.join(os.path.dirname(__file__), "../_local/models")
    os.makedirs(models_dir, exist_ok=True)

    # PaddleOCR-VL
    # The 0.9B model seems to be gated or under a different name, but the main VL repo is public.
    # We will try downloading the main repo which likely contains the model or instructions.
    print("Downloading PaddleOCR-VL...")
    try:
        vl_model_path = snapshot_download(
            repo_id="PaddlePaddle/PaddleOCR-VL",
            local_dir=os.path.join(models_dir, "PaddleOCR-VL"),
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        print(f"PaddleOCR-VL downloaded to: {vl_model_path}")
    except Exception as e:
        print(f"Failed to download PaddleOCR-VL: {e}")

    # PP-DocLayoutV2
    # Note: Checking if it's on HF. If not, we might need a direct link from PaddleOCR docs.
    # Searching HF for PP-DocLayoutV2 usually points to PaddlePaddle repos.
    # If it's not on HF directly as a standalone repo, it might be part of a larger collection or need a direct URL.
    # For now, I will assume it might be available or we'll need to fetch it via paddleocr tools.
    # Let's try to look for a specific Hugging Face repo for layout.

    # Based on search results, there is `PaddlePaddle/PP-DocLayoutV2`
    print("Downloading PP-DocLayoutV2...")
    try:
        layout_model_path = snapshot_download(
            repo_id="PaddlePaddle/PP-DocLayoutV2",
            local_dir=os.path.join(models_dir, "PP-DocLayoutV2"),
        )
        print(f"PP-DocLayoutV2 downloaded to: {layout_model_path}")
    except Exception as e:
        print(f"Failed to download PP-DocLayoutV2 from HF: {e}")
        print(
            "Please check if the repo ID is correct or if it requires authentication."
        )


if __name__ == "__main__":
    download_models()
