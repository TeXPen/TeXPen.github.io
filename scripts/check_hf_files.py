from huggingface_hub import list_repo_files


def check_repo(repo_id):
    print(f"--- Files in {repo_id} ---")
    try:
        files = list_repo_files(repo_id)
        with open("hf_files.log", "w") as f:
            for file in files:
                f.write(file + "\n")
        print("Logged to hf_files.log")
    except Exception as e:
        print(f"Failed: {e}")


if __name__ == "__main__":
    # check_repo("marsena/paddleocr-onnx-models") # Check if it has any layout?
    # check_repo("PaddlePaddle/PP-StructureV3") # Failed usually

    # Try searching known ones
    check_repo("marsena/paddleocr-onnx-models")
    # check_repo("PaddlePaddle/PaddleOCR")
