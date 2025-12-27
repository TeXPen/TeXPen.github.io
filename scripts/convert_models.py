import os
import subprocess
import sys


def convert_layout_model():
    models_dir = os.path.join(os.path.dirname(__file__), "../_local/models")
    layout_model_dir = os.path.join(models_dir, "PP-DocLayoutV2")
    output_dir = os.path.join(os.path.dirname(__file__), "../public/models/layout")
    os.makedirs(output_dir, exist_ok=True)

    print("Converting PP-DocLayoutV2...")

    cmd = [
        "paddle2onnx",
        "--model_dir",
        layout_model_dir,
        "--model_filename",
        "inference.pdmodel",
        "--params_filename",
        "inference.pdiparams",
        "--save_file",
        os.path.join(output_dir, "model.onnx"),
        "--opset_version",
        "11",
        "--enable_onnx_checker",
        "True",
    ]

    model_filename = "inference.pdmodel"
    if not os.path.exists(os.path.join(layout_model_dir, model_filename)):
        files = os.listdir(layout_model_dir)
        pdmodel_files = [f for f in files if f.endswith(".pdmodel")]
        if pdmodel_files:
            model_filename = pdmodel_files[0]
        elif os.path.exists(os.path.join(layout_model_dir, "inference.json")):
            model_filename = "inference.json"
        else:
            print(
                "Error: No .pdmodel or inference.json file found in PP-DocLayoutV2 directory."
            )
            return

    cmd[cmd.index("--model_filename") + 1] = model_filename

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Conversion complete.")


if __name__ == "__main__":
    convert_layout_model()
