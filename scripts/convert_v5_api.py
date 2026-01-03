import os
import paddle2onnx
from paddle2onnx.command import program2onnx

# Explicitly use the library function if possible, or subprocess if not exposed nicely.
# Checking help via introspection

TEMP_DIR = "_temp_paddle_conversion"
OUTPUT_DIR = "public/models/paddle"


def attempt_direct_conversion(name, subdir):
    model_dir = os.path.join(TEMP_DIR, subdir)
    model_file = os.path.join(model_dir, "inference.json")
    params_file = os.path.join(model_dir, "inference.pdiparams")

    save_file = os.path.join(OUTPUT_DIR, name)

    print(f"Converting {name} from {model_dir}...")

    if not os.path.exists(model_file):
        print(f"Error: {model_file} not found")
        return

    try:
        # direct API usage: paddle2onnx.export?
        # usually:
        # paddle2onnx.export_inference_model(model_dir, model_filename, params_filename, save_file, opset_version=11)
        # Note: save_file should be path to .onnx file?

        # Let's inspect library first? No time. Assuming export_inference_model exists or similar.
        # Actually, let's look at how command calls it.
        # But for now, try to load model file content if needed?

        # We will assume paddle2onnx.export works for PIR if installed version supports it.

        paddle2onnx.export_inference(
            model_dir=model_dir,
            model_filename="inference.json",
            params_filename="inference.pdiparams",
            save_file=save_file,
            opset_version=11,
            enable_onnx_checker=True,
        )
        print("Conversion Success!")

    except Exception as e:
        print(f"Failed via paddle2onnx lib: {e}")
        # import traceback
        # traceback.print_exc()


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    attempt_direct_conversion("det.onnx", "det")
    attempt_direct_conversion("rec.onnx", "rec")
