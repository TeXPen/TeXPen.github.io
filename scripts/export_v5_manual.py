import paddle
import os

# Ensure dynamic mode
# paddle.disable_static() # Explicitly ensure
# Default is dynamic.

TEMP_DIR = "_temp_paddle_conversion"
OUTPUT_DIR = "public/models/paddle"


def export_model(name, subdir):
    model_dir = os.path.join(TEMP_DIR, subdir)
    prefix = os.path.join(model_dir, "inference")

    print(f"Loading {name} from {prefix} ...")

    try:
        # Load as Layer (translated)
        model = paddle.jit.load(prefix)
        model.eval()  # Ensure eval mode
        print("Model loaded successfully.")

        save_path = os.path.join(OUTPUT_DIR, name)

        # Define Input Spec
        if "det" in name:
            # Det input: [Batch, 3, H, W]
            input_spec = [
                paddle.static.InputSpec(
                    shape=[None, 3, -1, -1], dtype="float32", name="x"
                )
            ]
        elif "rec" in name:
            # Rec input: [Batch, 3, 48, W]
            # Some rec models have fixed W=320, some dynamic. V5 is likely dynamic W.
            input_spec = [
                paddle.static.InputSpec(
                    shape=[None, 3, 48, -1], dtype="float32", name="x"
                )
            ]
        else:
            input_spec = None

        print(f"Exporting to {save_path} ...")

        with paddle.no_grad():
            paddle.onnx.export(
                model, save_path, input_spec=input_spec, opset_version=11
            )

        print("Export success.")

    except Exception as e:
        print(f"Failed to export {name}: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # We assume models are already downloaded in _temp by previous script runs
    # If not, run convert_paddle.py first (which fails at conversion but downloads).

    export_model("det", "det")
    export_model("rec", "rec")
