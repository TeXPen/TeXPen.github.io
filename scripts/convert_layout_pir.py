import os
import paddle
import paddle2onnx


def convert_layout_pir():
    models_dir = os.path.join(os.path.dirname(__file__), "../_local/models")
    layout_model_dir = os.path.join(models_dir, "PP-DocLayoutV2")
    output_dir = os.path.join(os.path.dirname(__file__), "../public/models/layout")
    os.makedirs(output_dir, exist_ok=True)

    model_file = os.path.join(layout_model_dir, "inference.json")
    params_file = os.path.join(layout_model_dir, "inference.pdiparams")
    save_file = os.path.join(output_dir, "model.onnx")

    print(f"Attempting to load PIR model from: {model_file}")

    try:
        # Enable PIR in Paddle (if needed, usually auto-detected or env var)
        # paddle.set_flags({"FLAGS_enable_pir_api": 1})

        # Load the inference model
        # For PIR, we might need to use create_predictor or paddle.jit.load

        # Method 1: paddle.jit.translate (usually for dygraph -> static)

        # Method 2: paddle2onnx command line failed.

        # Method 3: Load as Predictor and try to get program?
        from paddle.inference import Config, create_predictor

        config = Config(model_file, params_file)
        config.enable_use_gpu(100, 0)
        config.switch_ir_optim(True)
        # config.enable_memory_optim()
        predictor = create_predictor(config)

        print("Predictor created successfully. Model loaded.")

        # But we can't export predictor directly to ONNX easily without paddle2onnx.

        # Let's try paddle2onnx with different parameters or updated version check.
        print(f"Paddle2ONNX version: {paddle2onnx.__version__}")

    except Exception as e:
        print(f"Failed to load/convert: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    convert_layout_pir()
