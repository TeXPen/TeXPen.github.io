import os
import paddle2onnx
import logging


def convert_layout_debug():
    models_dir = os.path.join(os.path.dirname(__file__), "../_local/models")
    layout_model_dir = os.path.join(models_dir, "PP-DocLayoutV2")
    output_dir = os.path.join(os.path.dirname(__file__), "../public/models/layout")
    os.makedirs(output_dir, exist_ok=True)

    model_file = os.path.join(layout_model_dir, "inference.json")
    params_file = os.path.join(layout_model_dir, "inference.pdiparams")
    save_file = os.path.join(output_dir, "model.onnx")

    print(f"Converting model: {model_file}")

    try:
        paddle2onnx.export_inference_model(
            model_filename=model_file,
            params_filename=params_file,
            save_file=save_file,
            opset_version=11,
            enable_onnx_checker=True,
        )
        print("Conversion successful!")
    except Exception as e:
        print(f"Conversion failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    convert_layout_debug()
