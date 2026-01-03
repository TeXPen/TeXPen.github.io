import onnxruntime as ort
import os
import numpy as np

MODELS_DIR = os.path.join(os.getcwd(), "public", "models", "paddle")


def inspect_model(name, input_shape):
    model_path = os.path.join(MODELS_DIR, name)
    print(f"\n--- Inspecting {name} ---")
    try:
        session = ort.InferenceSession(model_path)
        print("Loaded.")

        input_name = session.get_inputs()[0].name
        print(f"Input Name: {input_name}")

        # Create dummy tensor
        # Shape: [1, 3, H, W]
        data = np.random.rand(*input_shape).astype(np.float32)

        # Run
        results = session.run(None, {input_name: data})

        output_name = session.get_outputs()[0].name
        print(f"Output Name: {output_name}")
        print(f"Output Shape: {results[0].shape}")

        # Verify shape details
        if "rec" in name:
            # Check last dim
            dim = results[0].shape[2]
            print(f"Rec Vocab+1 Size: {dim}")

    except Exception as e:
        print(f"Failed to test {name}: {e}")


if __name__ == "__main__":
    # Det: [1, 3, 640, 640]
    inspect_model("det.onnx", [1, 3, 640, 640])

    # Rec: [1, 3, 48, 320]
    inspect_model("rec.onnx", [1, 3, 48, 320])
