import os
from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process


def quantize_sota(input_path, output_path):
    print(f"Quantizing {input_path} -> {output_path}")

    # 1. Pre-processing (Graph Optimization)
    preprocessed_model_path = input_path.replace(".onnx", "_pre.onnx")
    print("Step 1: Running Graph Optimization (quant_pre_process)...")
    used_preprocessed = False
    try:
        # Check if file exists, if so delete it
        if os.path.exists(preprocessed_model_path):
            os.remove(preprocessed_model_path)

        quant_pre_process(input_path, preprocessed_model_path)
        print("Graph Optimization complete.")
        used_preprocessed = True
    except Exception as e:
        print(f"Pre-processing failed (skipping): {e}")
        preprocessed_model_path = input_path

    model_to_quantize = preprocessed_model_path if used_preprocessed else input_path

    # 2. Enhanced INT8 Dynamic (SoTA for accuracy/size balance)
    print("Step 2: Running Enhanced Dynamic INT8...")
    try:
        # "Better than naive":
        # - per_channel=True: Quantize weights per-channel (standard for accuracy)
        # - weight_type=QUInt8: Standard
        quantize_dynamic(
            model_input=model_to_quantize,
            model_output=output_path,
            weight_type=QuantType.QUInt8,
            per_channel=True,  # Improved accuracy
            reduce_range=False,
            extra_options={
                "MatMulConstBOnly": True  # Important for Transformer weights
            },
        )

        # Cleanup
        if used_preprocessed and os.path.exists(preprocessed_model_path):
            os.remove(preprocessed_model_path)

        original_size = os.path.getsize(input_path)
        new_size = os.path.getsize(output_path)
        print(
            f"Success! Reduced from {original_size / 1024 / 1024:.2f}MB to {new_size / 1024 / 1024:.2f}MB"
        )
    except Exception as e:
        print(f"Quantization failed: {e}")


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "../public/models/vlm")

    targets = [
        ("vision_transformer.onnx", "vision_transformer_q8.onnx"),
        ("llm_init.onnx", "llm_init_q8.onnx"),
        ("llm_with_past.onnx", "llm_with_past_q8.onnx"),
    ]

    print("--- Robust Model Quantizer (Pre-Process + Optimized INT8) ---")

    for src, dst in targets:
        src_path = os.path.join(base_dir, src)
        dst_path = os.path.join(base_dir, dst)

        if not os.path.exists(src_path):
            print(f"Skipping {src} (Source not found)")
            continue

        quantize_sota(src_path, dst_path)


if __name__ == "__main__":
    main()
