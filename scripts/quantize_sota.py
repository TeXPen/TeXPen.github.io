import os
import onnx
import sys
from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process

try:
    from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer

    HAS_INT4 = True
except ImportError:
    HAS_INT4 = False


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

    # 2. Try SOTA INT4
    if HAS_INT4:
        print("Step 2: Attempting Blockwise INT4 (SOTA)...")
        try:
            quantizer = MatMul4BitsQuantizer(
                model_or_path=model_to_quantize,
                block_size=128,
                is_symmetric=True,
                accuracy_level=None,
            )
            quantizer.process()
            quantizer.model.save_model_to_file(output_path)

            # Cleanup
            if used_preprocessed and os.path.exists(preprocessed_model_path):
                os.remove(preprocessed_model_path)

            original_size = os.path.getsize(input_path)
            new_size = os.path.getsize(output_path)
            print(
                f"Success! Reduced from {original_size / 1024 / 1024:.2f}MB to {new_size / 1024 / 1024:.2f}MB"
            )
            return
        except Exception as e:
            print(f"INT4 Quantization failed: {e}. Falling back to INT8...")

    # 3. Fallback to Enhanced INT8 Dynamic
    print("Step 2: Running Enhanced Dynamic INT8 (Fallback)...")
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
        ("vision_transformer.onnx", "vision_transformer_q4.onnx"),
        ("llm.onnx", "llm_q4.onnx"),
    ]

    print("--- Robust Model Quantizer (Pre-Process + Optimized) ---")
    print(f"INT4 Support Available: {HAS_INT4}")

    for src, dst in targets:
        src_path = os.path.join(base_dir, src)
        dst_path = os.path.join(base_dir, dst)

        if not os.path.exists(src_path):
            print(f"Skipping {src} (Source not found)")
            continue

        quantize_sota(src_path, dst_path)


if __name__ == "__main__":
    main()
