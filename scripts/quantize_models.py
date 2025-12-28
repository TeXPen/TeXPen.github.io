
import os
import glob
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_matmul_4bits

def quantize_model(input_path, output_path, method='int4'):
    print(f"Quantizing {input_path} to {output_path} using {method}...")
    
    if method == 'int4':
        # Blockwise INT4 quantization (SOTA for WebGPU late 2025)
        # This targets MatMuls specifically which is great for Transformers
        quantize_matmul_4bits(
            input_path,
            output_path,
            block_size=128, # Standard block size
            is_symmetric=True,
            accuracy_level=None # None = Optimized for performance/size
        )
    elif method == 'int8':
        # Dynamic INT8 - good fallback
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QUInt8
        )
    else:
        raise ValueError(f"Unknown method: {method}")
        
    print(f"Done. Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

def main():
    # Base directory for models
    base_dir = os.path.join(os.path.dirname(__file__), '../public/models/vlm')
    
    # Models to quantize (Large ones only)
    targets = [
        'vision_transformer.onnx',
        'llm.onnx'
    ]

    for filename in targets:
        input_path = os.path.join(base_dir, filename)
        if not os.path.exists(input_path):
            print(f"Skipping {filename} (not found)")
            continue
            
        # Create INT4 version
        output_path = os.path.join(base_dir, filename.replace('.onnx', '_q4.onnx'))
        quantize_model(input_path, output_path, method='int4')

if __name__ == "__main__":
    main()
