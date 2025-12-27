import os
import torch
from transformers import AutoModel, AutoConfig


def inspect_model():
    models_dir = os.path.join(
        os.path.dirname(__file__), "../_local/models/PaddleOCR-VL"
    )

    print(f"Loading model from {models_dir}...")
    try:
        # Load config first
        config = AutoConfig.from_pretrained(models_dir, trust_remote_code=True)
        print("Config loaded.")

        # Load model structure (on CPU to save memory/time)
        model = AutoModel.from_pretrained(
            models_dir, config=config, trust_remote_code=True, torch_dtype=torch.float32
        )
        print("Model loaded successfully.")

        print("\nModel Structure:")
        print(model)

        print("\nModules:")
        for name, module in model.named_children():
            print(f"- {name}: {type(module)}")

    except Exception as e:
        print(f"Error loading model: {e}")


if __name__ == "__main__":
    inspect_model()
