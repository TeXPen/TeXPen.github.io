import os
import torch
from transformers import AutoModel, AutoConfig
import numpy as np
import math
from torch.nn import functional as F


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = attn_bias.masked_fill(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_bias + attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=False)
    return attn_weight @ value


# Monkeypatch SDPA for ONNX export compatibility
torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention


def convert_vlm_components():
    models_dir = os.path.join(
        os.path.dirname(__file__), "../_local/models/PaddleOCR-VL"
    )
    output_dir = os.path.join(os.path.dirname(__file__), "../public/models/vlm")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading PaddleOCR-VL model...")
    config = AutoConfig.from_pretrained(models_dir, trust_remote_code=True)

    full_model = AutoModel.from_pretrained(
        models_dir, config=config, trust_remote_code=True, torch_dtype=torch.float32
    )
    full_model.eval()

    print("Converting Vision Components...")
    convert_vision_patch_embedding(full_model, output_dir)
    convert_vision_transformer(full_model, output_dir)
    convert_projector(full_model, output_dir)
    save_position_embeddings(full_model, output_dir)
    convert_text_embedding(full_model, output_dir)

    print("Converting LLM...")
    convert_llm(full_model, output_dir)


def convert_vision_patch_embedding(model, output_dir):
    print("  Exporting Patch Embedding...")
    patch_embed = model.visual.vision_model.embeddings.patch_embedding

    dummy_input = torch.randn(1, 3, 28, 28).float()

    torch.onnx.export(
        patch_embed,
        dummy_input,
        os.path.join(output_dir, "vision_patch_embed.onnx"),
        input_names=["pixel_values"],
        output_names=["patch_features"],
        dynamic_axes={
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "patch_features": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=14,
    )


def convert_vision_transformer(model, output_dir):
    print("  Exporting Vision Transformer...")
    encoder = model.visual.vision_model.encoder

    class VisionTransformerWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, inputs_embeds):
            batch, seq, _ = inputs_embeds.shape
            attention_mask = torch.zeros(batch, 1, seq, seq).to(inputs_embeds.device)

            outputs = self.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=False,
                cu_seqlens=None,
            )
            return outputs[0]

    wrapper = VisionTransformerWrapper(encoder)

    hidden_dim = model.config.vision_config.hidden_size
    dummy_input = torch.randn(1, 100, hidden_dim).float()

    torch.onnx.export(
        wrapper,
        dummy_input,
        os.path.join(output_dir, "vision_transformer.onnx"),
        input_names=["inputs_embeds"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "seq_len"},
            "last_hidden_state": {0: "batch", 1: "seq_len"},
        },
        opset_version=14,
    )


def convert_projector(model, output_dir):
    print("  Exporting Projector...")
    projector = model.mlp_AR

    class ProjectorWrapper(torch.nn.Module):
        def __init__(self, projector):
            super().__init__()
            self.projector = projector

        def forward(self, image_features):
            return self.projector(image_features, image_grid_thw=[])

    wrapper = ProjectorWrapper(projector)
    hidden_dim = model.config.vision_config.hidden_size
    dummy_input = torch.randn(1, 100, hidden_dim).float()

    torch.onnx.export(
        wrapper,
        dummy_input,
        os.path.join(output_dir, "vision_projector.onnx"),
        input_names=["image_features"],
        output_names=["projected_features"],
        dynamic_axes={
            "image_features": {0: "batch", 1: "seq_len"},
            "projected_features": {0: "batch", 1: "seq_len"},
        },
        opset_version=14,
    )


def save_position_embeddings(model, output_dir):
    print("  Saving Position Embeddings...")
    pos_embed = (
        model.visual.vision_model.embeddings.position_embedding.weight.detach()
        .cpu()
        .numpy()
    )
    np.save(os.path.join(output_dir, "pos_embed.npy"), pos_embed)
    print(f"  Saved pos_embed.npy shape: {pos_embed.shape}")


def convert_text_embedding(model, output_dir):
    print("  Exporting Text Embedding...")
    # Access the embedding layer: model.model.embed_tokens
    embed_tokens = model.model.embed_tokens

    dummy_input = torch.tensor([[0, 1, 2, 3, 4, 5]]).long()

    torch.onnx.export(
        embed_tokens,
        dummy_input,
        os.path.join(output_dir, "text_embed.onnx"),
        input_names=["input_ids"],
        output_names=["inputs_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "inputs_embeds": {0: "batch", 1: "seq_len"},
        },
        opset_version=14,
    )


def convert_llm(model, output_dir):
    print("  Exporting LLM (Ernie)...")

    class LLMWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model
            self.lm_head = model.lm_head

        def forward(self, inputs_embeds, attention_mask, position_ids):
            outputs = self.model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state
            logits = self.lm_head(hidden_states)
            return logits

    wrapper = LLMWrapper(model)

    hidden_size = model.config.hidden_size
    seq_len = 32
    dummy_embeds = torch.randn(1, seq_len, hidden_size).float()
    dummy_mask = torch.ones(1, seq_len).float()
    dummy_pos_ids = torch.arange(seq_len).unsqueeze(0).repeat(3, 1, 1).long()

    torch.onnx.export(
        wrapper,
        (dummy_embeds, dummy_mask, dummy_pos_ids),
        os.path.join(output_dir, "llm.onnx"),
        input_names=["inputs_embeds", "attention_mask", "position_ids"],
        output_names=["logits"],
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "position_ids": {1: "batch", 2: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=14,
    )


if __name__ == "__main__":
    convert_vlm_components()
