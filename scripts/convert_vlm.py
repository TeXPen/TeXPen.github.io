import os
import torch
from transformers import AutoModel, AutoConfig
import transformers.utils as hf_utils
import transformers.utils.generic as hf_generic
import numpy as np
import math
from torch.nn import functional as F

ONNX_OPSET_VERSION = 18


def _no_check_model_inputs(fn):
    return fn


hf_utils.check_model_inputs = _no_check_model_inputs
hf_generic.check_model_inputs = _no_check_model_inputs


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
        opset_version=ONNX_OPSET_VERSION,
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
        opset_version=ONNX_OPSET_VERSION,
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
        opset_version=ONNX_OPSET_VERSION,
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
        opset_version=ONNX_OPSET_VERSION,
    )


def _flatten_past_key_values(past_key_values):
    flat = []
    for idx, (k, v) in enumerate(past_key_values):
        flat.append((f"present.{idx}.key", k))
        flat.append((f"present.{idx}.value", v))
    return flat


def _past_input_names(num_layers):
    names = []
    for i in range(num_layers):
        names.append(f"past_key_values.{i}.key")
        names.append(f"past_key_values.{i}.value")
    return names


def convert_llm(model, output_dir):
    print("  Exporting LLM (Ernie) with cache...")

    class LLMInitWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model
            self.lm_head = model.lm_head

        def forward(self, inputs_embeds, attention_mask, position_ids):
            outputs = self.model(
                None,  # input_ids
                attention_mask,
                position_ids,
                None,  # past_key_values
                inputs_embeds,
                None,  # cache_position
                True,  # use_cache
                False,  # output_attentions
            )
            logits = self.lm_head(outputs.last_hidden_state)
            past = outputs.past_key_values
            return (logits, *[t for _, t in _flatten_past_key_values(past)])

    class LLMWithPastWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model
            self.lm_head = model.lm_head

        def forward(self, inputs_embeds, attention_mask, position_ids, *past_key_values):
            # Rebuild past_key_values tuple
            num_layers = len(past_key_values) // 2
            past = []
            for i in range(num_layers):
                past.append((past_key_values[2 * i], past_key_values[2 * i + 1]))

            outputs = self.model(
                None,  # input_ids
                attention_mask,
                position_ids,
                tuple(past),
                inputs_embeds,
                None,  # cache_position
                True,  # use_cache
                False,  # output_attentions
            )
            logits = self.lm_head(outputs.last_hidden_state)
            past_out = outputs.past_key_values
            return (logits, *[t for _, t in _flatten_past_key_values(past_out)])

    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads

    seq_len = 32
    past_seq = 16

    dummy_embeds = torch.randn(1, seq_len, hidden_size).float()
    dummy_mask = torch.ones(1, seq_len).float()
    dummy_pos_ids = torch.arange(seq_len).unsqueeze(0).repeat(3, 1, 1).long()

    past_key_values = []
    for _ in range(num_layers):
        k = torch.zeros(1, num_heads, past_seq, head_dim).float()
        v = torch.zeros(1, num_heads, past_seq, head_dim).float()
        past_key_values.extend([k, v])

    # Init model (no past inputs)
    init_wrapper = LLMInitWrapper(model)
    init_output_names = ["logits"] + _past_input_names(num_layers)
    torch.onnx.export(
        init_wrapper,
        (dummy_embeds, dummy_mask, dummy_pos_ids),
        os.path.join(output_dir, "llm_init.onnx"),
        input_names=["inputs_embeds", "attention_mask", "position_ids"],
        output_names=init_output_names,
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "position_ids": {1: "batch", 2: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
            **{name: {0: "batch", 2: "past_seq"} for name in _past_input_names(num_layers)},
        },
        opset_version=ONNX_OPSET_VERSION,
        dynamo=False,
    )

    # Cached model (with past inputs)
    with_past_wrapper = LLMWithPastWrapper(model)
    with_past_output_names = ["logits"] + _past_input_names(num_layers)
    torch.onnx.export(
        with_past_wrapper,
        (dummy_embeds, dummy_mask, dummy_pos_ids, *past_key_values),
        os.path.join(output_dir, "llm_with_past.onnx"),
        input_names=["inputs_embeds", "attention_mask", "position_ids"] + _past_input_names(num_layers),
        output_names=with_past_output_names,
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "position_ids": {1: "batch", 2: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
            **{name: {0: "batch", 2: "past_seq"} for name in _past_input_names(num_layers)},
        },
        opset_version=ONNX_OPSET_VERSION,
        dynamo=False,
    )


if __name__ == "__main__":
    convert_vlm_components()
