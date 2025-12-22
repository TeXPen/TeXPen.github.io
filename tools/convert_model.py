import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import os
import shutil
import numpy as np
import sys
from typing import Optional, List, Union, Tuple
from torch.nn import CrossEntropyLoss


def convert_manual():
    model_id = "PaddlePaddle/PaddleOCR-VL"
    output_dir = "./onnx_output"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print(f"Loading processor for {model_id}...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    print(f"Loading model for {model_id}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    # Monkeypatch forward method to fix .numpy() issue
    print("Applying monkeypatch to model.forward...", flush=True)

    # Retrieve necessary class from the model's module
    model_module = sys.modules[model.__class__.__module__]
    PaddleOCRVLCausalLMOutputWithPast = model_module.PaddleOCRVLCausalLMOutputWithPast

    def patched_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        r"""
        Returns:
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                pixel_values = pixel_values.unsqueeze(0)
                siglip_position_ids = list()
                image_grid_hws = list()
                sample_indices = list()
                cu_seqlens = [0]

                pro = 0
                for idx, thw in enumerate(image_grid_thw):
                    # PATCHED: removed .numpy()
                    thw_tuple = tuple(thw.detach().cpu().tolist())
                    numel = np.prod(thw_tuple)
                    image_grid_hws.append(thw_tuple)
                    image_position_ids = torch.arange(numel) % np.prod(thw_tuple[1:])
                    siglip_position_ids.append(image_position_ids)
                    sample_indices.append(torch.full((numel,), idx, dtype=torch.int64))
                    cu_seqlens.append(cu_seqlens[-1] + numel)

                siglip_position_ids = torch.concat(siglip_position_ids, dim=0).to(
                    pixel_values.device
                )
                cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32).to(
                    pixel_values.device
                )
                sample_indices = torch.concat(sample_indices, dim=0).to(
                    pixel_values.device
                )

                vision_outputs = self.visual(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_hws,
                    position_ids=siglip_position_ids,
                    vision_return_embed_list=True,
                    interpolate_pos_encoding=True,
                    sample_indices=sample_indices,
                    cu_seqlens=cu_seqlens,
                    return_pooler_output=False,
                    use_rope=True,
                    window_size=-1,
                )
                image_embeds = vision_outputs.last_hidden_state

                image_embeds = self.mlp_AR(image_embeds, image_grid_thw)

                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                # image_embeds is a list of tensor, each tensor is a image feature,I want to concat them all into a tensor
                image_embeds = torch.cat(image_embeds, dim=0)
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )

                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
        # position_ids = None
        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (
            attention_mask is None or attention_mask.ndim == 2
        ):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return PaddleOCRVLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    # Assign patched method to class
    model.__class__.forward = patched_forward

    model.eval()

    print("Generating dummy inputs...", flush=True)
    dummy_image = Image.new("RGB", (384, 384), color="white")
    dummy_text = "<image>"

    inputs = processor(text=dummy_text, images=dummy_image, return_tensors="pt")

    # Manually fix input_ids to match the feature count (196) produced by the visual encoder
    # The processor seems to produce mismatched token counts for the dummy image
    image_token_id = model.config.image_token_id
    # We need 196 tokens as per the error message (features 196)
    inputs["input_ids"] = torch.full((1, 196), image_token_id, dtype=torch.long)
    inputs["attention_mask"] = torch.ones((1, 196), dtype=torch.long)

    # Inspect inputs to ensure we handle them all
    print(f"Input keys: {inputs.keys()}", flush=True)

    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
            # Explicitly pass arguments to the model
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

    print("Wrapping model for export...", flush=True)
    wrapped_model = ModelWrapper(model)

    # Dummy inputs for wrapper
    dummy_input_tuple = (
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["pixel_values"],
        inputs["image_grid_thw"],
    )

    output_file = os.path.join(output_dir, "model.onnx")

    print(f"Exporting to {output_file}...", flush=True)
    torch.onnx.export(
        wrapped_model,
        dummy_input_tuple,
        output_file,
        input_names=["input_ids", "attention_mask", "pixel_values", "image_grid_thw"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "image_grid_thw": {0: "batch"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=14,
        dynamo=False,
    )

    print("Saving processor...", flush=True)
    processor.save_pretrained(output_dir)

    print("Conversion complete.", flush=True)


if __name__ == "__main__":
    try:
        convert_manual()
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"FAILED: {e}")
