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
    # Use 224x224 to ensure patch size 14 divides cleanly (224/14 = 16)
    # This results in 16x16 = 256 patches/tokens.
    dummy_image = Image.new("RGB", (224, 224), color="white")
    dummy_text = "<image>"

    inputs = processor(text=dummy_text, images=dummy_image, return_tensors="pt")

    # Manually fix input_ids to match the feature count
    # 224x224 -> 16x16 patches -> 256 tokens
    image_token_id = model.config.image_token_id
    inputs["input_ids"] = torch.full((1, 256), image_token_id, dtype=torch.long)
    inputs["attention_mask"] = torch.ones((1, 256), dtype=torch.long)

    # Also update image_grid_thw dummy if it exists later in the script
    # We need to find where d_image_grid_thw is defined.

    # Create cleanup wrapper for tensor conversions
    def to_numpy(tensor):
        if hasattr(tensor, "detach"):
            return tensor.detach().cpu().numpy()
        return tensor

    # --- 1. Export Visual Encoder ---
    print("\n=== Exporting Visual Encoder ===", flush=True)

    print(f"Model keys: {model.__dict__.keys()}", flush=True)
    if hasattr(model, "model"):
        print(f"Submodel keys: {model.model.__dict__.keys()}", flush=True)

    class VisualEncoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            # Try to find visual component
            if hasattr(model, "visual"):
                self.model = model
            elif hasattr(model.model, "visual"):
                self.model = model.model
            elif hasattr(model, "vision_model"):
                self.model = model
                self.model.visual = model.vision_model  # Alias
            else:
                # Fallback: scan for likely candidates
                print("Scanning for visual module...", flush=True)
                for name, module in model.named_children():
                    print(f" - {name}: {type(module)}", flush=True)
                # Fail hard if not found but we need to see output first
                self.model = model

        def forward(self, pixel_values):
            # Based on model.forward logic for vision
            # We urge the vision model to run just the visual part
            # pixel_values: [1, 3, H, W]

            # Note: The original forward has complex logic for 'image_grid_thw' etc.
            # But for standard inference with 1 image, we can simplify or pre-compute.
            # However, this model uses 'image_grid_thw' to handle variable res/grids.
            # For simplicity in browser, we might want fixed size.

            # Let's try to capture the specific visual call:
            # self.visual(pixel_values, ...) -> image_embeds
            # Then self.mlp_AR(image_embeds, ...)

            # We need to reproduce the 'inputs_embeds is None' block from patched_forward
            # but ONLY return image_embeds.

            # Re-creating the essential parts of the logic:
            pixel_values = pixel_values.type(self.model.visual.dtype)
            pixel_values = pixel_values.unsqueeze(
                0
            )  # Add batch dim if missing? No, mostly input is [1, 3, H, W]
            # Actually input is usually [bp, 3, 384, 384] where bp is num_patches
            # But here we assume input 'pixel_values' is the standard [1, 3, H, W] passed to the processor?
            # Wait, processor returns [1, 3, H, W] or similar.

            # Let's inspect what the processor returns for dummy input
            # inputs['pixel_values'] shape is likely [N, 3, H, W]

            # We will use the exact logic from the forward patch but simplified

            # Hardcoded assumption for single image inference:
            # We need 'image_grid_thw'. We can make it an input or hardcode it if we fix size.
            # Let's make it an input for flexibility.
            pass

    # Actually, we can just extract the vision part as a separate module if possible,
    # but since it relies on 'paddle_ocr_vl' specific structures, strict tracing might be better.

    # Let's stick to the 'ModelWrapper' approach but only return image_embeds for encoder

    class VisualEncoderExport(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model  # Submodel (PaddleOCRVLModel)

        def forward(self, pixel_values, image_grid_thw):
            # Logic from patched_forward
            # pixel_values: [N, 3, H, W]
            # image_grid_thw: [1, 3] usually

            pixel_values = pixel_values.type(self.model.visual.dtype)
            # Ensure 4D
            if pixel_values.ndim == 3:
                pixel_values = pixel_values.unsqueeze(0)

            # We need to construct arguments for self.model.visual
            # ... (logic copied from patched_forward)

            # To avoid copying too much logic, we can rely on the fact that 'model.model'
            # might run this if we pass input_ids=None?
            # No, model.model wants input_ids usually.

            # START LOGIC COPY
            pixel_values = pixel_values.unsqueeze(0)  # [1, N, 3, H, W] ?
            # Logic says: pixel_values = pixel_values.unsqueeze(0) inside forward.
            # But inputs['pixel_values'] is [1, 3, H, W] usually.
            # Let's assume input to ONNX is [N, 3, H, W] (patches)

            siglip_position_ids = []
            image_grid_hws = []
            sample_indices = []
            cu_seqlens = [0]

            # We can't use 'list()' loops efficiently in ONNX tracing usually unless unrolled.
            # Since we trace, loop unrolling happens for the dummy input.
            # Verify if this works for dynamic shapes.

            # For 1 image, loop runs once.
            idx = 0
            thw = image_grid_thw[0]

            # thw_tuple = tuple(thw.detach().cpu().tolist()) # Cannot use .tolist() in trace easily?
            # Actually, for tracing with PyTorch, .tolist() is evaluated at trace time (constant).
            # This breaks dynamic shapes if 'thw' changes.
            # But 'image_grid_thw' determines the visual grid.
            # If we fix image size (e.g. 448x448), thw is constant.
            # For robustness, we'll assume fixed input size scheme or accept that valid trace depends on it.

            # ... Trying to be cleaner ...
            # Let's look at self.model.visual forward

            # Simplified for trace (assuming batch=1, and fixed grid logic or trace-compatible)
            # If we can't easily trace 'thw' logic, we might need a custom wrapper that
            # acts as "The Vision Encoder".

            # Let's try to rely on the existing code structure.

            # We will use the model.model.visual directly if possible?
            # But we also need 'mlp_AR'.

            pass

    # Redefine wrapper to be concrete
    class VisionPart(torch.nn.Module):
        def __init__(self, parent_model):
            super().__init__()
            # Find visual module
            if hasattr(parent_model, "visual"):
                self.visual = parent_model.visual
            elif hasattr(parent_model, "vision_model"):
                self.visual = parent_model.vision_model
            elif hasattr(parent_model, "model") and hasattr(
                parent_model.model, "visual"
            ):
                self.visual = parent_model.model.visual
            else:
                print(
                    "DEBUG: Children of parent_model:",
                    [n for n, _ in parent_model.named_children()],
                )
                if hasattr(parent_model, "model"):
                    print(
                        "DEBUG: Children of parent_model.model:",
                        [n for n, _ in parent_model.model.named_children()],
                    )
                # Try to look for 'visual' in children manually
                found = False
                for name, module in parent_model.named_children():
                    if name == "visual":
                        self.visual = module
                        found = True
                        break
                if not found:
                    raise AttributeError("Could not find 'visual' or 'vision_model'")

            # Find MLP AR (Projector)
            if hasattr(parent_model, "mlp_AR"):
                self.mlp_AR = parent_model.mlp_AR
            elif hasattr(parent_model, "model") and hasattr(
                parent_model.model, "mlp_AR"
            ):
                self.mlp_AR = parent_model.model.mlp_AR
            else:
                # Check children
                found = False
                for name, module in parent_model.named_children():
                    if name == "mlp_AR":
                        self.mlp_AR = module
                        found = True
                        break
                if not found:
                    raise AttributeError("Could not find 'mlp_AR' module")

            self.config = parent_model.config

        def forward(self, pixel_values, image_grid_thw):
            # Re-implementing the vision block from 'patched_forward'
            # Inputs: pixel_values [N_patches, 3, H, W], image_grid_thw [1, 3]

            # Note: inputs['pixel_values'] from processor is likely [N_patches, 3, H, W]

            # Inside patched_forward:
            # pixel_values = pixel_values.unsqueeze(0)  -> [1, N_patches, 3, H, W]
            pixel_values_unsqueezed = pixel_values.unsqueeze(0)

            # Grid logic - tricky to trace if dynamic.
            # We will assume specific grid behavior works or is traced for specific size.
            # If we assume standard size 384x384 or similar.

            # For trace compatibility, we might have to use torch ops instead of numpy/list
            # But 'image_grid_thw' is a tensor.

            # HACK: For the purpose of this task (WebGPU Inference),
            # we assume the JS side prepares 'image_grid_thw' correctly.
            # We execute the logic:

            siglip_position_ids_list = []
            image_grid_hws_list = []
            sample_indices_list = []
            cu_seqlens_list = [0]

            # Assume batch size 1 for image_grid_thw
            thw = image_grid_thw[0]

            # Convert to list for visual module (it expects list of tuples)
            # This is the hard part for ONNX export: passing lists of tuples is not standard ONNX.
            # We need to modify 'visual' to accept tensors if possible, or reliance on trace unrolling.

            # If we trace, 'thw' values at trace time become constants in the graph if we use .tolist().
            # VALIDATION: If we change image size effectively, we need re-export?
            # Usually yes for ViT with specific patch layouts.
            # So we will document: "Model exported for checking dynamic axes or fixed size".
            # We will try to leave it dynamic if 'thw' is tensor.

            # BUT: self.visual expects 'image_grid_thw' as list of tuples.
            # We simply can't pass that into ONNX.
            # So we rely on tracing baking in the structure.
            # We must ensure the JS side sends the SAME grid structure (e.g. 1 image).

            # ...

            # Pre-calculation for visual
            thw_cpu = thw.cpu()  # Force CPU for list conversion during trace?
            # In trace, .cpu() is recorded.

            # Let's perform the "SETUP" logic that was in the loop:
            # thw_tuple = tuple(thw.tolist()) -> This runs at trace time!

            # So the ONNX model will have HARDCODED grid sizes if we do this.
            # This is acceptable if we always use the same resolution (e.g. 448x448) in browser.

            # To allow dynamic resolution, we'd need to rewrite the underlying model code
            # to avoid Python lists/tuples and use Tensors. That is a huge task.
            # DECISION: Export for a fixed "Block" or dynamic shape but fixed structure (1 image).

            # The code does:
            # image_grid_hws.append(thw_tuple)

            # We will proceed with tracing. It will bake in the fact that we have 1 image.
            # It might bake in the resolution if not careful.
            # But pixel_values has dynamic axes.
            # If 'thw_tuple' is used to reshape/view, it might be fixed.

            # Let's use the code from patched_forward but adapted for single image

            # We need to reconstruct the inputs for self.visual

            # We will just call the visual directly?
            # No, 'visual' signature: (pixel_values, image_grid_thw=list_of_tuples, ...)

            # We will proceed with the trace and check if it works.

            # Replicating the loop logic for 1 item:
            thw_val = thw  # Tensor
            # We need to construct the args.

            # Logic from patched_forward lines 81-105:
            # We use the tensor arithmetic where possible.

            # We cannot easily fix the list input issue without modifying the library code deeper.
            # We will wrap the WHOLE thing and hope torch.onnx handles the list-unrolling for the sub-module.

            # ... Actually, to enable 'dynamic' input image size in ONNX,
            # we need the model to compute 'position_ids' based on input tensor shape, not baked constants.

            # Given constraints, we will aim for the 'trace guarantees structure' approach.
            # If the user provides a different image size, it might fail if shapes don't match baked constants.
            # We will define dynamic axes for pixel_values.

            # Let's try to implement the `VisionPart` carefully.

            return_embed_list = True

            # Recalculate everything inside forward to capture it in graph?
            # 'image_grid_thw' usage in visual is: `for thw in image_grid_thw:`

            # We will simply instantiate the visual part using the original classes if we can.

            # Let's try a diff approach:
            # We trace `model` but with `input_ids=None`.
            # But the logic in `patched_forward` (lines 76-141) handles vision execution.
            # If we call `model(input_ids=None, pixel_values=...)` it should run vision?
            # `patched_forward` line 76: `if inputs_embeds is None: inputs_embeds = ...`
            # It computes image embeddings, then combines.
            # We want JUST the image embeddings.

            # We will create a wrapper that mocks `embed_tokens` to return nothing
            # or simply extracts the `image_embeds` calculation code.

            tensor_pixel_values = pixel_values.unsqueeze(0)

            # We assume batch size 1
            thw = image_grid_thw[0]
            thw_tuple = (
                thw[0],
                thw[1],
                thw[2],
            )  # Tracing might capture individual items?

            # We'll use a hack: Use the original `patched_forward` logic but return early.

            # ... copy-paste logic ...

            # See patched_forward lines 87-118
            # We need to manually perform these steps because `self.visual` expects specific Python types (lists).

            # Since we can't easily change `self.visual` signature, we have to prepare the arguments it expects.
            # If we pass a list to `self.visual` during tracing, the tracer usually handles it if the list structure is constant.

            # So:
            image_grid_hws = [(thw[0], thw[1], thw[2])]  # List of Tensors?
            # Function expects list of tuples of ints/tensors.
            # If tensors, might work.

            # Calculate position ids
            # ...
            # To save complexity, I will just reference `model.model.visual` call
            # and let the tracer resolve the internal operations.

            # NOTE: Getting the exact arguments right for `visual` is key.

            # Constructing inputs for `visual`
            # We need to execute the prep code (lines 87-105) inside the wrapper.

            numel = torch.prod(thw)
            image_position_ids = torch.arange(
                numel, device=pixel_values.device
            ) % torch.prod(thw[1:])
            siglip_position_ids = image_position_ids.unsqueeze(0)  # [1, numel]

            sample_indices = torch.zeros(
                (numel,), dtype=torch.int64, device=pixel_values.device
            )
            pad = torch.tensor([0], dtype=torch.int32, device=pixel_values.device)
            numel_t = numel.reshape(1).to(dtype=torch.int32)
            cu_seqlens = torch.cat([pad, numel_t])

            # Cast inputs to visual
            # image_grid_hws must be list of tuples.
            # We will try passing tensors inside tuples.

            vision_outputs = self.visual(
                pixel_values=tensor_pixel_values,
                image_grid_thw=image_grid_thw,  # Pass raw tensor
                position_ids=siglip_position_ids,
                vision_return_embed_list=False,
                interpolate_pos_encoding=True,
                sample_indices=sample_indices,
                cu_seqlens=cu_seqlens,
                return_pooler_output=False,
                use_rope=True,
                window_size=-1,
            )
            image_embeds = vision_outputs.last_hidden_state

            # MLP AR
            image_embeds = self.mlp_AR(image_embeds, image_grid_thw)

            # Concat (usually just 1 tensor if 1 image)
            image_embeds = torch.cat(image_embeds, dim=0)

            return image_embeds

    vision_model = VisionPart(model)
    vision_model.eval()

    # Export Vision
    vision_out = os.path.join(output_dir, "visual_encoder.onnx")

    # Dummy inputs
    d_pixel_values = inputs["pixel_values"]  # [N, 3, H, W]
    d_image_grid_thw = inputs["image_grid_thw"]  # [1, 3]

    torch.onnx.export(
        vision_model,
        (d_pixel_values, d_image_grid_thw),
        vision_out,
        input_names=["pixel_values", "image_grid_thw"],
        output_names=["image_embeds"],
        dynamic_axes={
            "pixel_values": {0: "num_patches", 2: "height", 3: "width"},
            # "image_grid_thw": {0: "batch"}, # Keep grid dynamic?
            "image_embeds": {0: "num_tokens"},
        },
        opset_version=14,
    )
    print(f"Visual encoder exported to {vision_out}")

    # --- 2. Export Text Decoder ---
    print("\n=== Exporting Text Decoder ===", flush=True)

    # The decoder needs to accept:
    # - input_ids
    # - encoder_hidden_states (the image_embeds)
    # - past_key_values (cache)

    class TextDecoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.config = model.config

        def forward(
            self, input_ids, encoder_hidden_states, attention_mask, past_key_values=None
        ):
            # We need to bypass the 'visual' call in `patched_forward`
            # We can do this by passing 'inputs_embeds' directly?
            # 'patched_forward' lines 76-142 handle image embedding.
            # If we pass 'inputs_embeds', it skips that block.

            # HOWEVER, we want to mix text embeddings with image embeddings.
            # Standard LLaVA/VL logic:
            # - Input IDs contain <image> tokens.
            # - Embeddings for text are looked up.
            # - Embeddings for <image> are replaced by 'encoder_hidden_states'.

            # We need to implement this "Merge" logic inside the ONNX model
            # OR do it in JS.
            # Doing it in JS is flexible but slow/complex (looking up embeddings).
            # Doing it in ONNX is better.

            # If we use `patched_forward`, it expects `pixel_values` to trigger image logic.
            # But line 76: `if inputs_embeds is None:` -> computes text embeds.
            # Then line 141: `inputs_embeds = inputs_embeds.masked_scatter(...)`

            # PROPOSAL:
            # We pass `input_ids` (with <image> tokens) and `encoder_hidden_states`.
            # We modify `patched_forward` (or our wrapper) to:
            # 1. Compute text embeddings from `input_ids`.
            # 2. Perform the `masked_scatter` replacement using `encoder_hidden_states`.
            # 3. Pass to `self.model`.

            # Let's extract that logic.

            # 1. Embed text
            inputs_embeds = self.model.model.embed_tokens(input_ids)

            # 2. Merge Image Embeds
            # Assumption: input_ids contains 'image_token_id' where image goes.
            # encoder_hidden_states: [num_image_tokens, hidden_size]

            image_token_id = self.config.image_token_id
            mask = input_ids == image_token_id

            # We need to flatten/expand correctly.
            # inputs_embeds: [batch, seq, hidden]
            # mask: [batch, seq]

            # Safety check (optional in trace)
            # n_img = mask.sum()
            # if n_img != encoder_hidden_states.shape[0]: ...

            mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds)

            # Ensure types match
            encoder_hidden_states = encoder_hidden_states.to(inputs_embeds.dtype)

            # Replace
            inputs_embeds = inputs_embeds.masked_scatter(
                mask_expanded, encoder_hidden_states
            )

            # 3. Forward through LLM
            # We call the internal model which is usually a Llama/Qwen with standard signature
            # self.model.model( ... ) -> but that's the transparent one.
            # self.model(...) calls patched_forward.
            # We want 'self.model.model' (the transformer base) + 'self.model.lm_head'

            outputs = self.model.model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            hidden_states = outputs[0]
            logits = self.model.lm_head(hidden_states)

            return logits, outputs.past_key_values

    decoder_model = TextDecoderWrapper(model)
    decoder_model.eval()

    decoder_out = os.path.join(output_dir, "text_decoder.onnx")

    # Prepare dummy inputs for decoder
    # We need:
    # - input_ids: [1, seq_len] containing image placeholder
    # - encoder_hidden_states: [196, hidden_dim] (from dummy run)

    # Run vision once to get real shape/values
    with torch.no_grad():
        d_image_embeds = vision_model(d_pixel_values, d_image_grid_thw)

    print(f"Dummy Image Embeds Shape: {d_image_embeds.shape}")

    # input_ids from 'inputs' contains 196 image tokens.
    d_input_ids = inputs["input_ids"]
    d_attention_mask = inputs["attention_mask"]

    # Create past_key_values
    # Check config for num_layers, num_heads, head_dim
    # Usually: [num_layers, 2, batch, num_heads, seq_len, head_dim]
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads  # Approximated

    # Verify exact shape requirements for past_kv
    # For export, we usually start with empty past (seq_len=0) OR
    # just export the signature.
    # But usually providing a dummy past is required for tracing 'use_cache=True'.

    # If we want to support "First Run" (no past) and "Later Runs" (with past),
    # we typically export 2 models or 1 merged model with dynamic axes.
    # Merged model: Past is optional input? ONNX doesn't like optional inputs well.
    # Standard practice: Pass empty/zeros for first run.

    # Let's create dummy past
    past_seq_len = 0
    d_past_key_values = []
    # Shape: (batch, num_heads, past_seq_len, head_dim)
    # We use past_seq_len=1 for dummy to establish shapes?
    # Or 0? 0 often causes issues with some ops.
    # Let's try 0.

    # Actually, for 'merged' export, usually we use distinct inputs.
    # Simplest for now: Export 'no_past' model?
    # Plan asks for KV Cache.
    # Usage:
    # 1. Prefill (all tokens + image) -> get KV
    # 2. Decode (1 token + past KV) -> get new KV

    # We will export 'decoder_model_merged.onnx' style?
    # Or just one model that accepts Past.
    # If Past is empty tensor, it works as prefill.

    # Correct shape for Past in Llama usually:
    # (batch, num_heads, seq_len, head_dim)

    # NOTE: To simplify 'convert_model.py' changes, we will skip sophisticated "merged" ONNX handling (if/else inside graph)
    # and just export a model that expects past.
    # But wait, first run HAS NO past.
    # So we need to handle "past is None" or "past is empty".

    # TRICK: Use `dummy_past` of size 0.
    for _ in range(num_layers):
        k = torch.zeros(1, num_heads, 0, head_dim)
        v = torch.zeros(1, num_heads, 0, head_dim)
        d_past_key_values.append((k, v))

    # We need to flatten past_key_values for ONNX export (cannot pass list of tuples)
    # Wrapper needs to unpack.

    # Updating wrapper to unpack past
    class TextDecoderExport(TextDecoderWrapper):
        def forward(
            self,
            input_ids,
            encoder_hidden_states,
            attention_mask,
            *past_key_values_flat,
        ):
            # Reconstruct past_key_values list of tuples
            past_key_values = []
            group_size = 2
            for i in range(0, len(past_key_values_flat), group_size):
                past_key_values.append(
                    (past_key_values_flat[i], past_key_values_flat[i + 1])
                )

            # Call parent
            logits, new_past = super().forward(
                input_ids, encoder_hidden_states, attention_mask, past_key_values
            )

            # Flatten new_past
            new_past_flat = []
            for k, v in new_past:
                new_past_flat.extend([k, v])

            return logits, *new_past_flat

    final_decoder = TextDecoderExport(model)

    # Flatten dummy arguments
    flat_past = []
    for k, v in d_past_key_values:
        flat_past.extend([k, v])

    # Dynamic axes setup
    # input_ids: [1, seq_len]
    # encoder_hidden_states: [num_img_tokens, hidden]
    # past_key_values: [1, n_heads, past_seq, head_dim]

    input_names = ["input_ids", "encoder_hidden_states", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {1: "seq_len"},
        "encoder_hidden_states": {0: "num_img_tokens"},
        "attention_mask": {1: "total_seq_len"},
        "logits": {1: "seq_len"},
    }

    # Add past names
    for i in range(num_layers):
        input_names.extend([f"past_k_{i}", f"past_v_{i}"])
        output_names.extend([f"present_k_{i}", f"present_v_{i}"])
        dynamic_axes[f"past_k_{i}"] = {2: "past_seq_len"}
        dynamic_axes[f"past_v_{i}"] = {2: "past_seq_len"}
        dynamic_axes[f"present_k_{i}"] = {2: "total_seq_len"}
        dynamic_axes[f"present_v_{i}"] = {2: "total_seq_len"}

    torch.onnx.export(
        final_decoder,
        (d_input_ids, d_image_embeds, d_attention_mask, *flat_past),
        decoder_out,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
    )
    print(f"Text decoder exported to {decoder_out}")

    # --- 3. Quantization ---
    print("\n=== Quantizing Models ===", flush=True)
    from onnxruntime.quantization import quantize_dynamic, QuantType

    # Quantize Decoder (INT8) - Largest part
    decoder_quant = os.path.join(output_dir, "text_decoder_int8.onnx")
    print(f"Quantizing decoder to {decoder_quant}...", flush=True)
    quantize_dynamic(
        decoder_out,
        decoder_quant,
        weight_type=QuantType.QUInt8,
    )

    # Quantize Encoder (INT8) - Optional, visual encoder is usually smaller/sensitive
    # But for max savings:
    encoder_quant = os.path.join(output_dir, "visual_encoder_int8.onnx")
    print(f"Quantizing encoder to {encoder_quant}...", flush=True)
    quantize_dynamic(
        vision_out,
        encoder_quant,
        weight_type=QuantType.QUInt8,
    )

    print("Quantization complete.")

    processor.save_pretrained(output_dir)

    print("Conversion complete.", flush=True)


if __name__ == "__main__":
    try:
        convert_manual()
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"FAILED: {e}")
