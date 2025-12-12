import { Tensor } from "@huggingface/transformers";
import { VisionEncoderDecoderModel } from "../types";
import { logToWindow } from "../utils/debugUtils";
import { sliceTensorSequence } from "../utils/tensorUtils";

export interface DecoderInputs {
  pixel_values: Tensor;
  encoder_outputs: any;
  encoder_hidden_states: Tensor;
  decoder_input_ids: Tensor;
  past_key_values: Record<string, Tensor> | null;
  step: number;
}

export interface DecoderOutputs {
  logits: Tensor;
  pastKeyValues: Record<string, Tensor>;
}

export class DecoderRunner {
  constructor(private model: VisionEncoderDecoderModel) { }

  async run(inputs: DecoderInputs): Promise<DecoderOutputs> {
    const {
      encoder_hidden_states,
      decoder_input_ids,
      past_key_values,
      step,
    } = inputs;




    const sessions = this.model.sessions ?? {};
    const decoderSession = sessions['decoder_model_merged'] || sessions['decoder'];

    if (!decoderSession) {
      throw new Error(
        "No decoder session found (checked 'decoder_model_merged', 'decoder')."
      );
    }

    const inputNames = (decoderSession as any).inputNames || [];
    const runInputs: any = {
      encoder_hidden_states: encoder_hidden_states,
    };

    if (inputNames.includes("input_ids")) {
      runInputs.input_ids = decoder_input_ids;
    } else {
      runInputs.decoder_input_ids = decoder_input_ids;
    }

    if (past_key_values) {
      Object.assign(runInputs, past_key_values);
    }

    if (inputNames.includes("use_cache_branch")) {
      const useCacheVal = step > 0;
      runInputs.use_cache_branch = new Tensor("bool", [useCacheVal], [1]);
    }

    const batchSize = decoder_input_ids.dims[0];
    const cfg = this.model.config ?? {};
    const decoderCfg = (cfg['decoder'] as Record<string, unknown>) || cfg;

    // Fill missing past_key_values with empty tensors for step 0
    // This happens if inputs are missing OR if an upstream step produced 'batch 0' results that were dropped
    for (const name of inputNames) {
      if (name.startsWith("past_key_values.") && !runInputs[name]) {
        const numHeads = (decoderCfg['decoder_attention_heads'] as number) || (decoderCfg['num_attention_heads'] as number) || (cfg['num_attention_heads'] as number) || 16;
        const hiddenSize = (decoderCfg['d_model'] as number) || (decoderCfg['hidden_size'] as number) || (cfg['hidden_size'] as number) || 1024;
        let headDim = Math.floor(hiddenSize / numHeads);
        if (numHeads === 16 && headDim === 48) headDim = 64;

        // Initializing with size 0 helps ONNX Runtime understand these are empty/placeholder
        // rather than 'actual data of size 1'.
        const seqLen = 0;

        if (step === 0 && !runInputs[name]) {
          logToWindow(
            `[DecoderRunner] Creating dummy ${name} with seqLen: ${seqLen}`,
            `(heads: ${numHeads}, headDim: ${headDim})`
          );
        }

        runInputs[name] = new Tensor(
          "float32",
          new Float32Array(batchSize * numHeads * seqLen * headDim),
          [batchSize, numHeads, seqLen, headDim]
        );
      }
    }

    const outputs = await decoderSession.run(runInputs);

    const logits = outputs.logits || outputs.decoder_logits;
    if (!logits) {
      throw new Error("No logits returned from model.forward");
    }

    // Build flat PKV map
    const pkvFlat: Record<string, Tensor> = {};
    for (const key of Object.keys(outputs)) {
      if (key.startsWith("present.")) {
        const flatName = key.replace(/^present\./, "past_key_values.");
        let val = (outputs as any)[key] as Tensor;

        // Fallback for invalid (batch 0) outputs - common with some ONNX export/runtime edge cases
        if (val.dims && val.dims[0] === 0) {
          logToWindow(`[DecoderRunner] WARNING: Output ${key} has batch 0. Attempting fallback to input...`);
          if (past_key_values && past_key_values[flatName] && past_key_values[flatName].dims[0] > 0) {
            logToWindow(`[DecoderRunner] Fallback successful for ${key}. Reusing input tensor.`);
            val = past_key_values[flatName];
          } else {
            logToWindow(`[DecoderRunner] Fallback failed for ${key}. Input missing or invalid.`);
          }
        }

        pkvFlat[flatName] = val;
      }
    }

    if (Object.keys(pkvFlat).length === 0) {
      throw new Error(
        "Model did not expose key/value cache outputs (present.*)."
      );
    }

    // Transient Dummy Cleanup
    if (step === 0) {
      logToWindow("[DecoderRunner] Slicing dummy from present tensors...");
      for (const key of Object.keys(pkvFlat)) {
        const t = pkvFlat[key];
        const isDecoderKey = key.includes("decoder");

        if (isDecoderKey) {
          if (t.dims && t.dims.length > 2 && t.dims[2] > 1) {
            pkvFlat[key] = await sliceTensorSequence(t, 1);
          } else {
            // logToWindow(`[DecoderRunner] Skipping slice for ${key}: dims ${t.dims} too small or not matching.`);
          }
        }
      }
    }

    // Dispose outputs except logits and cache
    for (const key of Object.keys(outputs)) {
      if (
        key === "logits" ||
        key === "decoder_logits" ||
        key === "past_key_values" ||
        key.startsWith("present.")
      ) {
        continue;
      }
      const val = (outputs as any)[key];
      if (val && typeof val.dispose === "function") {
        val.dispose();
      }
    }

    return { logits: logits as Tensor, pastKeyValues: pkvFlat };
  }
}
