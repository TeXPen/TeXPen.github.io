import { MODEL_CONFIG } from './model';
import { GENERATION_CONFIG } from './generation';

export { MODEL_CONFIG, GENERATION_CONFIG };


export interface SessionConfig {
  device: string;
  dtype: string | { encoder_model: string; decoder_with_past_model: string; };
  encoder_model_file_name: string;
  decoder_model_file_name: string;
}

export function getSessionOptions(device: string, dtype: string): SessionConfig {
  // We’ll always choose explicit files per dtype
  if (dtype === 'fp16') {
    return {
      device,
      dtype: {
        encoder_model: 'fp32',
        decoder_with_past_model: 'fp16',
      },
      encoder_model_file_name: 'encoder_model_fp16.onnx', // or 'encoder_model.onnx' if you prefer mixed
      decoder_model_file_name: 'decoder_with_past_model_fp16.onnx',
    };
  }

  if (dtype === 'q8') {
    return {
      device,
      dtype: 'q8',
      encoder_model_file_name: 'encoder_model_int8.onnx',
      decoder_model_file_name: 'decoder_with_past_model_int8.onnx',
    };
  }

  // default: fp32 with KV cache
  return {
    device,
    dtype: 'fp32',
    encoder_model_file_name: 'encoder_model.onnx',
    decoder_model_file_name: 'decoder_with_past_model.onnx',
  };
}

export function getGenerationConfig(dtype: string, tokenizer: { pad_token_id: number; eos_token_id: number; bos_token_id: number; }) {
  return {
    max_new_tokens: GENERATION_CONFIG.MAX_NEW_TOKENS,
    do_sample: GENERATION_CONFIG.DO_SAMPLE,
    num_beams: GENERATION_CONFIG.NUM_BEAMS,
    pad_token_id: tokenizer.pad_token_id,
    eos_token_id: tokenizer.eos_token_id,
    bos_token_id: tokenizer.bos_token_id,
    decoder_start_token_id: 0,
    // Only apply repetition penalty for fp16 to prevent loops
    ...(dtype === 'fp16' ? { repetition_penalty: GENERATION_CONFIG.FP16_REPETITION_PENALTY } : {}),
  };
}
