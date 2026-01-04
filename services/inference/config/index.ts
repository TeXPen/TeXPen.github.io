import { MODEL_CONFIG } from './model';
import { GENERATION_CONFIG } from './generation';
import { Quantization } from '../types';

export { MODEL_CONFIG, GENERATION_CONFIG };


export interface SessionConfig {
  device: string;
  dtype: string;
  encoder_model_file_name: string;
  decoder_model_file_name: string;
}

const QUANT_MODEL_MAP: Record<Quantization, { encoder: string; decoder: string }> = {
  fp32: {
    encoder: 'encoder_model.onnx',
    decoder: 'decoder_model_merged.onnx',
  },
  fp16: {
    encoder: 'encoder_model_fp16.onnx',
    decoder: 'decoder_model_merged.onnx', // FP16 decoder removed, use FP32
  },
  int8: {
    encoder: 'encoder_model_int8.onnx',
    decoder: 'decoder_model_merged_int8.onnx',
  },
  int4: {
    encoder: 'encoder_model_int4.onnx',
    decoder: 'decoder_model_merged_int4.onnx',
  },
};

export interface QuantizationConfig {
  encoder?: Quantization;
  decoder?: Quantization;
  overall?: Quantization;
}

export function getSessionOptions(
  device: string,
  quantization: QuantizationConfig | Quantization = 'int8'
): SessionConfig {
  const quant = typeof quantization === 'string'
    ? { overall: quantization }
    : quantization;

  const encoderQuant = quant.encoder || quant.overall || 'int8';
  const decoderQuant = quant.decoder || quant.overall || 'int8';

  const encoderFile = (QUANT_MODEL_MAP[encoderQuant] || QUANT_MODEL_MAP.fp32).encoder;
  const decoderFile = (QUANT_MODEL_MAP[decoderQuant] || QUANT_MODEL_MAP.fp32).decoder;

  return {
    device,
    // Force fp32 to prevent transformers.js from auto-selecting quantized models
    // which may not exist on the model repository
    dtype: 'fp32',
    encoder_model_file_name: encoderFile,
    decoder_model_file_name: decoderFile,
  };
}

export function getGenerationConfig(tokenizer: { pad_token_id: number; eos_token_id: number; bos_token_id: number; }) {
  return {
    max_new_tokens: GENERATION_CONFIG.MAX_NEW_TOKENS,
    do_sample: GENERATION_CONFIG.DO_SAMPLE,
    num_beams: GENERATION_CONFIG.NUM_BEAMS,
    pad_token_id: tokenizer.pad_token_id,
    eos_token_id: tokenizer.eos_token_id,
    bos_token_id: tokenizer.bos_token_id,
    decoder_start_token_id: 0,
  };
}
