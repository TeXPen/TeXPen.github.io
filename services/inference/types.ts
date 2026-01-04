import { PreTrainedModel, Tensor, PretrainedConfig } from '@huggingface/transformers';

export type Quantization = 'fp32' | 'fp16' | 'int8' | 'int4';
export const QUANTIZATION_OPTIONS: { value: Quantization; label: string }[] = [
  { value: 'fp32', label: 'FP32' },
  { value: 'fp16', label: 'FP16' },
  { value: 'int8', label: 'Int8' },
  { value: 'int4', label: 'Int4' },
];

export type PerformanceProfile = 'high_quality' | 'fast' | 'balanced' | 'low_memory' | 'custom';

export const PROFILE_OPTIONS: { value: PerformanceProfile; label: string; description?: string }[] = [
  { value: 'high_quality', label: 'Highest Quality', description: 'Best accuracy, uses FP32' },
  { value: 'fast', label: 'Fastest', description: 'Balanced speed/quality (FP16 Enc / FP32 Dec)' },
  { value: 'balanced', label: 'Balanced', description: 'Good for CPU (Int8)' },
  { value: 'low_memory', label: 'Lowest Memory', description: 'Smallest download (Int4)' },
];

export interface InferenceConfig {
  encoderModelUrl: string;
  decoderModelUrl: string;
  tokenizerUrl: string;
  imageSize: number;
  encoderInputName: string;
  decoderInputName: string;
  decoderOutputName: string;
  mean: number[];
  std: number[];
  invert: boolean;
  eosToken: string;
  bosToken: string;
  padToken: string;
  preferredProvider: 'webgpu' | 'wasm';
}

export interface InferenceResult {
  latex: string;
  candidates: string[];
  debugImage: string;
}

export interface SamplingOptions {
  do_sample?: boolean;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  num_beams?: number; // legacy numCandidates mapping
  onPreprocess?: (debugImage: string) => void;
}

export interface InferenceOptions {
  device?: 'webgpu' | 'wasm';
  modelId?: string;
  quantization?: Quantization;
  encoderQuantization?: Quantization;
  decoderQuantization?: Quantization;
}

export interface BeamState {
  tokens: number[];
  score: number;
  done: boolean;
  parentIndex: number;
}
export type Beam = BeamState; // Alias for backward compatibility if needed


// Helper alignment for transformers.js models which often lack precise types
export type VisionEncoder = (inputs: { pixel_values: Tensor }) => Promise<Record<string, Tensor>>;
export type Seq2SeqForward = (inputs: {
  encoder_outputs: unknown;
  decoder_input_ids: Tensor;
  pixel_values?: Tensor; // Added for validation bypass if needed
  use_cache?: boolean
}) => Promise<Record<string, Tensor>>;


export interface InferenceSessionShim {
  run: (feeds: Record<string, unknown>) => Promise<Record<string, Tensor>>;
  inputNames: string[];
  outputNames: string[];
}

export interface VisionEncoderDecoderModel extends PreTrainedModel {
  encoder?: VisionEncoder;
  forward: Seq2SeqForward;
  generate: (options: Record<string, unknown>) => Promise<number[][] | Tensor | unknown>;
  config: PretrainedConfig & { decoder?: Record<string, unknown>; d_model?: number; hidden_size?: number; decoder_attention_heads?: number; num_attention_heads?: number;[key: string]: unknown; };
  sessions: Record<string, InferenceSessionShim>;
  dispose: () => Promise<unknown[]>;
}
