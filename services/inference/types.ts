import { PreTrainedModel, Tensor } from '@huggingface/transformers';

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
  dtype?: string;
  device?: 'webgpu' | 'wasm';
  modelId?: string;
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

export interface VisionEncoderDecoderModel extends PreTrainedModel {
  encoder?: VisionEncoder;
  forward: Seq2SeqForward;
  generate: (options: Record<string, unknown>) => Promise<unknown>;
  config: Record<string, unknown>;
  sessions?: Record<string, any>;
  dispose: () => Promise<unknown[]>;
}
