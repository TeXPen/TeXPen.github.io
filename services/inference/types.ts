import { PreTrainedModel, PreTrainedTokenizer, Tensor } from '@huggingface/transformers';

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
  preferredProvider: 'webgpu' | 'wasm' | 'webgl';
}

export interface InferenceResult {
  latex: string;
  candidates: string[];
  debugImage: string;
}

export interface InferenceOptions {
  dtype?: string;
  device?: 'webgpu' | 'wasm' | 'webgl';
  modelId?: string;
}

export interface Beam {
  tokens: number[];
  score: number;
  done: boolean;
}

// Helper alignment for transformers.js models which often lack precise types
export type VisionEncoder = (inputs: { pixel_values: Tensor }) => Promise<any>;
export type Seq2SeqForward = (inputs: {
  encoder_outputs: any;
  decoder_input_ids: Tensor;
  pixel_values?: Tensor; // Added for validation bypass if needed
  use_cache?: boolean
}) => Promise<any>;

export interface VisionEncoderDecoderModel extends PreTrainedModel {
  encoder?: VisionEncoder;
  forward: Seq2SeqForward;
  generate: (options: any) => Promise<any>;
}
