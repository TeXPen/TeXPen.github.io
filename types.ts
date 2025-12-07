// types.ts
import { InferenceSession } from 'onnxruntime-web';

export interface ModelConfig {
  encoderModelUrl: string;
  decoderModelUrl: string;
  tokenizerUrl: string;
  preprocessorConfigUrl?: string;
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
  preferredProvider: 'webgpu' | 'webgl' | 'wasm';
}

export interface Candidate {
  id: number;
  latex: string;
}

export interface HistoryItem {
  id: string;
  sessionId: string;
  latex: string;
  timestamp: number;
  source?: 'draw' | 'upload';
  imageData?: string; // Data URL for uploaded images
  versions?: HistoryItem[];
}

export type ModelStatus = 'loading' | 'ready' | 'error' | 'inferencing';

// Extend Window for MathJax and ONNX
declare global {
  interface Window {
    MathJax?: {
      tex: { inlineMath: string[][] };
      svg: { fontCache: string };
      startup: { typeset: boolean };
      typesetPromise: (nodes?: Element[]) => Promise<void>;
      typesetClear: (nodes?: Element[]) => void;
      tex2chtml: (tex: string) => HTMLElement;
    };
    ort: {
      env: {
        wasm: {
          wasmPaths: string;
        };
      };
      InferenceSession: typeof InferenceSession;
    };
  }
}