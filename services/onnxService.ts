import * as ort from 'onnxruntime-web';
import { ModelConfig } from '../types';
import { getModelFromCache, storeModelInCache } from './cacheService';

// Configure ONNX Runtime Web
ort.env.wasm.wasmPaths = '/';
ort.env.wasm.numThreads = 4;

let encoderSession: ort.InferenceSession | null = null;
let decoderSession: ort.InferenceSession | null = null;

interface Tokenizer {
  model: {
    vocab: { [token: string]: number };
    unk_token: string;
  };
}

let tokenizer: Tokenizer | null = null;
let reverseTokenizer: { [id: number]: string } = {};

export const DEFAULT_CONFIG: ModelConfig = {
  encoderModelUrl: 'https://huggingface.co/OleehyO/TexTeller/resolve/main/encoder_model.onnx',
  decoderModelUrl: 'https://huggingface.co/OleehyO/TexTeller/resolve/main/decoder_model_merged.onnx',
  tokenizerUrl: 'https://huggingface.co/OleehyO/TexTeller/resolve/main/tokenizer.json',
  imageSize: 384,
  encoderInputName: 'pixel_values',
  decoderInputName: 'input_ids',
  decoderOutputName: 'logits',
  // Model expects 1 channel, so we only need 1 mean/std value, but keeping structure is fine
  mean: [0.5], 
  std: [0.5],
  invert: false,
  eosToken: '</s>',
  bosToken: '<s>',
  padToken: '<pad>',
  preferredProvider: 'wasm' // Stick to WASM for stability given the strict kernel checks
};

export const initModel = async (
  config: ModelConfig,
  onProgress?: (phase: string, progress: number) => void
): Promise<void> => {
  try {
    // 1. Load Tokenizer
    if (onProgress) onProgress('Loading Tokenizer', 0);
    const cachedTokenizer = localStorage.getItem('tokenizer');
    if (cachedTokenizer) {
      tokenizer = JSON.parse(cachedTokenizer);
    } else {
      const tokenizerRes = await fetch(config.tokenizerUrl);
      if (!tokenizerRes.ok) throw new Error('Failed to load tokenizer.json');
      const tokenizerData = await tokenizerRes.json();
      tokenizer = tokenizerData;
      localStorage.setItem('tokenizer', JSON.stringify(tokenizerData));
    }

    if (tokenizer && Object.keys(reverseTokenizer).length === 0) {
      reverseTokenizer = Object.fromEntries(
        Object.entries(tokenizer.model.vocab).map(([key, value]) => [value, key])
      );
    }
    if (onProgress) onProgress('Loading Tokenizer', 100);

    // 2. Load Models
    const options: ort.InferenceSession.SessionOptions = {
      executionProviders: [config.preferredProvider, 'wasm'], 
      graphOptimizationLevel: 'all',
    };

    console.log(`Loading models...`);
    if (onProgress) onProgress('Loading Models', 0);

    const [encoderData, decoderData] = await Promise.all([
      fetchWithProgress(config.encoderModelUrl, 'Loading Encoder', onProgress),
      fetchWithProgress(config.decoderModelUrl, 'Loading Decoder', onProgress)
    ]);

    const [newEncoderSession, newDecoderSession] = await Promise.all([
      ort.InferenceSession.create(encoderData, options),
      ort.InferenceSession.create(decoderData, options)
    ]);

    encoderSession = newEncoderSession;
    decoderSession = newDecoderSession;

    console.log('Models Loaded!');
    if (onProgress) onProgress('Ready', 100);
  } catch (e) {
    console.error("Failed to load models:", e);
    throw e;
  }
};

export const runInference = async (
  image: ImageData,
  config: ModelConfig
): Promise<string> => {
  if (!encoderSession || !decoderSession || !tokenizer) {
    throw new Error('Models not initialized');
  }

  try {
    // 1. Preprocess (Converts to 1-Channel Grayscale Tensor)
    const pixelValues = await preprocessImage(image, config);

    // 2. Run Encoder
    const encoderFeeds = { [config.encoderInputName]: pixelValues };
    const encoderResults = await encoderSession.run(encoderFeeds);
    const encoderHiddenStates = encoderResults.last_hidden_state || encoderResults[Object.keys(encoderResults)[0]];

    // 3. Decoder Loop
    let decoderInputIds = [getVocabId(config.bosToken)]; 
    const outputTokens: string[] = [];
    const maxSteps = 40; 

    for (let i = 0; i < maxSteps; i++) {
      const inputTensor = new ort.Tensor(
        'int64',
        BigInt64Array.from(decoderInputIds.map(n => BigInt(n))),
        [1, decoderInputIds.length]
      );

      const decoderFeeds: Record<string, ort.Tensor> = {
        [config.decoderInputName]: inputTensor,
        'encoder_hidden_states': encoderHiddenStates
      };

      // Handle 'use_cache_branch' if model requires it (common in merged models)
      if (decoderSession.inputNames.includes('use_cache_branch')) {
        decoderFeeds['use_cache_branch'] = new ort.Tensor('bool', [false], [1]);
      }

      const decoderResults = await decoderSession.run(decoderFeeds);
      const logits = decoderResults[config.decoderOutputName]; 

      const [batch, seqLen, vocabSize] = logits.dims;
      const lastTokenOffset = (seqLen - 1) * vocabSize;
      const lastTokenLogits = logits.data.slice(lastTokenOffset, lastTokenOffset + vocabSize) as Float32Array;

      let maxIdx = 0;
      let maxVal = -Infinity;
      for (let j = 0; j < lastTokenLogits.length; j++) {
        if (lastTokenLogits[j] > maxVal) {
          maxVal = lastTokenLogits[j];
          maxIdx = j;
        }
      }

      const token = getTokenFromId(maxIdx);

      if (token === config.eosToken) break;

      outputTokens.push(token);
      decoderInputIds.push(maxIdx);
    }

    return cleanOutput(outputTokens.join(''));
  } catch (e) {
    console.error("Inference Failed:", e);
    throw e;
  }
};

const preprocessImage = async (inputImageData: ImageData, config: ModelConfig): Promise<ort.Tensor> => {
  const targetSize = config.imageSize; // 384

  // 1. Resize via Canvas
  const canvas = document.createElement('canvas');
  canvas.width = targetSize;
  canvas.height = targetSize;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Canvas context failed');

  const bitmap = await createImageBitmap(inputImageData);
  
  // Fill white background
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, targetSize, targetSize);
  ctx.drawImage(bitmap, 0, 0, targetSize, targetSize);

  const resizedData = ctx.getImageData(0, 0, targetSize, targetSize);
  const { data } = resizedData;

  // 2. Convert to SINGLE CHANNEL (Grayscale) Tensor
  // Size is just Width * Height (not * 3)
  const floatData = new Float32Array(targetSize * targetSize);

  for (let i = 0; i < targetSize * targetSize; i++) {
    const rIdx = i * 4;
    const gIdx = i * 4 + 1;
    const bIdx = i * 4 + 2;

    let r = data[rIdx] / 255.0;
    let g = data[gIdx] / 255.0;
    let b = data[bIdx] / 255.0;

    if (config.invert) {
      r = 1.0 - r;
      g = 1.0 - g;
      b = 1.0 - b;
    }

    // Convert to Grayscale
    const gray = (r + g + b) / 3.0;

    // Normalize
    // We utilize mean[0] and std[0] since we only have 1 channel
    const norm = (gray - config.mean[0]) / config.std[0];

    // Assign directly (No stride offset needed for 1 channel)
    floatData[i] = norm;
  }

  // Shape: [Batch=1, Channels=1, Height=384, Width=384]
  return new ort.Tensor('float32', floatData, [1, 1, targetSize, targetSize]);
};

// --- Helpers --- (No changes needed below, but included for completeness)

const fetchWithProgress = async (url: string, phase: string, onProgress?: (p: string, v: number) => void) => {
  const cachedData = await getModelFromCache(url);
  if (cachedData) {
    if (onProgress) onProgress(phase, 100);
    return cachedData;
  }

  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to load ${url}`);

  const contentLength = response.headers.get('content-length');
  const total = contentLength ? parseInt(contentLength, 10) : 0;
  let loaded = 0;

  const reader = response.body?.getReader();
  if (!reader) throw new Error('Response body is null');

  const chunks = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (total && onProgress) {
      onProgress(phase, Math.round((loaded / total) * 100));
    }
  }

  const blob = new Blob(chunks);
  const buffer = await blob.arrayBuffer();
  const data = new Uint8Array(buffer);
  await storeModelInCache(url, data);
  return data;
};

const getVocabId = (token: string): number => {
  if (!tokenizer) return 0;
  return tokenizer.model.vocab[token] || tokenizer.model.vocab[tokenizer.model.unk_token] || 0;
};

const getTokenFromId = (id: number): string => {
  return reverseTokenizer[id] || '';
};

const cleanOutput = (text: string): string => {
  return text.replace(/ |Ġ/g, ' ').replace(/<\/s>/g, '').replace(/<s>/g, '').trim();
};

export const generateVariations = (base: string): string[] => {
  return [base];
};

export const clearModelCache = async () => {
  try {
    const dbs = await window.indexedDB.databases();
    for (const db of dbs) {
      if (db.name) window.indexedDB.deleteDatabase(db.name);
    }
    localStorage.clear();
    console.log("Cache cleared.");
  } catch (e) {
    console.error("Error clearing cache:", e);
  }
};