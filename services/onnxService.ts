import * as ort from 'onnxruntime-web';
import { ModelConfig } from '../types';
import { getModelFromCache, storeModelInCache } from './cacheService';

// 1. Setup ONNX Runtime Env
ort.env.wasm.wasmPaths = '/';
ort.env.wasm.numThreads = 4; 
ort.env.wasm.proxy = false;

// 2. Global Sessions
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

// 3. EXACT CONFIGURATION (448x448, Specific Mean/Std)
export const DEFAULT_CONFIG: ModelConfig = {
  encoderModelUrl: 'https://huggingface.co/OleehyO/TexTeller/resolve/main/encoder_model.onnx',
  decoderModelUrl: 'https://huggingface.co/OleehyO/TexTeller/resolve/main/decoder_model_merged.onnx',
  tokenizerUrl: 'https://huggingface.co/OleehyO/TexTeller/resolve/main/tokenizer.json',
  
  imageSize: 448, 
  mean: [0.9545467], 
  std: [0.15394445],
  
  encoderInputName: 'pixel_values',
  decoderInputName: 'input_ids',
  decoderOutputName: 'logits',
  invert: false, // Logic handled in useInkModel now
  eosToken: '</s>',
  bosToken: '<s>',
  padToken: '<pad>',
  preferredProvider: 'wasm' 
};

export const initModel = async (
  config: ModelConfig,
  onProgress?: (phase: string, progress: number) => void
): Promise<void> => {
  try {
    // --- Load Tokenizer ---
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

    // --- Load Models ---
    const options: ort.InferenceSession.SessionOptions = {
      executionProviders: [config.preferredProvider, 'wasm'],
      graphOptimizationLevel: 'all',
    };

    console.log(`Loading models (Image Size: ${config.imageSize})...`);
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

    console.log('Models Loaded. Ready for Inference.');
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
    // 1. Preprocess (Grayscale, 448x448, Specific Mean/Std)
    const pixelValues = await preprocessImage(image, config);

    // 2. Run Encoder
    const encoderFeeds = { [config.encoderInputName]: pixelValues };
    const encoderResults = await encoderSession.run(encoderFeeds);
    const encoderHiddenStates = encoderResults.last_hidden_state || encoderResults[Object.keys(encoderResults)[0]];

    // 3. Initialize Decoder
    let decoderInputIds = [getVocabId(config.bosToken)]; 
    const outputTokens: string[] = [];
    const maxSteps = 40; 

    // 4. Create KV Cache (Dimensions from config.json: 16 heads, 1024 hidden size)
    const dummyPast = createPastKeyValues(decoderSession, 16, 64);

    // 5. Decode Loop
    for (let i = 0; i < maxSteps; i++) {
      const inputTensor = new ort.Tensor(
        'int64',
        BigInt64Array.from(decoderInputIds.map(n => BigInt(n))),
        [1, decoderInputIds.length]
      );

      const decoderFeeds: Record<string, ort.Tensor> = {
        [config.decoderInputName]: inputTensor,
        'encoder_hidden_states': encoderHiddenStates,
        ...dummyPast
      };

      if (decoderSession.inputNames.includes('use_cache_branch')) {
        decoderFeeds['use_cache_branch'] = new ort.Tensor('bool', [false], [1]);
      }

      const decoderResults = await decoderSession.run(decoderFeeds);
      const logits = decoderResults[config.decoderOutputName]; 

      const [batch, seqLen, vocabSize] = logits.dims;
      const lastTokenOffset = (seqLen - 1) * vocabSize;
      const lastTokenLogits = logits.data.slice(lastTokenOffset, lastTokenOffset + vocabSize) as Float32Array;

      // Greedy Argmax
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
  const targetSize = config.imageSize; // 448

  const canvas = document.createElement('canvas');
  canvas.width = targetSize;
  canvas.height = targetSize;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Canvas context failed');

  const bitmap = await createImageBitmap(inputImageData);
  
  // We assume inputImageData is already correct (Black on White) from useInkModel
  // Just draw it on a white background to ensure size match and no alpha issues
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, targetSize, targetSize);
  ctx.drawImage(bitmap, 0, 0, targetSize, targetSize);

  // --- DEBUGGER START ---
  canvas.style.position = 'fixed';
  canvas.style.bottom = '0';
  canvas.style.left = '0';
  canvas.style.zIndex = '9999';
  canvas.style.border = '2px solid red';
  canvas.style.width = '200px'; 
  canvas.style.height = '200px';
  const existing = document.getElementById('debug-onnx-canvas');
  if (existing) existing.remove();
  canvas.id = 'debug-onnx-canvas';
  document.body.appendChild(canvas);
  // --- DEBUGGER END ---

  const resizedData = ctx.getImageData(0, 0, targetSize, targetSize);
  const { data } = resizedData;

  const floatData = new Float32Array(targetSize * targetSize);

  for (let i = 0; i < targetSize * targetSize; i++) {
    const rIdx = i * 4;
    
    // Note: Since we forced Black Ink on White BG in useInkModel:
    // r=0 is Ink, r=255 is Background.
    let r = data[rIdx] / 255.0;

    // We do NOT check config.invert here anymore.

    // Standard TexTeller Normalization
    const norm = (r - config.mean[0]) / config.std[0];

    floatData[i] = norm;
  }

  return new ort.Tensor('float32', floatData, [1, 1, targetSize, targetSize]);
};

// --- Helpers ---

const createPastKeyValues = (session: ort.InferenceSession, numHeads: number, headDim: number): Record<string, ort.Tensor> => {
  const feeds: Record<string, ort.Tensor> = {};
  const batchSize = 1;
  const seqLen = 0; 

  session.inputNames.forEach(name => {
    if (name.startsWith('past_key_values')) {
      feeds[name] = new ort.Tensor(
        'float32', 
        new Float32Array(0), 
        [batchSize, numHeads, seqLen, headDim]
      );
    }
  });
  return feeds;
};

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
  return text
    .replace(/ |Ġ/g, ' ')
    .replace(/<\/s>/g, '')
    .replace(/<s>/g, '')
    .trim();
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