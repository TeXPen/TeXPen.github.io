
import { InferenceSession, Tensor, env } from "onnxruntime-web";
import { AutoTokenizer, PreTrainedTokenizer } from "@huggingface/transformers";
import { preprocessPaddleVL } from "./utils/paddleVLPreprocess";
import { MODEL_CONFIG } from "./config";
import { downloadManager } from "../downloader/DownloadManager";
import { VLMInferenceResult } from "./types";
import { paddleVLPostprocess } from "./utils/paddleVLPostprocess";

env.wasm.numThreads = 1; // WASM stability
env.wasm.wasmPaths = '/'; // Load from public root

// Helper to parse simple NPY files (float32 only)
function parseNpy(buffer: ArrayBuffer): Tensor {
  const magic = new Uint8Array(buffer, 0, 6);
  if (magic[0] !== 0x93 || String.fromCharCode(...magic.slice(1)) !== 'NUMPY') {
    throw new Error("Invalid NPY file");
  }
  const view = new DataView(buffer);
  const headerLen = view.getUint16(8, true);
  const offset = 10 + headerLen;
  // Assume float32 and known shape for simplicity from logs if parsing is hard
  // But let's try to find shape in header
  const headerStr = new TextDecoder().decode(new Uint8Array(buffer, 10, headerLen));
  // Header format: {'descr': '<f4', 'fortran_order': False, 'shape': (729, 1152), ...}
  const shapeMatch = headerStr.match(/'shape':\s*\(([^)]+)\)/);
  if (!shapeMatch) throw new Error("Could not parse shape from NPY header");

  const shape = shapeMatch[1].split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));

  // float32 data
  const data = new Float32Array(buffer, offset);
  return new Tensor('float32', data, shape);
}

// Argmax helper
function argmax(data: Float32Array): number {
  let maxVal = -Infinity;
  let maxIdx = -1;
  for (let i = 0; i < data.length; i++) {
    if (data[i] > maxVal) {
      maxVal = data[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

export class VLMInferenceEngine {
  private patchEmbedSession: InferenceSession | null = null;
  private visionTransformerSession: InferenceSession | null = null;
  private visionProjectorSession: InferenceSession | null = null;
  private textEmbedSession: InferenceSession | null = null;
  private llmSession: InferenceSession | null = null;

  private tokenizer: PreTrainedTokenizer | null = null;
  private posEmbed: Tensor | null = null;

  private initialized = false;

  public async init(onProgress?: (status: string, progress?: number) => void) {
    if (this.initialized) return;

    const sessionOptions: InferenceSession.SessionOptions = {
      executionProviders: ['wasm'], // Fallback to WASM for reliability
      executionMode: 'sequential',
      graphOptimizationLevel: 'all'
    };

    const loadModel = async (key: keyof typeof MODEL_CONFIG.VLM_COMPONENTS, sessionProp: 'patchEmbedSession' | 'visionTransformerSession' | 'visionProjectorSession' | 'textEmbedSession' | 'llmSession') => {
      const filename = MODEL_CONFIG.VLM_COMPONENTS[key];
      const origin = typeof window !== 'undefined' ? window.location.origin : self.location.origin;
      const path = `${origin}/models/vlm/${filename}`;

      if (onProgress) onProgress(`Loading ${key}...`, 0);

      try {
        // Determine if it's NPY
        if (filename.endsWith('.npy')) {
          const response = await fetch(path);
          const buffer = await response.arrayBuffer();
          this.posEmbed = parseNpy(buffer);
        } else {
          // Fetch model and potential external data
          const [modelResp, dataResp] = await Promise.all([
            fetch(path),
            fetch(`${path}.data`).then(r => r.ok ? r : null).catch(() => null)
          ]);

          if (!modelResp.ok) throw new Error(`Failed to fetch ${filename}`);
          const modelBuffer = await modelResp.arrayBuffer();

          const options: InferenceSession.SessionOptions = { ...sessionOptions };

          if (dataResp && dataResp.ok) {
            const dataBuffer = await dataResp.arrayBuffer();
            options.externalData = [
              {
                path: `${filename}.data`,
                data: new Uint8Array(dataBuffer)
              }
            ];
          }

          const session = await InferenceSession.create(new Uint8Array(modelBuffer), options);
          this[sessionProp] = session;
        }
      } catch (e) {
        console.error(`Failed to load ${filename}`, e);
        throw e;
      }
    };

    await Promise.all([
      loadModel('VISION_PATCH_EMBED', 'patchEmbedSession'),
      loadModel('VISION_TRANSFORMER', 'visionTransformerSession'),
      loadModel('VISION_PROJECTOR', 'visionProjectorSession'),
      loadModel('TEXT_EMBED', 'textEmbedSession'),
      loadModel('LLM', 'llmSession')
    ]);

    // Load pos_embed explicitly if not caught above (it is in VLM_COMPONENTS)
    // Check tokenizer
    if (onProgress) onProgress("Loading Tokenizer...", 90);
    this.tokenizer = await AutoTokenizer.from_pretrained(MODEL_CONFIG.PADDLE_VL_ID);

    this.initialized = true;
    if (onProgress) onProgress("Ready", 100);
  }

  public async inferVLM(imageBlob: Blob, prompt: string = "Describe this image."): Promise<VLMInferenceResult> {
    if (!this.initialized) throw new Error("Engine not initialized");

    const timings: Record<string, number> = {};
    const startTotal = performance.now();

    // 1. Preprocess Image
    const t0 = performance.now();
    const pixelValues = await preprocessPaddleVL(imageBlob); // [1, 3, 378, 378]
    timings['preprocess'] = performance.now() - t0;

    // 2. Vision Pipeline
    const t1 = performance.now();
    // Patch Embed
    const patchRes = await this.patchEmbedSession!.run({ 'pixel_values': pixelValues });
    const patchFeatures = patchRes['patch_features']; // [1, 729, 1152]

    // Add Pos Embed (Broadcasting addition in JS)
    // posEmbed is [729, 1152]. patchFeatures is [1, 729, 1152].
    // We can just loop and add.
    const patchData = patchFeatures.data as Float32Array;
    const posData = this.posEmbed!.data as Float32Array;
    const enrichedFeatures = new Float32Array(patchData.length);

    for (let i = 0; i < patchData.length; i++) {
      // patchData is flattened. posData repeats every 729*1152? No, posData is exact match for inner dims.
      // patchFeatures shape [1, 729, 1152]. size = 729*1152.
      // posEmbed shape [729, 1152]. size = 729*1152.
      enrichedFeatures[i] = patchData[i] + posData[i % posData.length];
    }
    const enrichedTensor = new Tensor('float32', enrichedFeatures, [1, 729, 1152]);

    // Transformer
    const transRes = await this.visionTransformerSession!.run({ 'inputs_embeds': enrichedTensor });
    const lastHidden = transRes['last_hidden_state']; // [1, 729, 1152]

    // Projector
    const projRes = await this.visionProjectorSession!.run({ 'image_features': lastHidden });
    const imageEmbeds = projRes['projected_features']; // [1, 729, hidden?]
    timings['vision_encoder'] = performance.now() - t1;

    // 3. Text Embedding & Concat
    // Tokenize Prompt
    // "<img>" tokens? The convert script prepared LLM, but how do we feed image?
    // Usually: <image_tokens> + <text_tokens>
    // Qwen-VL: Uses special tokens.
    // For this replication, we assume simplified concat: Image Embeds FIRST, then Text Embeds.
    // We do NOT use input_ids for the image part, we pass inputs_embeds to LLM.

    const { input_ids } = await this.tokenizer!(prompt, { return_tensor: false, padding: true, truncation: true });
    // Array of numbers
    const inputIdsTensor = new Tensor('int64', BigInt64Array.from((input_ids as number[]).map(BigInt)), [1, (input_ids as number[]).length]);

    // Text Embed
    const textEmbedRes = await this.textEmbedSession!.run({ 'input_ids': inputIdsTensor });
    const textEmbeds = textEmbedRes['inputs_embeds']; // [1, seq_len, hidden]

    // Concat: Image [1, 729, H] + Text [1, S, H] -> [1, 729+S, H]
    const imgData = imageEmbeds.data as Float32Array;
    const txtData = textEmbeds.data as Float32Array;
    const hiddenDim = imageEmbeds.dims[2];
    const combinedLen = (imageEmbeds.dims[1] + textEmbeds.dims[1]) * hiddenDim;
    const combinedData = new Float32Array(combinedLen);
    combinedData.set(imgData);
    combinedData.set(txtData, imgData.length);

    const combinedEmbeds = new Tensor('float32', combinedData, [1, imageEmbeds.dims[1] + textEmbeds.dims[1], hiddenDim]);

    // 4. Generation Loop
    const t2 = performance.now();
    const startLen = combinedEmbeds.dims[1];
    let currentEmbeds = combinedEmbeds;

    // Initial run (Pre-fill)
    // "inputs_embeds"
    // We need attention_mask and position_ids?
    // convert_vlm.py exported with: inputs_embeds, attention_mask, position_ids.

    const seqLen = currentEmbeds.dims[1];
    const attentionMask = new Tensor('int64', new BigInt64Array(seqLen).fill(1n), [1, seqLen]);
    const positionIds = new Tensor('int64', BigInt64Array.from({ length: seqLen }, (_, i) => BigInt(i)), [1, seqLen]);

    let feeds: Record<string, Tensor> = {
      'inputs_embeds': currentEmbeds,
      'attention_mask': attentionMask,
      'position_ids': positionIds
    };

    // We need KV cache support?
    // The export script `convert_llm` used:
    // wrapper forward(inputs_embeds, attention_mask, position_ids)
    // It DOES NOT seemingly export past_key_values in inputs?
    // Line 148: use_cache=False !!!
    // If use_cache=False, we cannot do efficient autoregression. We must re-feed everything.
    // This is slow (quadratic) but stateless and easier to implement for first static demo.
    // Since I blindly fixed the script, I kept use_cache=False.
    // So I must concat new token embed to input each step.

    const generatedIds: number[] = [];
    const MAX_NEW_TOKENS = 128;
    let pastIds = input_ids as number[];

    // Initial run
    const res = await this.llmSession!.run(feeds);
    let logits = res['logits']; // [1, seq, vocab]

    // Greedy decode last token
    let lastParams = logits.dims[1] * logits.dims[2];
    let lastLogits = (logits.data as Float32Array).slice(logits.data.length - logits.dims[2]);
    let nextId = argmax(lastLogits);
    generatedIds.push(nextId);
    pastIds.push(nextId);

    // Loop
    for (let i = 0; i < MAX_NEW_TOKENS; i++) {
      // Embed nextId
      const nextIdTensor = new Tensor('int64', BigInt64Array.from([BigInt(nextId)]), [1, 1]);
      const nextEmbedRes = await this.textEmbedSession!.run({ 'input_ids': nextIdTensor });
      const nextEmbed = nextEmbedRes['inputs_embeds']; // [1, 1, H]

      // Append to currentEmbeds?
      // Or re-run whole sequence?
      // If model has no cache, we might need to pass WHOLE sequence embeddings?
      // Or does it support partial forward if we don't mask?
      // Without cache, standard attention attends to all provided inputs.
      // So we must provide ALL inputs every time.

      // Re-assemble full embeddings
      // This is expensive. We re-allocate and copy.
      const oldData = currentEmbeds.data as Float32Array;
      const newData = nextEmbed.data as Float32Array;
      const totalLen = oldData.length + newData.length;
      const fullData = new Float32Array(totalLen);
      fullData.set(oldData);
      fullData.set(newData, oldData.length);

      currentEmbeds = new Tensor('float32', fullData, [1, currentEmbeds.dims[1] + 1, hiddenDim]);

      // Update mask/pos
      const newSeqLen = currentEmbeds.dims[1];
      const newMask = new Tensor('int64', new BigInt64Array(newSeqLen).fill(1n), [1, newSeqLen]);
      const newPos = new Tensor('int64', BigInt64Array.from({ length: newSeqLen }, (_, i) => BigInt(i)), [1, newSeqLen]); // 0 to N-1

      feeds = {
        'inputs_embeds': currentEmbeds,
        'attention_mask': newMask,
        'position_ids': newPos
      };

      const stepH = performance.now();
      const stepRes = await this.llmSession!.run(feeds);
      // console.log(`Step ${i} took ${performance.now() - stepH}ms`);

      logits = stepRes['logits'];
      lastLogits = (logits.data as Float32Array).slice(logits.data.length - logits.dims[2]);
      nextId = argmax(lastLogits);

      if (nextId === this.tokenizer!.eos_token_id) break;
      generatedIds.push(nextId);
    }
    timings['generation'] = performance.now() - t2;

    const markdown = this.tokenizer!.decode(generatedIds, { skip_special_tokens: true });

    return {
      markdown: paddleVLPostprocess(markdown),
      timings
    };
  }
}
