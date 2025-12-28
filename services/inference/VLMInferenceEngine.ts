
import { InferenceSession, Tensor, env } from "onnxruntime-web";
import { AutoTokenizer, PreTrainedTokenizer } from "@huggingface/transformers";
import { preprocessPaddleVL } from "./utils/paddleVLPreprocess";
import { MODEL_CONFIG } from "./config";
import { downloadManager } from "../downloader/DownloadManager";
import { VLMInferenceResult, TokenCallback } from "./types";
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
  private loading = false;
  private sizeCache: Partial<Record<keyof typeof MODEL_CONFIG.VLM_COMPONENTS, number>> = {};
  private initPromise: Promise<void> | null = null;

  private sessionOptions: InferenceSession.SessionOptions = {
    executionProviders: ['webgpu', 'wasm'], // Use WebGPU if available
    executionMode: 'sequential',
    graphOptimizationLevel: 'all'
  };

  private async loadModel(
    key: keyof typeof MODEL_CONFIG.VLM_COMPONENTS,
    sessionProp?: 'patchEmbedSession' | 'visionTransformerSession' | 'visionProjectorSession' | 'textEmbedSession' | 'llmSession',
    providers: string[] = ['wasm'],
    onProgress?: (msg: string) => void
  ) {
    if (sessionProp && this[sessionProp]) return; // Already loaded

    const baseFilename = MODEL_CONFIG.VLM_COMPONENTS[key];
    let filename: string = baseFilename;

    // Check if this component should be quantized
    // Typically we only quantize the heavy lifters: Transformer and LLM
    if (MODEL_CONFIG.QUANTIZED && (key === 'VISION_TRANSFORMER' || key === 'LLM')) {
      filename = baseFilename.replace('.onnx', MODEL_CONFIG.QUANTIZED_SUFFIX);
    }

    const origin = typeof window !== 'undefined' ? window.location.origin : self.location.origin;
    const path = `${origin}/models/vlm/${filename}`;

    if (onProgress) onProgress(`Loading ${key}...`);

    try {
      // Fetch model and potential external data
      let [modelResp, dataResp] = await Promise.all([
        fetch(path),
        fetch(`${path}.data`).then(r => r.ok ? r : null).catch(() => null)
      ]);

      if (!modelResp.ok) throw new Error(`Failed to fetch ${filename}`);
      let modelBuffer: ArrayBuffer | null = await modelResp.arrayBuffer();

      let dataBuffer: ArrayBuffer | null = null;
      if (dataResp && dataResp.ok) {
        dataBuffer = await dataResp.arrayBuffer();
      }

      // Helper to create session with fallback
      const createSession = async (providers: string[]): Promise<InferenceSession> => {
        const options: InferenceSession.SessionOptions = {
          ...this.sessionOptions,
          executionProviders: providers
        };

        // Clone buffers for this attempt to avoid detachment issues if the attempt fails
        // but the buffer was transferred/detached by ORT.
        const modelBufferClone = modelBuffer!.slice(0);
        let externalDataClone: { path: string, data: Uint8Array }[] | undefined;

        if (dataBuffer) {
          const dataBufferClone = dataBuffer.slice(0);
          externalDataClone = [
            {
              path: `${filename}.data`,
              data: new Uint8Array(dataBufferClone)
            }
          ];
          options.externalData = externalDataClone;
        }

        return await InferenceSession.create(new Uint8Array(modelBufferClone), options);
      };

      try {
        console.log(`[VLM] Attempting to load ${key} with providers: ${providers.join(',')}`);
        const session = await createSession(providers);
        this[sessionProp] = session;
        console.log(`[VLM] Loaded ${key} on ${providers[0]}`); // Assumption: first is primary
      } catch (gpuError) {
        // If we tried GPU and failed, fallback to CPU
        if (providers.includes('webgpu') || providers.includes('webgl')) {
          console.warn(`[VLM] Failed to load ${key} on GPU. Falling back to CPU (wasm). Error:`, gpuError);

          // Retry with fresh buffers (cloned inside createSession)
          const session = await createSession(['wasm']);

          this[sessionProp] = session;
          console.log(`[VLM] Loaded ${key} on CPU (fallback)`);
        } else {
          throw gpuError; // Already CPU or other fatal error
        }
      }

      // Cleanup to help GC (we don't need the originals anymore after all attempts)
      modelBuffer = null;
      dataBuffer = null;
    } catch (e) {
      console.error(`Failed to load ${filename}`, e);
      throw e;
    }
  }

  private async releaseSession(sessionProp: 'patchEmbedSession' | 'visionTransformerSession' | 'visionProjectorSession' | 'textEmbedSession' | 'llmSession') {
    if (this[sessionProp]) {
      try {
        const session = this[sessionProp] as any;
        if (session && typeof session.release === 'function') {
          await session.release();
        }
      } catch (e) { console.warn(`Failed to release ${sessionProp}`, e); }
      this[sessionProp] = null;

      // Force wait for GC?
      await new Promise(r => setTimeout(r, 50));
    }
  }

  // Strategy definitions
  // 1. ALL_GPU: All models on WebGPU (Most aggressive, highest VRAM)
  // 2. ALL_GPU_NO_EMBED: Vision + LLM on GPU, Text Embed on CPU (Save ~300-500MB VRAM)
  // 3. LLM_GPU: LLM + Text Embed on GPU, Vision on CPU (Standard "Balanced", saves ~1.8GB VRAM)
  // 4. LLM_ONLY_GPU: LLM on GPU, Vision + Text Embed on CPU (Saves slightly more VRAM than #3)
  // 5. CPU_ONLY: Everything on CPU (Failsafe)
  private strategies: ('ALL_GPU' | 'ALL_GPU_NO_EMBED' | 'LLM_GPU' | 'LLM_ONLY_GPU' | 'CPU_ONLY')[] =
    ['ALL_GPU', 'ALL_GPU_NO_EMBED', 'LLM_GPU', 'LLM_ONLY_GPU', 'CPU_ONLY'];

  private currentStrategyIndex = 0; // Default to ALL_GPU

  public get currentStrategy(): 'ALL_GPU' | 'ALL_GPU_NO_EMBED' | 'LLM_GPU' | 'LLM_ONLY_GPU' | 'CPU_ONLY' {
    return this.strategies[this.currentStrategyIndex];
  }

  private async fetchContentLength(path: string): Promise<number | null> {
    try {
      const head = await fetch(path, { method: 'HEAD' });
      if (head.ok) {
        const len = head.headers.get('content-length');
        if (len) return Number.parseInt(len, 10);
      }
    } catch {
      // Ignore and try range request.
    }

    try {
      const range = await fetch(path, { headers: { Range: 'bytes=0-0' } });
      if (range.ok) {
        const cr = range.headers.get('content-range');
        if (cr) {
          const total = cr.split('/')[1];
          if (total) return Number.parseInt(total, 10);
        }
      }
    } catch {
      // Ignore, caller will use size hints.
    }

    return null;
  }

  private async getComponentSizeBytes(key: keyof typeof MODEL_CONFIG.VLM_COMPONENTS): Promise<number> {
    if (this.sizeCache[key]) return this.sizeCache[key]!;

    const baseFilename = MODEL_CONFIG.VLM_COMPONENTS[key];
    let filename: string = baseFilename;
    if (MODEL_CONFIG.QUANTIZED && (key === 'VISION_TRANSFORMER' || key === 'LLM')) {
      filename = baseFilename.replace('.onnx', MODEL_CONFIG.QUANTIZED_SUFFIX);
    }

    const origin = typeof window !== 'undefined' ? window.location.origin : self.location.origin;
    const path = `${origin}/models/vlm/${filename}`;

    const sizeHints: Record<string, number> = {
      'vision_patch_embed.onnx': 564,
      'vision_transformer.onnx': 2179125,
      'vision_projector.onnx': 24047,
      'text_embed.onnx': 313,
      'llm.onnx': 2079423,
      'pos_embed.npy': 3359360,

      // Quantized Hints (Approx 50% of FP32 or less for INT4)
      'vision_transformer_q4.onnx': 1089562,
      'llm_q4.onnx': 1039711
    };
    const dataHints: Record<string, number> = {
      'vision_patch_embed.onnx.data': 2775040,
      'vision_transformer.onnx.data': 1647222784,
      'vision_projector.onnx.data': 103874560,
      'text_embed.onnx.data': 423624704,
      'llm.onnx.data': 1443037184,

      // Quantized Data Hints (Approx 4-bit vs 32-bit = ~1/8th size theoretically, but usually less dramatic due to overhead/metadata)
      'vision_transformer_q4.onnx.data': 205902848, // ~1/8th
      'llm_q4.onnx.data': 180379648 // ~1/8th
    };

    let size = await this.fetchContentLength(path);
    if (size === null) {
      size = sizeHints[filename] ?? sizeHints[baseFilename] ?? 64 * 1024 * 1024;
    }

    // Account for external data if present.
    const dataPath = `${path}.data`;
    const dataSize = await this.fetchContentLength(dataPath);
    if (dataSize !== null) {
      size += dataSize;
    } else {
      const dataHint = dataHints[`${filename}.data`];
      if (dataHint) size += dataHint;
    }

    this.sizeCache[key] = size;
    return size;
  }

  private estimateGpuBudgetBytes(adapter: GPUAdapter): number {
    const maxStorage = adapter.limits.maxStorageBufferBindingSize || 0;
    const maxBuffer = adapter.limits.maxBufferSize || 0;
    const singleBufferLimit = Math.max(maxStorage, maxBuffer);

    // Heuristic: treat per-buffer limit as a proxy for total usable VRAM.
    if (singleBufferLimit >= 512 * 1024 * 1024) {
      return Math.floor(singleBufferLimit);
    }

    return Math.floor(singleBufferLimit * 0.95);
  }

  private async buildStrategyPlan() {
    // Deprecated in favor of fixed 5-stage granular strategy
    // We rely on the external "retry-reload" loop to find the right strategy
    return [];
  }

  public async init(onProgress?: (status: string, progress?: number) => void, onCheckpoint?: (phase: string) => void) {
    if (this.initialized) {
      if (onProgress) onProgress("Ready", 100);
      return;
    }

    // If initialization is already in progress, wait for it.
    if (this.initPromise) {
      return this.initPromise;
    }

    // Start new initialization
    this.initPromise = (async () => {
      this.loading = true;
      try {
        // Ensure clean state before starting IF NOT already initialized
        // This dispose() is only for hard resets/recovery. 
        // Normal preloads or multiple calls while loading are handled above.
        await this.dispose();

        // Simplify Init Logic: Just use current strategy
        await this.initStrategy(null, onProgress, onCheckpoint);


        if (onProgress) onProgress("Loading Tokenizer...", 90);
        this.tokenizer = await AutoTokenizer.from_pretrained(MODEL_CONFIG.PADDLE_VL_ID);

        // Load Pos Embed (small NPY)
        const filename = MODEL_CONFIG.VLM_COMPONENTS.POS_EMBED;
        const origin = typeof window !== 'undefined' ? window.location.origin : self.location.origin;
        const path = `${origin}/models/vlm/${filename}`;
        const response = await fetch(path);
        const buffer = await response.arrayBuffer();
        this.posEmbed = parseNpy(buffer);

        this.initialized = true;
        if (onProgress) onProgress("Ready", 100);
      } catch (e) {
        console.error("[VLM] Fatal initialization error:", e);
        throw e;
      } finally {
        this.loading = false;
        this.initPromise = null;
      }
    })();

    return this.initPromise;
  }

  private async initStrategy(
    // We ignore the passed strategy argument now as we use internal state
    _ignoredResultStrat?: any,
    onProgress?: (status: string, progress?: number) => void,
    onCheckpoint?: (phase: string) => void
  ) {
    const strat = this.currentStrategy;
    console.log(`[VLM] Initializing with Active Strategy (${this.currentStrategyIndex + 1}/${this.strategies.length}): ${strat}`);

    const gpu = ['webgpu', 'wasm'];
    const cpu = ['wasm'];

    let visionProviders = gpu;
    let textEmbedProviders = gpu;
    let llmProviders = gpu;

    switch (strat) {
      case 'ALL_GPU':
        // Everything GPU
        break;
      case 'ALL_GPU_NO_EMBED':
        // TextEmbed on CPU
        textEmbedProviders = cpu;
        break;
      case 'LLM_GPU':
        // Vision on CPU
        visionProviders = cpu;
        break;
      case 'LLM_ONLY_GPU':
        // Vision and TextEmbed on CPU
        visionProviders = cpu;
        textEmbedProviders = cpu;
        break;
      case 'CPU_ONLY':
        visionProviders = cpu;
        textEmbedProviders = cpu;
        llmProviders = cpu;
        break;
    }

    // Load models sequentially - PRIORITIZING LLM FIRST (Fail Fast)

    // 1. Text Embed (Small, but needed for LLM)
    if (onCheckpoint) onCheckpoint('LOADING_TEXT_EMBED');
    await this.loadModel('TEXT_EMBED', 'textEmbedSession', textEmbedProviders, onProgress);
    await new Promise(r => setTimeout(r, 50));

    // 2. LLM (Largest Model)
    // If this crashes, we know GPU can't handle the LLM at all -> CPU_ONLY
    if (onCheckpoint) onCheckpoint('LOADING_LLM');
    await this.loadModel('LLM', 'llmSession', llmProviders, onProgress);
    await new Promise(r => setTimeout(r, 50));

    // 3. Vision Components (Large)
    // If this crashes, LLM fit but Vision didn't -> LLM_GPU (Vision on CPU)
    if (onCheckpoint) onCheckpoint('LOADING_VISION');
    await this.loadModel('VISION_PATCH_EMBED', 'patchEmbedSession', visionProviders, onProgress);
    await new Promise(r => setTimeout(r, 50));

    await this.loadModel('VISION_TRANSFORMER', 'visionTransformerSession', visionProviders, onProgress);
    await new Promise(r => setTimeout(r, 50));

    await this.loadModel('VISION_PROJECTOR', 'visionProjectorSession', visionProviders, onProgress);
    await new Promise(r => setTimeout(r, 50));

    if (onCheckpoint) onCheckpoint('READY');

  }

  public setStrategyIndex(index: number) {
    if (index >= 0 && index < this.strategies.length) {
      this.currentStrategyIndex = index;
    }
  }

  public getStrategyIndex(): number {
    return this.currentStrategyIndex;
  }

  // New property to track inference status
  private inferenceInProgress: boolean = false;

  // Helper to get inference lock (simple boolean for now)
  private getInferenceLock(): boolean {
    return this.inferenceInProgress;
  }

  public async runInference(imageBlob: Blob, prompt: string = "Describe this image.", onToken?: TokenCallback, signal?: AbortSignal): Promise<VLMInferenceResult> {
    const lock = this.getInferenceLock();
    if (lock) throw new Error("Inference already in progress");

    this.inferenceInProgress = true;
    try {
      return await this.executePipeline(imageBlob, prompt, onToken, signal);
    } catch (error) {
      if (error instanceof Error && error.message === "Aborted") {
        console.log("[VLM] Inference Aborted");
        throw error;
      }
      console.error(`[VLM] Inference Failed:`, error);
      const err = error instanceof Error ? error : new Error(String(error));

      // Check for fatal WASM/Environment errors
      if (err.message && (err.message.includes("Aborted()") || err.message.includes("valid external Instance"))) {
        throw new Error("FATAL_RELOAD_NEEDED");
      }

      // Check for GPU OOM/Crash
      if (this.isGpuError(error)) {
        console.warn(`[VLM] GPU Error detected (${error}). Requesting Fatal Reload.`);
        throw new Error("FATAL_RELOAD_NEEDED");
      }

      throw error;
    } finally {
      this.inferenceInProgress = false;
    }
  }

  private isGpuError(e: unknown): boolean {
    let msg = "";
    if (e instanceof Error) {
      msg = e.message;
    } else if (typeof e === 'string') {
      msg = e;
    } else {
      msg = String(e);
    }
    msg = msg.toLowerCase();

    if (msg.includes("createbuffer") ||
      msg.includes("out of memory") ||
      msg.includes("context lost") ||
      msg.includes("device lost") ||
      msg.includes("too large")) {
      return true;
    }

    // Treat numeric error codes (like WASM pointers or exit codes) as GPU/Resource errors
    if (typeof e === 'number') return true;
    if (typeof e === 'string' && /^\d+$/.test(e.trim())) return true;

    return false;
  }

  private async executePipeline(imageBlob: Blob, prompt: string, onToken?: TokenCallback, signal?: AbortSignal): Promise<VLMInferenceResult> {
    if (!this.initialized) throw new Error("Engine not initialized");

    const timings: Record<string, number> = {};
    const t0 = performance.now();

    // 1. Preprocess Image
    if (signal?.aborted) throw new Error("Aborted");
    const pixelValues = await preprocessPaddleVL(imageBlob);
    timings['preprocess'] = performance.now() - t0;

    // 2. Vision Pipeline
    const t1 = performance.now();

    // Run Vision Pipeline
    // Sessions are already loaded (either GPU or CPU fallback)
    if (signal?.aborted) throw new Error("Aborted");
    const imageEmbeds = await this.runVisionPipeline(pixelValues);

    timings['vision_encoder'] = performance.now() - t1;

    // 3. LLM Pipeline
    const t2 = performance.now();

    // Run LLM Pipeline
    if (signal?.aborted) throw new Error("Aborted");
    const markdown = await this.runLLMPipeline(imageEmbeds, prompt, onToken, signal);

    timings['generation'] = performance.now() - t2;

    return {
      markdown: paddleVLPostprocess(markdown),
      timings
    };
  }

  private async runVisionPipeline(pixelValues: Tensor): Promise<Tensor> {
    // 1. Patch Embed
    const patchRes = await this.patchEmbedSession!.run({ 'pixel_values': pixelValues });
    const patchFeatures = patchRes['patch_features'];

    // 2. Add Pos Embed (Still on CPU for now as it's a simple addition)
    // TODO: Move this to a WebGPU shader or custom op for true zero-copy
    const patchData = patchFeatures.data as Float32Array;
    const posData = this.posEmbed!.data as Float32Array;
    const enrichedFeatures = new Float32Array(patchData.length);
    for (let i = 0; i < patchData.length; i++) {
      enrichedFeatures[i] = patchData[i] + posData[i % posData.length];
    }
    const enrichedTensor = new Tensor('float32', enrichedFeatures, [1, 729, 1152]);

    // 3. Transformer & Projector with IO-Binding
    // We want the output of Transformer to stay on GPU for the Projector
    const transRes = await this.visionTransformerSession!.run(
      { 'inputs_embeds': enrichedTensor },
      { preferredOutputLocation: 'gpu-buffer' } as any
    );
    const lastHidden = transRes['last_hidden_state'];

    // Projector runs on the GPU buffer directly
    const projRes = await this.visionProjectorSession!.run(
      { 'image_features': lastHidden },
      { preferredOutputLocation: 'gpu-buffer' } as any
    );
    return projRes['projected_features'];
  }

  private async runLLMPipeline(imageEmbeds: Tensor, prompt: string, onToken?: TokenCallback, signal?: AbortSignal): Promise<string> {
    // 3. Text Embedding & Concat with Prompt Template
    // Template: <s>User: <|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>{prompt}\nAssistant: 
    const fullPrompt = `<s>User: <|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>${prompt}\nAssistant: `;
    const { input_ids } = await this.tokenizer!(fullPrompt, { return_tensor: false, padding: false, truncation: true });
    const ids = input_ids as number[];

    // Identify placeholder ID to correctly splice image embeddings
    const placeholderRes = await this.tokenizer!("<|IMAGE_PLACEHOLDER|>", { return_tensor: false, add_special_tokens: false });
    const placeholderId = (placeholderRes.input_ids as number[])[0];
    const placeholderIdx = ids.indexOf(placeholderId);

    const inputIdsTensor = new Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
    const textEmbedRes = await this.textEmbedSession!.run({ 'input_ids': inputIdsTensor });
    const textEmbeds = textEmbedRes['inputs_embeds'];

    const hiddenDim = imageEmbeds.dims[2];
    const imgLen = imageEmbeds.dims[1];
    const txtLen = textEmbeds.dims[1];

    let combinedEmbeds: Tensor;

    if (placeholderIdx !== -1) {
      // Splice image embeddings into the placeholder position
      const txtData = await textEmbeds.getData() as Float32Array;
      const imgData = await imageEmbeds.getData() as Float32Array;

      const combinedLen = (txtLen - 1 + imgLen) * hiddenDim;
      const combinedData = new Float32Array(combinedLen);

      // Prefix (0 to placeholderIdx)
      const prefixCount = placeholderIdx * hiddenDim;
      combinedData.set(txtData.subarray(0, prefixCount));

      // Image (replace placeholder)
      combinedData.set(imgData, prefixCount);

      // Suffix (placeholderIdx + 1 to end)
      const suffixStart = (placeholderIdx + 1) * hiddenDim;
      combinedData.set(txtData.subarray(suffixStart), prefixCount + imgData.length);

      combinedEmbeds = new Tensor('float32', combinedData, [1, txtLen - 1 + imgLen, hiddenDim]);
      console.log(`[VLM] Injected ${imgLen} image tokens at position ${placeholderIdx}. Sequence length: ${txtLen - 1 + imgLen}`);
    } else {
      // Fallback: prepend if placeholder not found (shouldn't happen with our template)
      const imgData = await imageEmbeds.getData() as Float32Array;
      const txtData = await textEmbeds.getData() as Float32Array;
      const combinedData = new Float32Array(imgData.length + txtData.length);
      combinedData.set(imgData);
      combinedData.set(txtData, imgData.length);
      combinedEmbeds = new Tensor('float32', combinedData, [1, imgLen + txtLen, hiddenDim]);
      console.warn("[VLM] Placeholder not found in prompt tokens, fallback to prepending.");
    }

    // 4. Generation Loop
    let currentEmbeds = combinedEmbeds;
    const generatedIds: number[] = [];
    const MAX_NEW_TOKENS = 512; // Increased for longer document responses

    console.log(`[VLM] Starting LLM Generation Loop. Input Tokens: ${combinedEmbeds.dims[1]}`);

    for (let i = 0; i < MAX_NEW_TOKENS; i++) {
      if (signal?.aborted) throw new Error("Aborted");

      const seqLen = currentEmbeds.dims[1];
      const attentionMask = new Tensor('int64', new BigInt64Array(seqLen).fill(1n), [1, seqLen]);
      const positionIds = new Tensor('int64', BigInt64Array.from({ length: seqLen }, (_, i) => BigInt(i)), [1, seqLen]);

      const feeds = {
        'inputs_embeds': currentEmbeds,
        'attention_mask': attentionMask,
        'position_ids': positionIds
      };

      const stepT0 = performance.now();
      const res = await this.llmSession!.run(feeds);
      const logits = res['logits'];
      const logitsData = await logits.getData() as Float32Array;
      const lastLogits = logitsData.slice(logitsData.length - logits.dims[2]);
      const nextId = argmax(lastLogits);

      const stepTime = performance.now() - stepT0;
      if (i % 10 === 0) {
        console.log(`[VLM] Gen Step ${i}: ${stepTime.toFixed(1)}ms. Token: ${nextId}`);
      }

      if (nextId === this.tokenizer!.eos_token_id) break;
      generatedIds.push(nextId);

      // Call streaming callback
      if (onToken) {
        const decoded = this.tokenizer!.decode([nextId], { skip_special_tokens: true });
        const fullSoFar = this.tokenizer!.decode(generatedIds, { skip_special_tokens: true });
        onToken(decoded, fullSoFar);
      }

      // Append next token embedding
      const nextIdTensor = new Tensor('int64', BigInt64Array.from([BigInt(nextId)]), [1, 1]);
      const nextEmbedRes = await this.textEmbedSession!.run({ 'input_ids': nextIdTensor });
      const nextEmbed = nextEmbedRes['inputs_embeds'];

      const oldData = await currentEmbeds.getData() as Float32Array;
      const newData = await nextEmbed.getData() as Float32Array;
      const fullData = new Float32Array(oldData.length + newData.length);
      fullData.set(oldData);
      fullData.set(newData, oldData.length);

      const prevEmbeds = currentEmbeds;
      currentEmbeds = new Tensor('float32', fullData, [1, currentEmbeds.dims[1] + 1, hiddenDim]);

      // Cleanup intermediate tensors in loop
      if (prevEmbeds !== combinedEmbeds) {
        // Only dispose if it's not the original combinedEmbeds we use as starting point
        // In ORT, disposing a tensor used as input to a finished run is safe.
        // Wait, currentEmbeds is a NEW tensor every loop.
        (prevEmbeds as any).dispose?.();
      }
      (nextIdTensor as any).dispose?.();
      (nextEmbed as any).dispose?.();
      (attentionMask as any).dispose?.();
      (positionIds as any).dispose?.();
      (logits as any).dispose?.();
    }

    return this.tokenizer!.decode(generatedIds, { skip_special_tokens: true });
  }

  public async dispose() {
    if (this.patchEmbedSession) await this.patchEmbedSession.release();
    if (this.visionTransformerSession) await this.visionTransformerSession.release();
    if (this.visionProjectorSession) await this.visionProjectorSession.release();
    if (this.textEmbedSession) await this.textEmbedSession.release();
    if (this.llmSession) await this.llmSession.release();

    this.patchEmbedSession = null;
    this.visionTransformerSession = null;
    this.visionProjectorSession = null;
    this.textEmbedSession = null;
    this.llmSession = null;

    this.initialized = false;
    this.posEmbed = null;
    this.tokenizer = null;
  }
}
