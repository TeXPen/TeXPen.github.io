
import { InferenceSession, Tensor, env } from "onnxruntime-web";
import { AutoTokenizer, PreTrainedTokenizer } from "@huggingface/transformers";
import { preprocessPaddleVL } from "./utils/paddleVLPreprocess";
import { MODEL_CONFIG } from "./config";
import { downloadManager } from "../downloader/DownloadManager";
import { VLMInferenceResult, TokenCallback } from "./types";
import { paddleVLPostprocess } from "./utils/paddleVLPostprocess";

env.wasm.numThreads = Math.min(typeof navigator !== 'undefined' ? (navigator.hardwareConcurrency || 4) : 4, 8); // Improved fallback performance
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
  private llmWithPastSession: InferenceSession | null = null;

  private tokenizer: PreTrainedTokenizer | null = null;
  private posEmbed: Tensor | null = null;

  private initialized = false;
  private loading = false;
  private sizeCache: Partial<Record<keyof typeof MODEL_CONFIG.VLM_COMPONENTS, number>> = {};
  private initPromise: Promise<void> | null = null;
  private sessionProviders: Partial<Record<keyof typeof MODEL_CONFIG.VLM_COMPONENTS, string>> = {};

  private sessionOptions: InferenceSession.SessionOptions = {
    executionProviders: ['webgpu', 'wasm'], // Use WebGPU if available
    executionMode: 'sequential', // 'parallel' can cause issues with high-resource models
    graphOptimizationLevel: 'extended' // 'all' is sometimes unstable on WebGPU
  };

  private useServerPipeline(): boolean {
    return MODEL_CONFIG.PADDLE_VL_SERVER_URL.trim().length > 0;
  }

  private async loadModel(
    key: keyof typeof MODEL_CONFIG.VLM_COMPONENTS,
    sessionProp?: 'patchEmbedSession' | 'visionTransformerSession' | 'visionProjectorSession' | 'textEmbedSession' | 'llmSession' | 'llmWithPastSession',
    providers: string[] = ['wasm'],
    onProgress?: (msg: string) => void
  ) {
    if (sessionProp && this[sessionProp]) return; // Already loaded

    const baseFilename = MODEL_CONFIG.VLM_COMPONENTS[key];
    const quantizedFilename = baseFilename.replace('.onnx', MODEL_CONFIG.QUANTIZED_SUFFIX);
    const wantsQuantized = MODEL_CONFIG.QUANTIZED && (key === 'VISION_TRANSFORMER' || key === 'LLM' || key === 'LLM_WITH_PAST');

    const origin = typeof window !== 'undefined' ? window.location.origin : self.location.origin;

    const fetchModelBuffers = async (targetFilename: string) => {
      const path = `${origin}/models/vlm/${targetFilename}`;
      const modelResp = await fetch(path);
      if (!modelResp.ok) return null;

      const contentType = (modelResp.headers.get('content-type') || '').toLowerCase();
      if (contentType.startsWith('text/') || contentType.includes('text/html')) {
        return null;
      }

      const modelBuffer = await modelResp.arrayBuffer();
      const prefix = new TextDecoder().decode(new Uint8Array(modelBuffer.slice(0, 64)));
      if (prefix.trimStart().startsWith('<')) {
        return null;
      }

      let dataBuffer: ArrayBuffer | null = null;
      const dataResp = await fetch(`${path}.data`).then(r => r.ok ? r : null).catch(() => null);
      if (dataResp && dataResp.ok) {
        dataBuffer = await dataResp.arrayBuffer();
      }

      return { modelBuffer, dataBuffer, filename: targetFilename };
    };

    let modelBuffer: ArrayBuffer | null = null;
    let dataBuffer: ArrayBuffer | null = null;
    let filename = baseFilename;

    if (wantsQuantized) {
      try {
        const buffers = await fetchModelBuffers(quantizedFilename);
        if (buffers) {
          ({ modelBuffer, dataBuffer, filename } = buffers);
        } else {
          console.warn(`[VLM] Quantized model ${quantizedFilename} unavailable or invalid. Falling back to ${baseFilename}.`);
        }
      } catch (e) {
        console.warn(`[VLM] Failed to fetch ${quantizedFilename}. Falling back to ${baseFilename}.`, e);
      }
    }

    if (!modelBuffer) {
      const buffers = await fetchModelBuffers(baseFilename);
      if (!buffers) {
        throw new Error(`Failed to fetch ${baseFilename}`);
      }
      ({ modelBuffer, dataBuffer, filename } = buffers);
    }

    if (onProgress) onProgress(`Loading ${key}...`);

    try {
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

      const setProvider = (provider: string) => {
        this.sessionProviders[key] = provider;
      };

      try {
        console.log(`[VLM] Attempting to load ${key} with providers: ${providers.join(',')}`);
        const session = await createSession(providers);
        this[sessionProp] = session;
        setProvider(providers[0] ?? 'wasm');
        console.log(`[VLM] Loaded ${key} on ${providers[0]}`); // Assumption: first is primary
      } catch (gpuError) {
        // If we tried GPU and failed, fallback to CPU
        if (providers.includes('webgpu') || providers.includes('webgl')) {
          console.warn(`[VLM] Failed to load ${key} on GPU. Falling back to CPU (wasm). Error:`, gpuError);

          // Retry with fresh buffers (cloned inside createSession)
          const session = await createSession(['wasm']);

          this[sessionProp] = session;
          setProvider('wasm');
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
  // 2. LLM_GPU: LLM + Text Embed on GPU, Vision on CPU (Balanced for low VRAM)
  // 3. CPU_ONLY: Everything on CPU (Failsafe)
  private strategies: ('ALL_GPU' | 'LLM_GPU' | 'CPU_ONLY')[] = ['ALL_GPU', 'LLM_GPU', 'CPU_ONLY'];

  private currentStrategyIndex = 0; // Default to ALL_GPU

  public get currentStrategy(): 'ALL_GPU' | 'LLM_GPU' | 'CPU_ONLY' {
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
    if (MODEL_CONFIG.QUANTIZED && (key === 'VISION_TRANSFORMER' || key === 'LLM' || key === 'LLM_WITH_PAST')) {
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
      'llm_init.onnx': 2079423,
      'llm_with_past.onnx': 2079423,
      'pos_embed.npy': 3359360,

      // Quantized Hints (Approx 25% of FP32 for INT8, ~50% for INT4)
      'vision_transformer_q8.onnx': 1630000,
      'llm_init_q8.onnx': 1550000,
      'llm_with_past_q8.onnx': 1550000,
      'vision_transformer_q4.onnx': 1089562,
      'llm_q4.onnx': 1039711
    };
    const dataHints: Record<string, number> = {
      'vision_patch_embed.onnx.data': 2775040,
      'vision_transformer.onnx.data': 1647222784,
      'vision_projector.onnx.data': 103874560,
      'text_embed.onnx.data': 423624704,
      'llm.onnx.data': 1443037184,
      'llm_init.onnx.data': 1443037184,
      'llm_with_past.onnx.data': 1443037184,

      // Quantized Data Hints (INT8 ~= 1/4 size, INT4 ~= 1/8 size)
      'vision_transformer_q8.onnx.data': 411805696, // ~1/4
      'llm_init_q8.onnx.data': 360759296, // ~1/4
      'llm_with_past_q8.onnx.data': 360759296, // ~1/4
      'vision_transformer_q4.onnx.data': 205902848, // ~1/8
      'llm_q4.onnx.data': 180379648 // ~1/8
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

  private async selectStrategyFromBudget(onProgress?: (status: string, progress?: number) => void) {
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
      this.currentStrategyIndex = this.strategies.indexOf('CPU_ONLY');
      return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      this.currentStrategyIndex = this.strategies.indexOf('CPU_ONLY');
      return;
    }

    const budgetBytes = this.estimateGpuBudgetBytes(adapter);
    const overheadFactor = 1.25;
    const minAllGpuBytes = MODEL_CONFIG.WEBGPU_ALL_GPU_MIN_GB * 1024 * 1024 * 1024;
    const minLlmGpuBytes = MODEL_CONFIG.WEBGPU_LLM_GPU_MIN_GB * 1024 * 1024 * 1024;

    const sumSizes = async (keys: (keyof typeof MODEL_CONFIG.VLM_COMPONENTS)[]) => {
      let total = 0;
      for (const key of keys) {
        total += await this.getComponentSizeBytes(key);
      }
      return total;
    };

    const llmGpuBytes = await sumSizes(['TEXT_EMBED', 'LLM', 'LLM_WITH_PAST']);
    const allGpuBytes = llmGpuBytes + await sumSizes([
      'VISION_PATCH_EMBED',
      'VISION_TRANSFORMER',
      'VISION_PROJECTOR'
    ]);

    let chosen: 'ALL_GPU' | 'LLM_GPU' | 'CPU_ONLY';
    if (budgetBytes >= allGpuBytes * overheadFactor && budgetBytes >= minAllGpuBytes) {
      chosen = 'ALL_GPU';
    } else if (budgetBytes >= llmGpuBytes * overheadFactor && budgetBytes >= minLlmGpuBytes) {
      chosen = 'LLM_GPU';
    } else {
      chosen = 'CPU_ONLY';
    }

    const idx = this.strategies.indexOf(chosen);
    if (idx >= 0) {
      this.currentStrategyIndex = idx;
      if (onProgress) {
        onProgress(`Auto strategy: ${chosen}`, 5);
      }
    }

    console.log(
      `[VLM] Strategy auto-select: budget=${(budgetBytes / (1024 ** 3)).toFixed(2)}GB ` +
      `allGpu≈${(allGpuBytes * overheadFactor / (1024 ** 3)).toFixed(2)}GB ` +
      `llmGpu≈${(llmGpuBytes * overheadFactor / (1024 ** 3)).toFixed(2)}GB ` +
      `minAll=${MODEL_CONFIG.WEBGPU_ALL_GPU_MIN_GB}GB ` +
      `minLlm=${MODEL_CONFIG.WEBGPU_LLM_GPU_MIN_GB}GB -> ${chosen}`
    );
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
        if (this.useServerPipeline()) {
          this.initialized = true;
          if (onProgress) onProgress("Ready", 100);
          return;
        }

        // Ensure clean state before starting IF NOT already initialized
        // This dispose() is only for hard resets/recovery. 
        // Normal preloads or multiple calls while loading are handled above.
        await this.dispose();

        await this.selectStrategyFromBudget(onProgress);

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
        console.log(`[VLM] Loaded Pos Embed. Shape: ${this.posEmbed.dims}`);

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
        // Use WebGPU for vision by default; WASM aborts on some large inputs.
        visionProviders = gpu;
        break;
      case 'LLM_GPU':
        visionProviders = cpu;
        textEmbedProviders = gpu;
        llmProviders = gpu;
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
    try {
      await this.loadModel('LLM_WITH_PAST', 'llmWithPastSession', llmProviders, onProgress);
      await new Promise(r => setTimeout(r, 50));
    } catch (e) {
      console.warn("[VLM] Optional LLM_WITH_PAST not available. Falling back to non-cached generation.", e);
    }

    // 3. Vision Components
    // Forced to WASM (CPU) to avoid WebGPU timeouts/hangs (Code 6583176)
    // Vision Transformer and Projector are dense and often cause TDRs on some GPUs.
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
    console.log(`[VLM] runInference started. Prompt: "${prompt}"`);
    const lock = this.getInferenceLock();
    if (lock) {
      console.warn("[VLM] Inference Lock Active. Aborting new request.");
      throw new Error("Inference already in progress");
    }

    this.inferenceInProgress = true;
    try {
      if (this.useServerPipeline()) {
        return await this.runServerPipeline(imageBlob, prompt, signal);
      }
      console.time("[VLM] Total Inference Time");
      const result = await this.executePipeline(imageBlob, prompt, onToken, signal);
      console.timeEnd("[VLM] Total Inference Time");
      return result;
    } catch (error) {
      console.timeEnd("[VLM] Total Inference Time");
      if (error instanceof Error && error.message === "Aborted") {
        console.log("[VLM] Inference Aborted");
        throw error;
      }

      const err = error instanceof Error ? error : new Error(String(error));
      console.error(`[VLM] Inference Failed. Error Type: ${typeof error}, Code/Msg:`, error);

      // Check for fatal WASM/Environment errors
      if (err.message && (err.message.includes("Aborted()") || err.message.includes("valid external Instance"))) {
        console.error("[VLM] Fatal WASM Error detected");
        throw new Error("FATAL_RELOAD_NEEDED");
      }

      // Check for GPU OOM/Crash
      if (this.isGpuError(error)) {
        console.warn(`[VLM] GPU/Resource Error detected (${error}). Requesting Fatal Reload Logic.`);
        throw new Error("FATAL_RELOAD_NEEDED");
      }

      throw error;
    } finally {
      this.inferenceInProgress = false;
      console.log("[VLM] runInference finished.");
    }
  }

  private async runServerPipeline(imageBlob: Blob, _prompt: string, signal?: AbortSignal): Promise<VLMInferenceResult> {
    if (!this.useServerPipeline()) {
      throw new Error("PaddleOCR-VL server URL not configured");
    }

    const t0 = performance.now();
    const bytes = new Uint8Array(await imageBlob.arrayBuffer());
    let binary = "";
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
    }
    const base64 = btoa(binary);

    const response = await fetch(MODEL_CONFIG.PADDLE_VL_SERVER_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        file: base64,
        fileType: 1
      }),
      signal
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`PaddleOCR-VL server error ${response.status}: ${text}`);
    }

    const data = await response.json();
    const result = data?.result?.layoutParsingResults?.[0]?.markdown?.text;
    if (typeof result !== "string") {
      throw new Error("PaddleOCR-VL server response missing markdown text");
    }

    return {
      markdown: result,
      timings: { server_ms: performance.now() - t0 }
    };
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

    // Treat numeric error codes (like WASM pointers or hex status codes) as GPU/Resource errors
    if (typeof e === 'number') {
      console.warn(`[VLM] Numeric error caught: 0x${e.toString(16)} (${e})`);
      return true;
    }
    if (typeof e === 'string' && /^\d+$/.test(e.trim())) {
      const num = parseInt(e.trim(), 10);
      console.warn(`[VLM] Numeric error string caught: 0x${num.toString(16)} (${num})`);
      return true;
    }

    return false;
  }

  private isWasmAbortError(e: unknown): boolean {
    const msg = e instanceof Error ? e.message : String(e);
    return msg.toLowerCase().includes("aborted()");
  }

  private async executePipeline(imageBlob: Blob, prompt: string, onToken?: TokenCallback, signal?: AbortSignal): Promise<VLMInferenceResult> {
    if (!this.initialized) throw new Error("Engine not initialized");

    const timings: Record<string, number> = {};
    const t0 = performance.now();

    // 1. Preprocess Image
    console.log("[VLM] Stage 1: Preprocessing Image...");
    if (signal?.aborted) throw new Error("Aborted");
    const { pixelValues, grid } = await preprocessPaddleVL(imageBlob);
    console.log(`[VLM] Image Preprocessed. Shape: ${pixelValues.dims}`);
    timings['preprocess'] = performance.now() - t0;

    // 2. Vision Pipeline
    console.log("[VLM] Stage 2: Vision Pipeline...");
    const t1 = performance.now();

    // Run Vision Pipeline
    // Sessions are already loaded (either GPU or CPU fallback)
    if (signal?.aborted) throw new Error("Aborted");
    const { embeds: imageEmbeds, grid: imageGrid } = await this.runVisionPipeline(pixelValues, grid);
    console.log(`[VLM] Vision Pipeline complete. Embeds Shape: ${imageEmbeds.dims}`);

    timings['vision_encoder'] = performance.now() - t1;

    // 3. LLM Pipeline
    console.log("[VLM] Stage 3: LLM Pipeline...");
    const t2 = performance.now();

    // Run LLM Pipeline
    if (signal?.aborted) {
      (imageEmbeds as any).dispose?.();
      (pixelValues as any).dispose?.();
      throw new Error("Aborted");
    }

    let markdown = "";
    try {
      markdown = await this.runLLMPipeline(imageEmbeds, imageGrid, prompt, onToken, signal);
    } finally {
      // Always cleanup heavy vision tensors
      (imageEmbeds as any).dispose?.();
      (pixelValues as any).dispose?.();
    }

    console.log("[VLM] LLM Pipeline complete.");

    timings['generation'] = performance.now() - t2;

    return {
      markdown: paddleVLPostprocess(markdown),
      timings
    };
  }

  private interpolatePosEmbed(
    posData: Float32Array,
    baseSize: number,
    targetH: number,
    targetW: number,
    dim: number
  ): Float32Array {
    if (baseSize === targetH && baseSize === targetW) {
      return posData;
    }

    const output = new Float32Array(targetH * targetW * dim);
    const scaleY = baseSize / targetH;
    const scaleX = baseSize / targetW;

    for (let y = 0; y < targetH; y++) {
      const inY = (y + 0.5) * scaleY - 0.5;
      const y0 = Math.max(Math.floor(inY), 0);
      const y1 = Math.min(y0 + 1, baseSize - 1);
      const wy1 = inY - y0;
      const wy0 = 1 - wy1;

      for (let x = 0; x < targetW; x++) {
        const inX = (x + 0.5) * scaleX - 0.5;
        const x0 = Math.max(Math.floor(inX), 0);
        const x1 = Math.min(x0 + 1, baseSize - 1);
        const wx1 = inX - x0;
        const wx0 = 1 - wx1;

        const base00 = (y0 * baseSize + x0) * dim;
        const base01 = (y0 * baseSize + x1) * dim;
        const base10 = (y1 * baseSize + x0) * dim;
        const base11 = (y1 * baseSize + x1) * dim;
        const outBase = (y * targetW + x) * dim;

        for (let d = 0; d < dim; d++) {
          const v00 = posData[base00 + d];
          const v01 = posData[base01 + d];
          const v10 = posData[base10 + d];
          const v11 = posData[base11 + d];
          const v0 = v00 * wx0 + v01 * wx1;
          const v1 = v10 * wx0 + v11 * wx1;
          output[outBase + d] = v0 * wy0 + v1 * wy1;
        }
      }
    }

    return output;
  }

  private reorderForProjector(
    data: Float32Array,
    gridH: number,
    gridW: number,
    hiddenDim: number,
    mergeSize: number
  ): Float32Array {
    if (gridH % mergeSize !== 0 || gridW % mergeSize !== 0) {
      throw new Error(`Grid ${gridH}x${gridW} not divisible by merge size ${mergeSize}`);
    }

    const seqLen = gridH * gridW;
    const output = new Float32Array(data.length);
    let outToken = 0;

    for (let bh = 0; bh < gridH; bh += mergeSize) {
      for (let bw = 0; bw < gridW; bw += mergeSize) {
        for (let dh = 0; dh < mergeSize; dh++) {
          for (let dw = 0; dw < mergeSize; dw++) {
            const h = bh + dh;
            const w = bw + dw;
            const tokenIndex = h * gridW + w;
            if (tokenIndex >= seqLen) {
              continue;
            }
            const srcOffset = tokenIndex * hiddenDim;
            const dstOffset = outToken * hiddenDim;
            output.set(data.subarray(srcOffset, srcOffset + hiddenDim), dstOffset);
            outToken++;
          }
        }
      }
    }

    return output;
  }

  private buildPositionIdsForPrompt(
    inputIds: number[],
    imageGrid: { t: number; h: number; w: number },
    imageTokenId: number,
    visionStartTokenId: number,
    attentionMask?: number[]
  ): { positionIds: Tensor; positionIdsData: BigInt64Array; nextPosition: number } {
    const seqLen = inputIds.length;
    const mask = attentionMask ?? new Array(seqLen).fill(1);
    const filteredIds = inputIds.filter((_, idx) => mask[idx] === 1);

    const spatialMerge = MODEL_CONFIG.VLM_MERGE_SIZE;
    const llmGridT = imageGrid.t;
    const llmGridH = Math.floor(imageGrid.h / spatialMerge);
    const llmGridW = Math.floor(imageGrid.w / spatialMerge);
    const visionTokenCount = llmGridT * llmGridH * llmGridW;

    let imageNums = 0;
    for (let i = 0; i < filteredIds.length - 1; i++) {
      if (filteredIds[i] === visionStartTokenId && filteredIds[i + 1] === imageTokenId) {
        imageNums++;
      }
    }

    const pos0: number[] = [];
    const pos1: number[] = [];
    const pos2: number[] = [];
    let st = 0;
    let maxPos = -1;

    for (let i = 0; i < imageNums; i++) {
      const ed = filteredIds.indexOf(imageTokenId, st);
      if (ed === -1) {
        break;
      }
      const textLen = ed - st;
      const baseText = maxPos + 1;
      for (let j = 0; j < textLen; j++) {
        const v = baseText + j;
        pos0.push(v);
        pos1.push(v);
        pos2.push(v);
      }
      maxPos = baseText + textLen - 1;

      const baseVision = maxPos + 1;
      for (let t = 0; t < llmGridT; t++) {
        const tIndex = 0;
        for (let h = 0; h < llmGridH; h++) {
          for (let w = 0; w < llmGridW; w++) {
            pos0.push(baseVision + tIndex);
            pos1.push(baseVision + h);
            pos2.push(baseVision + w);
          }
        }
      }
      const visionMax = Math.max(llmGridT - 1, llmGridH - 1, llmGridW - 1);
      maxPos = Math.max(maxPos, baseVision + visionMax);

      st = ed + visionTokenCount;
    }

    if (st < filteredIds.length) {
      const textLen = filteredIds.length - st;
      const baseText = maxPos + 1;
      for (let j = 0; j < textLen; j++) {
        const v = baseText + j;
        pos0.push(v);
        pos1.push(v);
        pos2.push(v);
      }
      maxPos = baseText + textLen - 1;
    }

    if (pos0.length !== filteredIds.length) {
      throw new Error(`Position id length mismatch: ${pos0.length} vs ${filteredIds.length}`);
    }

    const posData = new BigInt64Array(3 * seqLen);
    let posCursor = 0;
    for (let i = 0; i < seqLen; i++) {
      if (mask[i] === 1) {
        posData[i] = BigInt(pos0[posCursor]);
        posData[seqLen + i] = BigInt(pos1[posCursor]);
        posData[2 * seqLen + i] = BigInt(pos2[posCursor]);
        posCursor++;
      } else {
        posData[i] = 0n;
        posData[seqLen + i] = 0n;
        posData[2 * seqLen + i] = 0n;
      }
    }

    return {
      positionIds: new Tensor('int64', posData, [3, 1, seqLen]),
      positionIdsData: posData,
      nextPosition: maxPos + 1
    };
  }

  private async runVisionPipeline(
    pixelValues: Tensor,
    grid: { t: number; h: number; w: number }
  ): Promise<{ embeds: Tensor; grid: { t: number; h: number; w: number } }> {
    // 1. Patch Embed
    if (!this.patchEmbedSession) throw new Error("patchEmbedSession is null");
    console.log("[VLM] Vision: Running Patch Embed...");
    const patchRes = await this.patchEmbedSession.run({ 'pixel_values': pixelValues });
    const patchFeatures = patchRes['patch_features'];
    if (!patchFeatures) throw new Error("patch_features output is missing");
    console.log(`[VLM] Vision: Patch Embed complete. Shape: ${patchFeatures.dims}`);

    // 2. Add Pos Embed
    if (!this.posEmbed) throw new Error("posEmbed is missing");
    console.log("[VLM] Vision: Transposing Patch Embed and Adding Position Embeddings...");

    const patchDataRaw = await patchFeatures.getData() as Float32Array;
    const [, C_p, H_p, W_p] = patchFeatures.dims;
    const seqLen = H_p * W_p;

    if (grid.h !== H_p || grid.w !== W_p) {
      console.warn(`[VLM] Grid ${grid.h}x${grid.w} does not match patch output ${H_p}x${W_p}. Using patch output dims.`);
      grid = { ...grid, h: H_p, w: W_p };
    }

    const transposedPatch = new Float32Array(patchDataRaw.length);
    // NCHW -> NHWC [B, H, W, C]
    for (let c = 0; c < C_p; c++) {
      for (let i = 0; i < seqLen; i++) {
        transposedPatch[i * C_p + c] = patchDataRaw[c * seqLen + i];
      }
    }

    const dim = C_p;
    if (this.posEmbed.dims[this.posEmbed.dims.length - 1] !== dim) {
      throw new Error(`Pos Embed dim ${this.posEmbed.dims} mismatch with Patch dim ${dim}`);
    }

    const posData = await this.posEmbed.getData() as Float32Array;
    const baseCount = this.posEmbed.dims[0];
    const baseSize = Math.round(Math.sqrt(baseCount));
    if (baseSize * baseSize !== baseCount) {
      throw new Error(`Pos Embed count ${baseCount} is not a square grid`);
    }

    const posInterpolated = this.interpolatePosEmbed(posData, baseSize, grid.h, grid.w, dim);
    const patchTotal = transposedPatch.length;
    const posTotal = posInterpolated.length;
    if (posTotal !== patchTotal) {
      throw new Error(`Pos Embed length ${posTotal} does not match patch length ${patchTotal}`);
    }

    const fullSequence = new Float32Array(patchTotal);
    for (let i = 0; i < patchTotal; i++) {
      fullSequence[i] = transposedPatch[i] + posInterpolated[i];
    }

    const enrichedTensor = new Tensor('float32', fullSequence, [1, seqLen, dim]);

    // Cleanup patch results
    for (const k in patchRes) { (patchRes[k] as any).dispose?.(); }

    // 3. Transformer & Projector
    if (!this.visionTransformerSession) throw new Error("visionTransformerSession is null");
    console.log("[VLM] Vision: Running Vision Transformer...");
    console.log(`[VLM] Vision Transformer Input Names: ${this.visionTransformerSession.inputNames}`);
    console.log(`[VLM] Vision Transformer Input Shape: ${enrichedTensor.dims}`);
    let transRes: Record<string, Tensor>;
    try {
      transRes = await this.visionTransformerSession.run(
        { 'inputs_embeds': enrichedTensor }
      );
    } catch (e) {
      if (this.isWasmAbortError(e) && this.sessionProviders.VISION_TRANSFORMER === 'wasm') {
        console.warn("[VLM] Vision Transformer crashed on WASM. Retrying on WebGPU...");
        await this.releaseSession('visionTransformerSession');
        await this.loadModel('VISION_TRANSFORMER', 'visionTransformerSession', ['webgpu', 'wasm']);
        if (!this.visionTransformerSession) {
          throw e;
        }
        transRes = await this.visionTransformerSession.run(
          { 'inputs_embeds': enrichedTensor }
        );
      } else {
        throw e;
      }
    }
    const lastHidden = transRes['last_hidden_state'];
    if (!lastHidden) throw new Error("last_hidden_state output is missing");
    console.log(`[VLM] Vision: Transformer Output Shape: ${lastHidden.dims}`);

    // Cleanup enrichedTensor
    (enrichedTensor as any).dispose?.();

    // Projector runs on the transformer output
    // Feedback: Projector expects Rank 3 (image_features: [1, L, D])
    if (!this.visionProjectorSession) throw new Error("visionProjectorSession is null");
    console.log("[VLM] Vision: Running Projector...");
    console.log(`[VLM] Vision Projector Input Names: ${this.visionProjectorSession.inputNames}`);
    // Check if Projector expects image_features
    if (!this.visionProjectorSession.inputNames.includes('image_features')) {
      console.warn(`[VLM] VISION PROJECTOR EXPECTS: ${this.visionProjectorSession.inputNames.join(', ')} BUT WE ARE PASSING 'image_features'`);
    }
    const hiddenDim = lastHidden.dims[2];
    const lastHiddenData = await lastHidden.getData() as Float32Array;
    const reordered = this.reorderForProjector(
      lastHiddenData,
      grid.h,
      grid.w,
      hiddenDim,
      MODEL_CONFIG.VLM_MERGE_SIZE
    );
    const projectorInput = new Tensor('float32', reordered, [1, seqLen, hiddenDim]);

    const projRes = await this.visionProjectorSession.run(
      { 'image_features': projectorInput }
    );
    const projectedFeatures = projRes['projected_features'];
    if (!projectedFeatures) throw new Error("projected_features output is missing");
    console.log(`[VLM] Vision: Projector complete. Shape: ${projectedFeatures.dims}`);

    const projectedData = await projectedFeatures.getData() as Float32Array;
    const mergedLen = (grid.h / MODEL_CONFIG.VLM_MERGE_SIZE) * (grid.w / MODEL_CONFIG.VLM_MERGE_SIZE) * grid.t;
    if (projectedData.length % mergedLen !== 0) {
      throw new Error(`Projector output size ${projectedData.length} not divisible by merged length ${mergedLen}`);
    }
    const mergedHidden = projectedData.length / mergedLen;
    const mergedEmbeds = new Tensor('float32', projectedData, [1, mergedLen, mergedHidden]);
    console.log(`[VLM] Vision: Projector reshaped. Shape: ${mergedEmbeds.dims}`);

    // Cleanup transformer result
    for (const k in transRes) { (transRes[k] as any).dispose?.(); }
    (projectorInput as any).dispose?.();
    (lastHidden as any).dispose?.();
    (projectedFeatures as any).dispose?.();

    for (const k in projRes) {
      if (k !== 'projected_features') {
        (projRes[k] as any).dispose?.();
      }
    }

    return { embeds: mergedEmbeds, grid };
  }

  private async runLLMPipeline(
    imageEmbeds: Tensor,
    imageGrid: { t: number; h: number; w: number },
    prompt: string,
    onToken?: TokenCallback,
    signal?: AbortSignal
  ): Promise<string> {
    console.log("[VLM] LLM: Tokenizing Prompt...");
    const mergeSize = MODEL_CONFIG.VLM_MERGE_SIZE;
    const imageTokenCount = (imageGrid.t * imageGrid.h * imageGrid.w) / (mergeSize * mergeSize);
    if (!Number.isInteger(imageTokenCount) || imageTokenCount <= 0) {
      throw new Error(`Invalid image token count: ${imageTokenCount}`);
    }

    const imageTokenText = "<|IMAGE_PLACEHOLDER|>";
    const imageTokens = imageTokenText.repeat(imageTokenCount);
    const fullPrompt = `<|begin_of_sentence|>User: <|IMAGE_START|>${imageTokens}<|IMAGE_END|>${prompt}\nAssistant: `;
    const { input_ids } = await this.tokenizer!(fullPrompt, { return_tensor: false, padding: false, truncation: true });
    const ids = input_ids as number[];
    console.log(`[VLM] LLM: Tokenization complete. IDs: ${ids.length}`);

    const imageTokenRes = await this.tokenizer!(imageTokenText, { return_tensor: false, add_special_tokens: false });
    const imageTokenId = (imageTokenRes.input_ids as number[])[0];
    const visionStartRes = await this.tokenizer!("<|IMAGE_START|>", { return_tensor: false, add_special_tokens: false });
    const visionStartTokenId = (visionStartRes.input_ids as number[])[0];

    const imagePositions: number[] = [];
    for (let i = 0; i < ids.length; i++) {
      if (ids[i] === imageTokenId) {
        imagePositions.push(i);
      }
    }
    if (imagePositions.length !== imageTokenCount) {
      throw new Error(`Image token count mismatch: prompt=${imagePositions.length} expected=${imageTokenCount}`);
    }

    console.log("[VLM] LLM: Embedding Text IDs...");
    const inputIdsTensor = new Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
    const textEmbedRes = await this.textEmbedSession!.run({ 'input_ids': inputIdsTensor });
    const textEmbeds = textEmbedRes['inputs_embeds'];
    console.log(`[VLM] LLM: Text embeddings complete. Shape: ${textEmbeds.dims}`);

    const hiddenDim = textEmbeds.dims[2];
    if (imageEmbeds.dims[2] !== hiddenDim) {
      throw new Error(`Image embed dim ${imageEmbeds.dims[2]} != text embed dim ${hiddenDim}`);
    }

    const imageData = await imageEmbeds.getData() as Float32Array;
    if (imageData.length !== imageTokenCount * hiddenDim) {
      throw new Error(`Image embed length ${imageData.length} does not match token count ${imageTokenCount}`);
    }

    const textData = await textEmbeds.getData() as Float32Array;
    const combinedData = new Float32Array(textData.length);
    combinedData.set(textData);

    for (let i = 0; i < imagePositions.length; i++) {
      const dstOffset = imagePositions[i] * hiddenDim;
      const srcOffset = i * hiddenDim;
      combinedData.set(imageData.subarray(srcOffset, srcOffset + hiddenDim), dstOffset);
    }

    const combinedEmbeds = new Tensor('float32', combinedData, [1, ids.length, hiddenDim]);
    console.log(`[VLM] LLM: Injected ${imageTokenCount} image tokens.`);

    // Cleanup initial text embedding resources
    (inputIdsTensor as any).dispose?.();
    for (const k in textEmbedRes) { (textEmbedRes[k] as any).dispose?.(); }

    const attentionMaskNumbers = new Array(ids.length).fill(1);
    const attentionMaskTensor = new Tensor('float32', new Float32Array(ids.length).fill(1), [1, ids.length]);
    const {
      positionIds,
      positionIdsData: basePositionIdsData,
      nextPosition: nextPositionInit
    } = this.buildPositionIdsForPrompt(
      ids,
      imageGrid,
      imageTokenId,
      visionStartTokenId,
      attentionMaskNumbers
    );
    let positionIdsData = basePositionIdsData;
    let positionIdsSeqLen = ids.length;
    let nextPosition = nextPositionInit;

    // 4. Generation Loop
    let currentEmbeds = combinedEmbeds;
    const generatedIds: number[] = [];
    const MAX_NEW_TOKENS = 512; // Increased for longer document responses

    console.log(`[VLM] Starting LLM Generation Loop. Input Tokens: ${combinedEmbeds.dims[1]}`);

    const hasCache = !!this.llmWithPastSession;
    let pastLen = currentEmbeds.dims[1];
    let pastState: Record<string, Tensor> = {};

    const buildPositionIds = (pos: number) => {
      const posData = new BigInt64Array(3);
      const v = BigInt(pos);
      posData[0] = v;
      posData[1] = v;
      posData[2] = v;
      return new Tensor('int64', posData, [3, 1, 1]);
    };

    const updatePastState = (outputs: Record<string, Tensor>) => {
      const nextState: Record<string, Tensor> = {};
      for (const name of Object.keys(outputs)) {
        if (name.startsWith("present.")) {
          const pastName = name.replace(/^present\./, "past_key_values.");
          nextState[pastName] = outputs[name];
        }
      }
      for (const k in pastState) {
        if (!nextState[k]) {
          (pastState[k] as any).dispose?.();
        }
      }
      pastState = nextState;
    };

    let currentTokenPosition = -1;

    try {
      // First step: full context
      {
        const res = await this.llmSession!.run({
          'inputs_embeds': currentEmbeds,
          'attention_mask': attentionMaskTensor,
          'position_ids': positionIds
        });

        const logits = res['logits'];
        const logitsData = await logits.getData() as Float32Array;
        const lastLogits = logitsData.slice(logitsData.length - logits.dims[2]);
        const nextId = argmax(lastLogits);

        if (hasCache) {
          updatePastState(res);
        }

        for (const k in res) {
          if (!pastState[k]) (res[k] as any).dispose?.();
        }
        (attentionMaskTensor as any).dispose?.();
        (positionIds as any).dispose?.();

        if (nextId === this.tokenizer!.eos_token_id) {
          return "";
        }

        generatedIds.push(nextId);
        if (onToken) {
          const decoded = this.tokenizer!.decode([nextId], { skip_special_tokens: true });
          const fullSoFar = this.tokenizer!.decode(generatedIds, { skip_special_tokens: true });
          onToken(decoded, fullSoFar);
        }

        // Prepare first cached step
        currentTokenPosition = nextPosition;
        nextPosition += 1;
        const nextIdTensor = new Tensor('int64', BigInt64Array.from([BigInt(nextId)]), [1, 1]);
        const nextEmbedRes = await this.textEmbedSession!.run({ 'input_ids': nextIdTensor });
        const nextEmbed = nextEmbedRes['inputs_embeds'];

        if (hasCache) {
          currentEmbeds = nextEmbed;
        } else {
          const oldData = await currentEmbeds.getData() as Float32Array;
          const newData = await nextEmbed.getData() as Float32Array;
          const fullData = new Float32Array(oldData.length + newData.length);
          fullData.set(oldData);
          fullData.set(newData, oldData.length);
          const prevEmbeds = currentEmbeds;
          currentEmbeds = new Tensor('float32', fullData, [1, currentEmbeds.dims[1] + 1, hiddenDim]);
          if (prevEmbeds !== combinedEmbeds) {
            (prevEmbeds as any).dispose?.();
          }
          const oldSeqLen = positionIdsSeqLen;
          const newSeqLen = oldSeqLen + 1;
          const newPosData = new BigInt64Array(3 * newSeqLen);
          newPosData.set(positionIdsData.subarray(0, oldSeqLen), 0);
          newPosData.set(positionIdsData.subarray(oldSeqLen, 2 * oldSeqLen), newSeqLen);
          newPosData.set(positionIdsData.subarray(2 * oldSeqLen, 3 * oldSeqLen), 2 * newSeqLen);
          newPosData[newSeqLen - 1] = BigInt(currentTokenPosition);
          newPosData[newSeqLen + newSeqLen - 1] = BigInt(currentTokenPosition);
          newPosData[2 * newSeqLen + newSeqLen - 1] = BigInt(currentTokenPosition);
          positionIdsData = newPosData;
          positionIdsSeqLen = newSeqLen;
        }

        (nextIdTensor as any).dispose?.();
        for (const k in nextEmbedRes) { (nextEmbedRes[k] as any).dispose?.(); }
      }

      for (let i = 1; i < MAX_NEW_TOKENS; i++) {
        if (signal?.aborted) {
          console.log(`[VLM] Gen Loop: Aborted at step ${i}`);
          throw new Error("Aborted");
        }

        if (hasCache) {
          const attentionMask = new Tensor('float32', new Float32Array(pastLen + 1).fill(1), [1, pastLen + 1]);
          const positionIds = buildPositionIds(currentTokenPosition);
          const feeds: Record<string, Tensor> = {
            'inputs_embeds': currentEmbeds,
            'attention_mask': attentionMask,
            'position_ids': positionIds
          };
          for (const name of this.llmWithPastSession!.inputNames) {
            if (name.startsWith("past_key_values.") && pastState[name]) {
              feeds[name] = pastState[name];
            }
          }

          const res = await this.llmWithPastSession!.run(feeds);
          const logits = res['logits'];
          const logitsData = await logits.getData() as Float32Array;
          const lastLogits = logitsData.slice(logitsData.length - logits.dims[2]);
          const nextId = argmax(lastLogits);

          updatePastState(res);
          for (const k in res) {
            if (!pastState[k]) (res[k] as any).dispose?.();
          }
          (attentionMask as any).dispose?.();
          (positionIds as any).dispose?.();

          if (nextId === this.tokenizer!.eos_token_id) {
            console.log(`[VLM] Gen Loop: EOS reached at step ${i}`);
            break;
          }

          generatedIds.push(nextId);
          if (onToken) {
            const decoded = this.tokenizer!.decode([nextId], { skip_special_tokens: true });
            const fullSoFar = this.tokenizer!.decode(generatedIds, { skip_special_tokens: true });
            onToken(decoded, fullSoFar);
          }

          currentTokenPosition = nextPosition;
          nextPosition += 1;

          const nextIdTensor = new Tensor('int64', BigInt64Array.from([BigInt(nextId)]), [1, 1]);
          const nextEmbedRes = await this.textEmbedSession!.run({ 'input_ids': nextIdTensor });
          const nextEmbed = nextEmbedRes['inputs_embeds'];
          if (currentEmbeds !== combinedEmbeds) {
            (currentEmbeds as any).dispose?.();
          }
          currentEmbeds = nextEmbed;
          pastLen += 1;

          (nextIdTensor as any).dispose?.();
          for (const k in nextEmbedRes) { (nextEmbedRes[k] as any).dispose?.(); }
        } else {
          const seqLen = positionIdsSeqLen;
          const attentionMask = new Tensor('float32', new Float32Array(seqLen).fill(1), [1, seqLen]);
          const positionIds = new Tensor('int64', positionIdsData, [3, 1, seqLen]);

          const res = await this.llmSession!.run({
            'inputs_embeds': currentEmbeds,
            'attention_mask': attentionMask,
            'position_ids': positionIds
          });

          const logits = res['logits'];
          const logitsData = await logits.getData() as Float32Array;
          const lastLogits = logitsData.slice(logitsData.length - logits.dims[2]);
          const nextId = argmax(lastLogits);

          if (nextId === this.tokenizer!.eos_token_id) {
            console.log(`[VLM] Gen Loop: EOS reached at step ${i}`);
            for (const k in res) { (res[k] as any).dispose?.(); }
            (attentionMask as any).dispose?.();
            (positionIds as any).dispose?.();
            break;
          }
          generatedIds.push(nextId);

          if (onToken) {
            const decoded = this.tokenizer!.decode([nextId], { skip_special_tokens: true });
            const fullSoFar = this.tokenizer!.decode(generatedIds, { skip_special_tokens: true });
            onToken(decoded, fullSoFar);
          }

          currentTokenPosition = nextPosition;
          nextPosition += 1;

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

          if (prevEmbeds !== combinedEmbeds) {
            (prevEmbeds as any).dispose?.();
          }
          const oldSeqLen = positionIdsSeqLen;
          const newSeqLen = oldSeqLen + 1;
          const newPosData = new BigInt64Array(3 * newSeqLen);
          newPosData.set(positionIdsData.subarray(0, oldSeqLen), 0);
          newPosData.set(positionIdsData.subarray(oldSeqLen, 2 * oldSeqLen), newSeqLen);
          newPosData.set(positionIdsData.subarray(2 * oldSeqLen, 3 * oldSeqLen), 2 * newSeqLen);
          newPosData[newSeqLen - 1] = BigInt(currentTokenPosition);
          newPosData[newSeqLen + newSeqLen - 1] = BigInt(currentTokenPosition);
          newPosData[2 * newSeqLen + newSeqLen - 1] = BigInt(currentTokenPosition);
          positionIdsData = newPosData;
          positionIdsSeqLen = newSeqLen;

          (nextIdTensor as any).dispose?.();
          for (const k in nextEmbedRes) { (nextEmbedRes[k] as any).dispose?.(); }
          for (const k in res) { (res[k] as any).dispose?.(); }
          (attentionMask as any).dispose?.();
          (positionIds as any).dispose?.();
        }
      }
    } finally {
      // Final cleanup
      if (currentEmbeds) (currentEmbeds as any).dispose?.();
      for (const k in pastState) {
        (pastState[k] as any).dispose?.();
      }
      // Note: combinedEmbeds is currentEmbeds initially, handled by the loop/finally
    }

    return this.tokenizer!.decode(generatedIds, { skip_special_tokens: true });
  }

  public async dispose() {
    if (this.patchEmbedSession) await this.patchEmbedSession.release();
    if (this.visionTransformerSession) await this.visionTransformerSession.release();
    if (this.visionProjectorSession) await this.visionProjectorSession.release();
    if (this.textEmbedSession) await this.textEmbedSession.release();
    if (this.llmSession) await this.llmSession.release();
    if (this.llmWithPastSession) await this.llmWithPastSession.release();

    this.patchEmbedSession = null;
    this.visionTransformerSession = null;
    this.visionProjectorSession = null;
    this.textEmbedSession = null;
    this.llmSession = null;
    this.llmWithPastSession = null;

    this.initialized = false;
    this.posEmbed = null;
    this.tokenizer = null;
  }
}
