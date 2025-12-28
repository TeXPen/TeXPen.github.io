
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
  private loading = false;
  private sizeCache: Partial<Record<keyof typeof MODEL_CONFIG.VLM_COMPONENTS, number>> = {};

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

    const filename = MODEL_CONFIG.VLM_COMPONENTS[key];
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
  private strategies: ('ALL_GPU' | 'LLM_GPU' | 'CPU_ONLY')[] = ['ALL_GPU', 'LLM_GPU', 'CPU_ONLY'];
  private currentStrategyIndex = 0; // Default to ALL_GPU

  private get currentStrategy(): 'ALL_GPU' | 'LLM_GPU' | 'CPU_ONLY' {
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

    const filename = MODEL_CONFIG.VLM_COMPONENTS[key];
    const origin = typeof window !== 'undefined' ? window.location.origin : self.location.origin;
    const path = `${origin}/models/vlm/${filename}`;

    const sizeHints: Record<string, number> = {
      'vision_patch_embed.onnx': 564,
      'vision_transformer.onnx': 2179125,
      'vision_projector.onnx': 24047,
      'text_embed.onnx': 313,
      'llm.onnx': 2079423,
      'pos_embed.npy': 3359360
    };
    const dataHints: Record<string, number> = {
      'vision_patch_embed.onnx.data': 2775040,
      'vision_transformer.onnx.data': 1647222784,
      'vision_projector.onnx.data': 103874560,
      'text_embed.onnx.data': 423624704,
      'llm.onnx.data': 1443037184
    };

    let size = await this.fetchContentLength(path);
    if (size === null) {
      size = sizeHints[filename] ?? 64 * 1024 * 1024;
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

  private async buildStrategyPlan(): Promise<Array<{
    name: 'ALL_GPU' | 'LLM_GPU' | 'VISION_GPU' | 'CPU_ONLY';
    providers: {
      VISION_PATCH_EMBED: string[];
      VISION_TRANSFORMER: string[];
      VISION_PROJECTOR: string[];
      TEXT_EMBED: string[];
      LLM: string[];
    };
  }>> {
    if (typeof navigator === 'undefined') return [{
      name: 'CPU_ONLY',
      providers: {
        VISION_PATCH_EMBED: ['wasm'],
        VISION_TRANSFORMER: ['wasm'],
        VISION_PROJECTOR: ['wasm'],
        TEXT_EMBED: ['wasm'],
        LLM: ['wasm']
      }
    }];

    if (!('gpu' in navigator)) {
      console.warn("[VLM] WebGPU not supported. Using CPU_ONLY.");
      return [{
        name: 'CPU_ONLY',
        providers: {
          VISION_PATCH_EMBED: ['wasm'],
          VISION_TRANSFORMER: ['wasm'],
          VISION_PROJECTOR: ['wasm'],
          TEXT_EMBED: ['wasm'],
          LLM: ['wasm']
        }
      }];
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      console.warn("[VLM] No WebGPU adapter found. Using CPU_ONLY.");
      return [{
        name: 'CPU_ONLY',
        providers: {
          VISION_PATCH_EMBED: ['wasm'],
          VISION_TRANSFORMER: ['wasm'],
          VISION_PROJECTOR: ['wasm'],
          TEXT_EMBED: ['wasm'],
          LLM: ['wasm']
        }
      }];
    }

    const gpuBudget = this.estimateGpuBudgetBytes(adapter);
    const deviceMemory = (navigator as any).deviceMemory || 4;
    console.log(`[VLM] Detected Resources: DeviceMemory ~${deviceMemory}GB, GPU budget ~${(gpuBudget / 1024 / 1024).toFixed(0)}MB`);

    // Safety Margin Check
    // Total approx static size: ~3.5GB.
    // Inference overhead (KV cache, buffers): ~1.0GB+
    // Total required for ALL_GPU: ~4.5GB - 5.0GB

    // If deviceMemory < 8, likely a shared memory system (Apple Silicon / iGPU) with tight constraints.
    // Or if gpuBudget is oddly small.
    let allowAllGpu = true;

    // Conservative check: if system has 4GB or less RAM, definitely don't try ALL_GPU.
    if (deviceMemory <= 4) {
      console.warn("[VLM] Device memory too low (<=4GB) for ALL_GPU. Disabling.");
      allowAllGpu = false;
    }

    // "Margin of Error" requested by user:
    // If we are on the edge, safer to fallback to Balanced/LLM_GPU.
    // If we can't reliably determine >6GB memory, we should be cautious.

    // We'll trust the process: if it crashes, the reload recovery handles it. 
    // BUT we can help by checking if we have already crashed before? 
    // (Engine doesn't know about session history, but Demo does).

    const strategies: Array<{
      name: 'ALL_GPU' | 'LLM_GPU' | 'VISION_GPU' | 'CPU_ONLY';
      providers: {
        VISION_PATCH_EMBED: string[];
        VISION_TRANSFORMER: string[];
        VISION_PROJECTOR: string[];
        TEXT_EMBED: string[];
        LLM: string[];
      };
    }> = [];

    // 1. ALL_GPU (Best Performance, but high memory)
    if (allowAllGpu) {
      strategies.push({
        name: 'ALL_GPU',
        providers: {
          VISION_PATCH_EMBED: ['webgpu', 'wasm'],
          VISION_TRANSFORMER: ['webgpu', 'wasm'],
          VISION_PROJECTOR: ['webgpu', 'wasm'],
          TEXT_EMBED: ['webgpu', 'wasm'],
          LLM: ['webgpu', 'wasm']
        }
      });
    } else {
      // If we skipped ALL_GPU, we might still include it at the end? Or just skip?
      // Better to skip to enforce margin.
    }

    // 2. LLM_GPU (Balanced: LLM is heaviest compute, Vision on CPU saves VRAM)
    // Vision ~1.7GB (Transformer) + 100MB (Projector) -> Moves ~1.8GB to System RAM.
    // LLM ~1.4GB + Overhead -> stays on GPU. 
    // This is much safer for 4GB VRAM cards.
    strategies.push({
      name: 'LLM_GPU',
      providers: {
        VISION_PATCH_EMBED: ['wasm'],
        VISION_TRANSFORMER: ['wasm'],
        VISION_PROJECTOR: ['wasm'],
        TEXT_EMBED: ['webgpu', 'wasm'],
        LLM: ['webgpu', 'wasm']
      }
    });

    // 3. CPU_ONLY (Universal Fallback)
    strategies.push({
      name: 'CPU_ONLY',
      providers: {
        VISION_PATCH_EMBED: ['wasm'],
        VISION_TRANSFORMER: ['wasm'],
        VISION_PROJECTOR: ['wasm'],
        TEXT_EMBED: ['wasm'],
        LLM: ['wasm']
      }
    });

    return strategies;
  }

  public async init(onProgress?: (status: string, progress?: number) => void) {
    if (this.initialized) return;
    if (this.loading) {
      console.warn("[VLM] Initialization already in progress...");
      return;
    }

    this.loading = true;
    try {
      // Ensure clean state before starting
      await this.dispose();

      // Simplify Init Logic: Just use current strategy
      await this.initStrategy(null, onProgress);


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
    }
  }

  private async initStrategy(
    // We ignore the passed strategy argument now as we use internal state
    _ignoredResultStrat?: any,
    onProgress?: (status: string, progress?: number) => void
  ) {
    const strat = this.currentStrategy;
    console.log(`[VLM] Initializing with Active Strategy: ${strat}`);

    // Map strategy to providers
    const isCpu = strat === 'CPU_ONLY';
    const isLlmGpu = strat === 'LLM_GPU';

    // Providers lists
    const gpu = ['webgpu', 'wasm'];
    const cpu = ['wasm'];

    // LLM_GPU: LLM on GPU, Vision on CPU (Prioritizing LLM VRAM usage)
    // All GPU: All GPU
    // CPU: All CPU

    const visionProviders = (isCpu || isLlmGpu) ? cpu : gpu;
    const llmProviders = isCpu ? cpu : gpu;

    // Load models sequentially - PRIORITIZING LLM FIRST
    // This ensures if VRAM is tight, LLM gets it first.

    await this.loadModel('TEXT_EMBED', 'textEmbedSession', llmProviders, onProgress);
    await new Promise(r => setTimeout(r, 50));

    await this.loadModel('LLM', 'llmSession', llmProviders, onProgress);
    await new Promise(r => setTimeout(r, 50));

    await this.loadModel('VISION_PATCH_EMBED', 'patchEmbedSession', visionProviders, onProgress);
    await new Promise(r => setTimeout(r, 50));

    await this.loadModel('VISION_TRANSFORMER', 'visionTransformerSession', visionProviders, onProgress);
    await new Promise(r => setTimeout(r, 50));

    await this.loadModel('VISION_PROJECTOR', 'visionProjectorSession', visionProviders, onProgress);
    await new Promise(r => setTimeout(r, 50));
  }

  public setStrategyIndex(index: number) {
    if (index >= 0 && index < this.strategies.length) {
      this.currentStrategyIndex = index;
    }
  }

  public getStrategyIndex(): number {
    return this.currentStrategyIndex;
  }

  public async inferVLM(imageBlob: Blob, prompt: string = "Describe this image."): Promise<VLMInferenceResult> {
    const MAX_RETRIES = 3;

    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      try {
        return await this.executePipeline(imageBlob, prompt);
      } catch (error) {
        console.error(`[VLM] Inference Attempt ${attempt + 1} Failed:`, error);

        const err = error instanceof Error ? error : new Error(String(error));
        // Check for fatal WASM/Environment errors that require a full page reload
        if (err.message && (err.message.includes("Aborted()") || err.message.includes("valid external Instance"))) {
          throw new Error("FATAL_RELOAD_NEEDED");
        }

        // Check if it's a recoverable GPU error (including raw numeric codes)
        if (this.isGpuError(error) && this.canDowngrade()) {
          console.warn(`[VLM] GPU OOM/Crash/WASM Error detected (${error}). Downgrading strategy and restarting...`);

          // 1. Downgrade
          this.downgradeStrategy();

          // 2. Dispose everything (Context likely dead)
          await this.dispose();

          // 3. Re-Init
          // Note: We don't have the original progress callback here, logs will suffice.
          console.log("[VLM] Re-initializing models...");
          try {
            await this.init();
            // 4. Retry loop will run execution again
            continue;
          } catch (reinitError) {
            console.error("[VLM] Re-initialization failed:", reinitError);
            const reinitErr = reinitError as Error;
            if (reinitErr.message && (reinitErr.message.includes("Aborted()") || reinitErr.message.includes("valid external Instance"))) {
              throw new Error("FATAL_RELOAD_NEEDED");
            }
            // If re-init failed, we might want to try downgrading AGAIN if possible?
            if (!this.initialized && this.canDowngrade()) {
              this.downgradeStrategy();
              await this.dispose();
              try {
                await this.init();
                continue;
              } catch (e) { throw new Error("FATAL_RELOAD_NEEDED"); }
            }
            throw reinitError;
          }
        }

        throw error; // Not recoverable
      }
    }
    throw new Error("Max retries exceeded");
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

  private canDowngrade(): boolean {
    return this.currentStrategyIndex < this.strategies.length - 1;
  }

  private downgradeStrategy() {
    if (this.canDowngrade()) {
      this.currentStrategyIndex++;
      console.log(`[VLM] Downgraded to Strategy: ${this.currentStrategy}`);
    }
  }

  private async executePipeline(imageBlob: Blob, prompt: string): Promise<VLMInferenceResult> {
    if (!this.initialized) throw new Error("Engine not initialized");

    const timings: Record<string, number> = {};
    const t0 = performance.now();

    // 1. Preprocess Image
    const pixelValues = await preprocessPaddleVL(imageBlob);
    timings['preprocess'] = performance.now() - t0;

    // 2. Vision Pipeline
    const t1 = performance.now();

    // Run Vision Pipeline
    // Sessions are already loaded (either GPU or CPU fallback)
    const imageEmbeds = await this.runVisionPipeline(pixelValues);

    timings['vision_encoder'] = performance.now() - t1;

    // 3. LLM Pipeline
    const t2 = performance.now();

    // Run LLM Pipeline
    const markdown = await this.runLLMPipeline(imageEmbeds, prompt);

    timings['generation'] = performance.now() - t2;

    return {
      markdown: paddleVLPostprocess(markdown),
      timings
    };
  }

  private async runVisionPipeline(pixelValues: Tensor): Promise<Tensor> {
    // Patch Embed
    const patchRes = await this.patchEmbedSession!.run({ 'pixel_values': pixelValues });
    const patchFeatures = patchRes['patch_features'];

    // Add Pos Embed
    const patchData = patchFeatures.data as Float32Array;
    const posData = this.posEmbed!.data as Float32Array;
    const enrichedFeatures = new Float32Array(patchData.length);
    for (let i = 0; i < patchData.length; i++) {
      enrichedFeatures[i] = patchData[i] + posData[i % posData.length];
    }
    const enrichedTensor = new Tensor('float32', enrichedFeatures, [1, 729, 1152]);

    // Transformer
    const transRes = await this.visionTransformerSession!.run({ 'inputs_embeds': enrichedTensor });
    const lastHidden = transRes['last_hidden_state'];

    // Projector
    const projRes = await this.visionProjectorSession!.run({ 'image_features': lastHidden });
    return projRes['projected_features'];
  }

  private async runLLMPipeline(imageEmbeds: Tensor, prompt: string): Promise<string> {
    // 3. Text Embedding & Concat
    const { input_ids } = await this.tokenizer!(prompt, { return_tensor: false, padding: true, truncation: true });
    const inputIdsTensor = new Tensor('int64', BigInt64Array.from((input_ids as number[]).map(BigInt)), [1, (input_ids as number[]).length]);

    const textEmbedRes = await this.textEmbedSession!.run({ 'input_ids': inputIdsTensor });
    const textEmbeds = textEmbedRes['inputs_embeds'];

    // Concat
    const imgData = imageEmbeds.data as Float32Array;
    const txtData = textEmbeds.data as Float32Array;
    const hiddenDim = imageEmbeds.dims[2];
    const combinedLen = (imageEmbeds.dims[1] + textEmbeds.dims[1]) * hiddenDim;
    const combinedData = new Float32Array(combinedLen);
    combinedData.set(imgData);
    combinedData.set(txtData, imgData.length);

    const combinedEmbeds = new Tensor('float32', combinedData, [1, imageEmbeds.dims[1] + textEmbeds.dims[1], hiddenDim]);

    // 4. Generation Loop
    const seqLen = combinedEmbeds.dims[1];
    const attentionMask = new Tensor('int64', new BigInt64Array(seqLen).fill(1n), [1, seqLen]);
    const positionIds = new Tensor('int64', BigInt64Array.from({ length: seqLen }, (_, i) => BigInt(i)), [1, seqLen]);

    let currentEmbeds = combinedEmbeds;
    let feeds: Record<string, Tensor> = {
      'inputs_embeds': currentEmbeds,
      'attention_mask': attentionMask,
      'position_ids': positionIds
    };

    const generatedIds: number[] = [];
    const MAX_NEW_TOKENS = 128;
    // let pastIds = input_ids as number[]; // unused now

    // Initial run
    const res = await this.llmSession!.run(feeds);
    let logits = res['logits'];
    let lastLogits = (logits.data as Float32Array).slice(logits.data.length - logits.dims[2]);
    let nextId = argmax(lastLogits);
    generatedIds.push(nextId);
    // pastIds.push(nextId);

    // Loop
    for (let i = 0; i < MAX_NEW_TOKENS; i++) {
      // Optimization: If user cancels or something, we should break. 
      // But for now just standard loop.
      const nextIdTensor = new Tensor('int64', BigInt64Array.from([BigInt(nextId)]), [1, 1]);
      const nextEmbedRes = await this.textEmbedSession!.run({ 'input_ids': nextIdTensor });
      const nextEmbed = nextEmbedRes['inputs_embeds'];

      const oldData = currentEmbeds.data as Float32Array;
      const newData = nextEmbed.data as Float32Array;
      const totalLen = oldData.length + newData.length;
      const fullData = new Float32Array(totalLen);
      fullData.set(oldData);
      fullData.set(newData, oldData.length);

      currentEmbeds = new Tensor('float32', fullData, [1, currentEmbeds.dims[1] + 1, hiddenDim]);

      const newSeqLen = currentEmbeds.dims[1];
      const newMask = new Tensor('int64', new BigInt64Array(newSeqLen).fill(1n), [1, newSeqLen]);
      const newPos = new Tensor('int64', BigInt64Array.from({ length: newSeqLen }, (_, i) => BigInt(i)), [1, newSeqLen]);

      feeds = {
        'inputs_embeds': currentEmbeds,
        'attention_mask': newMask,
        'position_ids': newPos
      };

      const stepRes = await this.llmSession!.run(feeds);
      logits = stepRes['logits'];
      lastLogits = (logits.data as Float32Array).slice(logits.data.length - logits.dims[2]);
      nextId = argmax(lastLogits);

      if (nextId === this.tokenizer!.eos_token_id) break;
      generatedIds.push(nextId);
    }

    return this.tokenizer!.decode(generatedIds, { skip_special_tokens: true });
  }
  public async dispose() {
    await this.releaseSession('patchEmbedSession');
    await this.releaseSession('visionTransformerSession');
    await this.releaseSession('visionProjectorSession');
    await this.releaseSession('textEmbedSession');
    await this.releaseSession('llmSession');
    this.initialized = false;
    this.posEmbed = null;
    this.tokenizer = null;
  }
}
