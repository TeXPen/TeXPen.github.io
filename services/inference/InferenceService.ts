import {
  AutoTokenizer,
  PreTrainedTokenizer,
  Tensor,
} from "@huggingface/transformers";
import { removeStyle, addNewlines } from "../../utils/latex";
import { preprocess } from "./imagePreprocessing";
import { beamSearch } from "./beamSearch";
import { isWebGPUAvailable } from "../../utils/env";
import { InferenceQueue, InferenceRequest } from "./utils/InferenceQueue";
import {
  MODEL_CONFIG,
  getSessionOptions,
  getGenerationConfig,
} from "./config";
import {
  InferenceOptions,
  InferenceResult,
  VisionEncoderDecoderModel,
  SamplingOptions,
} from "./types";

export class InferenceService {
  private model: VisionEncoderDecoderModel | null = null;
  private tokenizer: PreTrainedTokenizer | null = null;
  private static instance: InferenceService;
  private dtype: string = MODEL_CONFIG.DEFAULT_QUANTIZATION;
  private currentModelId: string = MODEL_CONFIG.ID;
  private initPromise: Promise<void> | null = null;

  private queue: InferenceQueue;

  private constructor() {
    this.queue = new InferenceQueue((req, signal) => this.runInference(req, signal));
  }

  public static getInstance(): InferenceService {
    if (!InferenceService.instance) {
      InferenceService.instance = new InferenceService();
    }
    return InferenceService.instance;
  }

  private disposalGeneration: number = 0;
  private loadingMutex: Promise<void> = Promise.resolve();

  public async init(
    onProgress?: (status: string, progress?: number) => void,
    options: InferenceOptions = {}
  ): Promise<void> {
    // If initialization is already in progress, return the existing promise
    if (this.initPromise) {
      return this.initPromise;
    }

    // Append our actual work to the mutex chain
    const work = async () => {
      await this.runLoadingSequence(
        this.disposalGeneration,
        onProgress,
        options
      );
    };

    // Update the mutex - chain work onto it
    this.loadingMutex = this.loadingMutex.then(work).catch((err) => {
      console.error("Error in initialization sequence:", err);
    });

    // initPromise should track this specific work
    this.initPromise = this.loadingMutex;

    try {
      await this.initPromise;
    } finally {
      this.initPromise = null;
    }
  }

  private async runLoadingSequence(
    localGeneration: number,
    onProgress:
      | ((status: string, progress?: number) => void)
      | undefined,
    options: InferenceOptions
  ): Promise<void> {
    if (this.model && this.tokenizer) {
      // If the model is already loaded, but the quantization or device is different, we need to dispose and reload.
      if (
        (options.dtype &&
          this.model.config.dtype !== options.dtype) ||
        (options.device &&
          this.model.config.device !== options.device) ||
        (options.modelId && this.currentModelId !== options.modelId)
      ) {
        if (this.queue.getIsInferring()) {
          console.warn(
            "Changing model settings while inference is in progress."
          );
          throw new Error(
            "Cannot change model settings while an inference is in progress."
          );
        }
        await this.dispose();
        if (this.disposalGeneration !== localGeneration) {
          return;
        }
      } else {
        // Already loaded with correct settings (including model ID)
        return;
      }
    }

    try {
      // CHECK GENERATION before starting heavy work
      if (this.disposalGeneration !== localGeneration) return;

      await this.handleSessionWait(onProgress);

      const webgpuAvailable = await isWebGPUAvailable();
      const device =
        options.device || (webgpuAvailable ? MODEL_CONFIG.PROVIDERS.WEBGPU : MODEL_CONFIG.PROVIDERS.WASM);
      const dtype =
        options.dtype ||
        (webgpuAvailable
          ? MODEL_CONFIG.DEFAULT_QUANTIZATION
          : MODEL_CONFIG.QUANTIZATION.Q8);

      // Update current ID if provided, otherwise keep existing (or default on first run)
      if (options.modelId) {
        this.currentModelId = options.modelId;
      }

      if (onProgress)
        onProgress(
          `Loading model ${this.currentModelId} (${device}, ${dtype})...`
        );

      const sessionOptions = getSessionOptions(device, dtype);

      // Pre-download heavy model files using ModelLoader
      const { modelLoader } = await import("./ModelLoader");
      await modelLoader.preDownloadModels(
        this.currentModelId,
        sessionOptions,
        onProgress
      );

      if (onProgress) onProgress('Initializing model (compiling shaders)...');

      // Load Tokenizer and Model in parallel
      const [tokenizer, modelResult] = await Promise.all([
        AutoTokenizer.from_pretrained(this.currentModelId),
        modelLoader.loadModelWithFallback(
          this.currentModelId,
          device,
          onProgress
        ),
      ]);

      const { model, dtype: finalDtype } = modelResult;

      // CHECK GENERATION AGAIN
      if (this.disposalGeneration !== localGeneration) {
        if (model && "dispose" in model) {
          await model.dispose();
        }
        return;
      }

      this.tokenizer = tokenizer;
      this.model = model;
      this.dtype = finalDtype;

      // Clear loading flag - we're done
      try {
        sessionStorage.removeItem("__texpen_loading__");
      } catch {
        /* ignore */
      }

      if (onProgress) onProgress("Ready");
    } catch (error) {
      console.error("Failed to load model:", error);
      throw error;
    }
  }

  private async handleSessionWait(
    onProgress?: (status: string, progress?: number) => void
  ): Promise<void> {
    const UNLOAD_KEY = "__texpen_unloading__";
    const LOADING_KEY = "__texpen_loading__";

    if (typeof sessionStorage !== "undefined") {
      const unloadTime = sessionStorage.getItem(UNLOAD_KEY);
      const wasLoading = sessionStorage.getItem(LOADING_KEY);

      // If previous page was in the middle of loading, we need a longer delay
      if (wasLoading) {
        if (onProgress)
          onProgress("Waiting for cleanup...");
        await new Promise((resolve) => setTimeout(resolve, 3000));
        sessionStorage.removeItem(LOADING_KEY);
      } else if (unloadTime) {
        const elapsed = Date.now() - parseInt(unloadTime, 10);
        if (elapsed < 3000) {
          if (onProgress)
            onProgress("Cleaning up previous session...");
          const waitTime = Math.min(
            1500,
            Math.max(500, 1500 - elapsed)
          );
          await new Promise((resolve) => setTimeout(resolve, waitTime));
        }
      }
      sessionStorage.removeItem(UNLOAD_KEY);

      // Mark that we're starting to load
      sessionStorage.setItem(LOADING_KEY, Date.now().toString());
    }
  }

  public async infer(
    imageBlob: Blob,
    options: SamplingOptions
  ): Promise<InferenceResult> {
    // Default to num_beams=1 if not specified and not sampling
    if (!options.num_beams && !options.do_sample) {
      options.num_beams = 1;
    }
    return this.queue.infer(imageBlob, options);
  }

  private async runInference(
    req: InferenceRequest,
    signal: AbortSignal
  ): Promise<void> {
    let pixelValues: Tensor | null = null;
    let debugImage = "";

    const isDev = (import.meta as unknown as { env: { DEV: boolean } }).env?.DEV ?? false;
    const timings: {
      preprocess?: number;
      generation?: number;
      total?: number;
    } = {};
    const startTotal = performance.now();

    try {
      if (!this.model || !this.tokenizer) {
        await this.init();
      }
      if (signal.aborted) throw new Error("Aborted");

      // 1) Preprocess image -> pixelValues
      const startPreprocess = performance.now();
      const { tensor, debugImage: dbgImg } = await preprocess(
        req.blob
      );
      pixelValues = tensor;
      debugImage = dbgImg;
      timings.preprocess = performance.now() - startPreprocess;

      if (req.options.onPreprocess) {
        req.options.onPreprocess(debugImage);
      }

      if (signal.aborted) throw new Error("Aborted");

      // 2) Generation config (max tokens, repetition penalty, etc.)
      const generationConfig = getGenerationConfig(
        this.dtype,
        this.tokenizer!
      );
      const repetitionPenalty =
        generationConfig.repetition_penalty || 1.0;

      // 3) Generate candidates using hybrid strategy
      const startGeneration = performance.now();

      const candidates = await this.generateCandidates(
        pixelValues,
        generationConfig,
        repetitionPenalty,
        req.options,
        signal
      );

      timings.generation = performance.now() - startGeneration;

      if (signal.aborted) throw new Error("Aborted");

      // 4) Post-process LaTeX
      const processedCandidates = candidates.map((c) => this.postprocess(c));
      timings.total = performance.now() - startTotal;

      if (isDev) {
        console.log(
          `[InferenceService] Timing: ` +
          `preprocess=${timings.preprocess?.toFixed(
            1
          )}ms, ` +
          `generation=${timings.generation?.toFixed(
            1
          )}ms, ` +
          `total=${timings.total?.toFixed(1)}ms`
        );
      }

      req.resolve({
        latex: processedCandidates[0] || "",
        candidates: processedCandidates,
        debugImage,
      });
    } catch (e) {
      const err = e as Error;
      if (err.message === "Skipped") {
        req.reject(err);
      } else if (err.message === "Aborted" || signal.aborted) {
        console.warn("[InferenceService] Inference aborted.");
        req.reject(new Error("Aborted"));
      } else {
        console.error("[InferenceService] Error:", e);
        req.reject(e);
      }
    } finally {
      if (pixelValues) pixelValues.dispose();
      // isInferring is handled by queue
    }
  }

  private async generateCandidates(
    pixelValues: Tensor,
    generationConfig: ReturnType<typeof getGenerationConfig>,
    repetitionPenalty: number,
    options: SamplingOptions,
    signal: AbortSignal
  ): Promise<string[]> {
    const isDev = (import.meta as unknown as { env: { DEV: boolean } }).env?.DEV ?? false;
    let candidates: string[] = [];

    const doSample = options.do_sample || false;
    const effectiveNumBeams = options.num_beams || 1;

    // OPTIMIZATION: If only 1 candidate is requested, force greedy decoding even if sampling is enabled.
    if (effectiveNumBeams === 1) {
      // effectiveNumBeams is already 1, effectively disabling specialized beam search paths
      // We handle the actual forced greedy logic in the conditional below
    }

    // Hybrid strategy:
    // 1. Greedy (num_beams=1) -> Use optimal transformers.js generate
    // 2. Sampling (do_sample=true && num_beams>1) -> Use transformers.js generate with manual loop
    // 3. Beam Search (num_beams>1 && do_sample=false) -> Use custom beam search

    // Determine if we should treat this as a "generate" (greedy/sample) or "beam search" call
    // Force greedy if num_beams=1 even if doSample is true (optimization)
    const useGenerate = (doSample && effectiveNumBeams > 1) || effectiveNumBeams === 1;

    if (useGenerate) {
      const generateOptions: Record<string, unknown> = {
        inputs: pixelValues,
        max_new_tokens: generationConfig.max_new_tokens,
        repetition_penalty: repetitionPenalty,
        decoder_start_token_id: generationConfig.decoder_start_token_id,
      };

      if (doSample && effectiveNumBeams > 1) {
        generateOptions.do_sample = true;
        generateOptions.temperature = options.temperature;
        generateOptions.top_k = options.top_k;
        generateOptions.top_p = options.top_p;

        if (isDev) {
          console.log('[InferenceService] Sampling options:', generateOptions);
        }

        // Manual loop to ensure we get multiple candidates
        const promises = [];
        for (let i = 0; i < effectiveNumBeams; i++) {
          promises.push(this.model!.generate({ ...generateOptions }));
        }

        const results = await Promise.all(promises);

        for (const outputTokenIds of results) {
          const decoded = this.tokenizer!.batch_decode(outputTokenIds, {
            skip_special_tokens: true,
          });
          candidates.push(...decoded);
        }
      } else {
        // Greedy or Single Sample (optimized to greedy if beams=1)
        // If effectiveNumBeams === 1, we just run generate once.
        // By default do_sample is false in generateOptions unless we set it.
        // We intentionally DO NOT set do_sample=true if effectiveNumBeams=1 to force greedy optimization.


        const outputTokenIds = await this.model!.generate(generateOptions) as Tensor;
        const decoded = this.tokenizer!.batch_decode(outputTokenIds, {
          skip_special_tokens: true,
        });
        candidates = decoded;
      }

      if (signal.aborted) throw new Error("Aborted");
    } else {
      // Batched beam search for n > 1 (Deterministic)
      candidates = await beamSearch(
        this.model!,
        this.tokenizer!,
        pixelValues,
        effectiveNumBeams,
        signal,
        generationConfig.max_new_tokens,
        repetitionPenalty,
        generationConfig.decoder_start_token_id
      );
    }

    return candidates;
  }


  private postprocess(latex: string): string {
    let processed = removeStyle(latex);
    processed = addNewlines(processed);
    return processed;
  }

  public async dispose(force: boolean = false): Promise<void> {
    if (this.queue.getIsInferring() && !force) {
      console.warn(
        "Attempting to dispose model while inference is in progress. Ignoring (unless forced)."
      );
      return;
    }

    // Increment generation to invalidate any pending inits
    this.disposalGeneration++;

    await this.queue.dispose();

    if (this.model) {
      if (
        "dispose" in this.model &&
        typeof this.model.dispose === "function"
      ) {
        try {
          await this.model.dispose();
        } catch (e) {
          console.warn("Error disposing model:", e);
        }
      }
      this.model = null;
    }
    this.tokenizer = null;
    this.initPromise = null;
  }

  /**
   * Synchronous disposal for use in beforeunload handlers.
   * This fires and forgets the cleanup - doesn't wait for async operations.
   */
  public disposeSync(): void {
    // Increment generation to invalidate any pending inits
    this.disposalGeneration++;

    // Best effort sync disposal of queue (might not be full, but Queue doesn't support sync fully)
    // But we can at least abort controllers if we had access.
    // In fact, queue.dispose() is async.
    // We can at least clear the model.

    // We can't easily wait for queue disposal sync.
    // Trigger it async but don't wait.
    this.queue.dispose().catch(() => { });


    // Fire and forget the model disposal - don't await
    if (this.model) {
      const modelRef = this.model;
      this.model = null;
      if (
        "dispose" in modelRef &&
        typeof modelRef.dispose === "function"
      ) {
        // Trigger dispose but don't wait - browser is unloading
        Promise.resolve().then(() => {
          try {
            modelRef.dispose();
          } catch {
            // Ignore - page is unloading anyway
          }
        });
      }
    }

    this.tokenizer = null;
    this.initPromise = null;
  }
}

// Use a global singleton stored on window to survive HMR and prevent duplicates
declare global {
  interface Window {
    __texpen_inference_service__?: InferenceService;
  }
}

function getOrCreateInstance(): InferenceService {
  // In browser, use window-based singleton
  if (typeof window !== "undefined") {
    if (!window.__texpen_inference_service__) {
      window.__texpen_inference_service__ = new (InferenceService as unknown as new () => InferenceService)();
    }
    return window.__texpen_inference_service__;
  }
  // Fallback for non-browser (SSR/tests)
  return InferenceService.getInstance();
}

export const inferenceService = getOrCreateInstance();

// Cleanup on page unload (F5 refresh, tab close, etc.)
// Use disposeSync since beforeunload doesn't wait for async operations
if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", () => {
    // Mark that we're unloading so the next init can add a delay
    try {
      sessionStorage.setItem(
        "__texpen_unloading__",
        Date.now().toString()
      );
    } catch {
      /* ignore storage errors */
    }
    getOrCreateInstance().disposeSync();
  });
}

// HMR Cleanup - dispose the model when this module is hot-reloaded
if ((import.meta as unknown as { hot: { dispose: (cb: () => void) => void } }).hot) {
  (import.meta as unknown as { hot: { dispose: (cb: () => void) => void } }).hot.dispose(() => {
    getOrCreateInstance().dispose(true);
  });
}
