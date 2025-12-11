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
  INFERENCE_CONFIG,
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
  private dtype: string = INFERENCE_CONFIG.DEFAULT_QUANTIZATION;
  private currentModelId: string = INFERENCE_CONFIG.MODEL_ID;
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
          (this.model as any).config.dtype !== options.dtype) ||
        (options.device &&
          (this.model as any).config.device !== options.device) ||
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
      let device =
        options.device || (webgpuAvailable ? "webgpu" : "wasm");
      let dtype =
        options.dtype ||
        (webgpuAvailable
          ? INFERENCE_CONFIG.DEFAULT_QUANTIZATION
          : "q8");

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
          dtype,
          onProgress
        ),
      ]);

      const { model, dtype: finalDtype } = modelResult;

      // CHECK GENERATION AGAIN
      if (this.disposalGeneration !== localGeneration) {
        if (model && "dispose" in model) {
          await (model as any).dispose();
        }
        return;
      }

      this.tokenizer = tokenizer;
      this.model = model;
      this.dtype = finalDtype;

      // Clear loading flag - we're done
      try {
        sessionStorage.removeItem("__texpen_loading__");
      } catch (e) {
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
          onProgress("Waiting for GPU cleanup...");
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

    const isDev = (import.meta as any).env?.DEV ?? false;
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

      if (signal.aborted) throw new Error("Aborted");

      // 2) Generation config (max tokens, repetition penalty, etc.)
      const generationConfig = getGenerationConfig(
        this.dtype,
        this.tokenizer!
      );
      const repetitionPenalty =
        generationConfig.repetition_penalty || 1.0;

      const doSample = req.options.do_sample || false;
      const effectiveNumBeams = req.options.num_beams || 1;

      // 3) Hybrid Decoding Strategy
      // If do_sample is true, use model.generate (transformers.js supports sampling)
      // If numCandidates == 1, use optimized greedy from transformers.js directly (faster, more stable)
      // If numCandidates > 1, use custom beam search (until transformers.js supports num_return_sequences for this path)
      let candidates: string[] = [];

      // OPTIMIZATION: If only 1 candidate is requested, force greedy decoding even if sampling is enabled.
      // This is generally faster and desired by the user for single-candidate generation.
      if (effectiveNumBeams === 1) {
        // Force disable sampling for single candidate optimization
        // doSample = false; // const assignment - need to handle largely logic
      }

      if ((doSample && effectiveNumBeams > 1) || effectiveNumBeams === 1) {
        const startGeneration = performance.now();

        const generateOptions: any = {
          inputs: pixelValues,
          max_new_tokens: generationConfig.max_new_tokens,
          repetition_penalty: repetitionPenalty,
          decoder_start_token_id: generationConfig.decoder_start_token_id,
        };

        if (doSample && effectiveNumBeams > 1) {
          generateOptions.do_sample = true;
          generateOptions.temperature = req.options.temperature;
          generateOptions.top_k = req.options.top_k;
          generateOptions.top_p = req.options.top_p;

          if (isDev) {
            console.log('[InferenceService] Sampling options:', generateOptions);
          }

          if (effectiveNumBeams > 1) {
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
            // Single sample
            const outputTokenIds = await this.model!.generate(generateOptions);
            const decoded = this.tokenizer!.batch_decode(outputTokenIds, {
              skip_special_tokens: true,
            });
            candidates = decoded;
          }
        } else {
          // Greedy
          // @ts-ignore
          const outputTokenIds = await this.model!.generate(generateOptions);
          const decoded = this.tokenizer!.batch_decode(outputTokenIds, {
            skip_special_tokens: true,
          });
          candidates = decoded;
        }

        timings.generation = performance.now() - startGeneration;

        if (signal.aborted) throw new Error("Aborted");
        // candidates populated
      } else {
        // Batched beam search for n > 1 (Deterministic)
        const startGeneration = performance.now();
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
        timings.generation = performance.now() - startGeneration;
      }

      if (signal.aborted) throw new Error("Aborted");

      // 4) Post-process LaTeX
      candidates = candidates.map((c) => this.postprocess(c));
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
          `total=${timings.total?.toFixed(1)}ms` +
          ` (mode: ${doSample ? "sampling" : (effectiveNumBeams === 1 ? "greedy/native" : "beam/custom")})`
        );
      }

      req.resolve({
        latex: candidates[0] || "",
        candidates,
        debugImage,
      });
    } catch (e: any) {
      if (e.message === "Skipped") {
        req.reject(e);
      } else if (e.message === "Aborted" || signal.aborted) {
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
        typeof (this.model as any).dispose === "function"
      ) {
        try {
          await (this.model as any).dispose();
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
        typeof (modelRef as any).dispose === "function"
      ) {
        // Trigger dispose but don't wait - browser is unloading
        Promise.resolve().then(() => {
          try {
            (modelRef as any).dispose();
          } catch (e) {
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
      window.__texpen_inference_service__ = new (InferenceService as any)();
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
    } catch (e) {
      /* ignore storage errors */
    }
    getOrCreateInstance().disposeSync();
  });
}

// HMR Cleanup - dispose the model when this module is hot-reloaded
if ((import.meta as any).hot) {
  (import.meta as any).hot.dispose(() => {
    getOrCreateInstance().dispose(true);
  });
}
