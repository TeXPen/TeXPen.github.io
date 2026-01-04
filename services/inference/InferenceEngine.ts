import {
  AutoTokenizer,
  PreTrainedTokenizer,
  Tensor,
} from "@huggingface/transformers";
import { removeStyle, addNewlines } from "../../utils/latex";
import { preprocess } from "./imagePreprocessing";
import { beamSearch } from "./beamSearch";
import { isWebGPUAvailable } from "../../utils/env";
import {
  MODEL_CONFIG,
  getSessionOptions,
  getGenerationConfig,
} from "./config";
import {
  InferenceResult,
  VisionEncoderDecoderModel,
  SamplingOptions,
  Quantization,
  InferenceOptions,
} from "./types";
import { QuantizationConfig } from "./config";

export class InferenceEngine {
  private model: VisionEncoderDecoderModel | null = null;
  private tokenizer: PreTrainedTokenizer | null = null;

  private currentModelId: string = MODEL_CONFIG.ID;
  private currentQuantization: Quantization = 'int8';
  private currentEncoderQuantization?: Quantization;
  private currentDecoderQuantization?: Quantization;
  private initPromise: Promise<void> | null = null;
  private isLoading: boolean = false;

  private disposalGeneration: number = 0;
  private loadingMutex: Promise<void> = Promise.resolve();

  public async init(
    onProgress?: (status: string, progress?: number) => void,
    options: InferenceOptions = {}
  ): Promise<void> {
    // Robust guard: if already loading, wait for current load
    if (this.isLoading && this.initPromise) {
      return this.initPromise;
    }

    // If initialization is already in progress, return the existing promise
    if (this.initPromise) {
      return this.initPromise;
    }

    this.isLoading = true;

    // Capture the generation AT THE MOMENT OF CALL
    const capturedGeneration = this.disposalGeneration;

    // Append our actual work to the mutex chain
    const work = async () => {
      await this.runLoadingSequence(
        capturedGeneration,
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
      this.isLoading = false;
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
      // If the model is already loaded, but the device is different, we need to dispose and reload.
      if (
        (options.device &&
          this.model.config.device !== options.device) ||
        (options.modelId && this.currentModelId !== options.modelId) ||
        (options.quantization && this.currentQuantization !== options.quantization) ||
        (options.encoderQuantization && this.currentEncoderQuantization !== options.encoderQuantization) ||
        (options.decoderQuantization && this.currentDecoderQuantization !== options.decoderQuantization)
      ) {
        // Internal reconfiguration: Dispose OLD model but do NOT increment generation.

        // 1. Dispose model
        if (this.model && "dispose" in this.model) {
          try {
            await this.model.dispose();
          } catch (e) {
            console.warn("Error disposing model during reconfiguration:", e);
          }
        }
        this.model = null;
        this.tokenizer = null;

        // 2. Verify we are still valid before proceeding to load new one
        if (this.disposalGeneration !== localGeneration) {
          return;
        }
      } else {
        // Already loaded with correct settings
        return;
      }
    }

    try {
      // CHECK GENERATION before starting heavy work
      if (this.disposalGeneration !== localGeneration) return;

      // Note: Session wait logic (sessionStorage) is UI/Main thread specific.
      // We skip it here or handle it differently if needed, but for Worker we mostly ignore cross-tab coordination for now
      // or rely on the main thread to tell us when to start.
      // For now, removing `handleSessionWait` as it relies on DOM `sessionStorage`.

      // Determine initial device preference
      const webgpuAvailable = await isWebGPUAvailable();
      const preferredDevice =
        options.device || (webgpuAvailable ? MODEL_CONFIG.PROVIDERS.WEBGPU : MODEL_CONFIG.PROVIDERS.WASM);

      // Update current ID if provided, otherwise keep existing (or default on first run)
      if (options.modelId) {
        this.currentModelId = options.modelId;
      }
      if (options.quantization) {
        this.currentQuantization = options.quantization;
      }
      if (options.encoderQuantization) {
        this.currentEncoderQuantization = options.encoderQuantization;
      }
      if (options.decoderQuantization) {
        this.currentDecoderQuantization = options.decoderQuantization;
      }

      if (onProgress)
        onProgress(
          `Loading model ${this.currentModelId} (${preferredDevice})...`
        );

      const quantConfig: QuantizationConfig = {
        overall: this.currentQuantization,
        encoder: this.currentEncoderQuantization,
        decoder: this.currentDecoderQuantization,
      };

      console.log(`[InferenceEngine] Initializing with device: ${preferredDevice}, Quantization: ${JSON.stringify(quantConfig)}`);

      const sessionOptions = getSessionOptions(preferredDevice, quantConfig);

      // Pre-download heavy model files using ModelLoader
      const { modelLoader } = await import("./ModelLoader");
      await modelLoader.preDownloadModels(
        this.currentModelId,
        sessionOptions,
        onProgress
      );

      if (onProgress) onProgress('Initializing model (compiling shaders)...', 0);

      // Load Tokenizer and Model in parallel
      const [tokenizer, modelResult] = await Promise.all([
        AutoTokenizer.from_pretrained(this.currentModelId),
        modelLoader.loadModelWithFallback(
          this.currentModelId,
          preferredDevice,
          onProgress
        ),
      ]);

      const { model } = modelResult;

      // CHECK GENERATION AGAIN
      if (this.disposalGeneration !== localGeneration) {
        if (model && "dispose" in model) {
          await model.dispose();
        }
        return;
      }

      this.tokenizer = tokenizer;
      this.model = model;

      if (onProgress) onProgress("Ready");
    } catch (error) {
      console.error("Failed to load model:", error);
      throw error;
    }
  }

  public async infer(
    imageBlob: Blob,
    options: SamplingOptions,
    signal?: AbortSignal
  ): Promise<InferenceResult> {
    const isDev = (import.meta as unknown as { env: { DEV: boolean } }).env?.DEV ?? false;

    // Default to num_beams=1 if not specified and not sampling
    if (!options.num_beams && !options.do_sample) {
      options.num_beams = 1;
    }

    let pixelValues: Tensor | null = null;
    let debugImage = "";
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

      if (signal?.aborted) throw new Error("Aborted");

      // 1) Preprocess image -> pixelValues
      const startPreprocess = performance.now();
      const { tensor, debugImage: dbgImg } = await preprocess(
        imageBlob,
        !!options.onPreprocess
      );
      pixelValues = tensor;
      debugImage = dbgImg;
      timings.preprocess = performance.now() - startPreprocess;

      if (options.onPreprocess) {
        options.onPreprocess(debugImage);
      }

      if (signal?.aborted) throw new Error("Aborted");

      // 2) Generation config (max tokens, etc.)
      const generationConfig = getGenerationConfig(this.tokenizer!);

      // 3) Generate candidates using hybrid strategy
      const startGeneration = performance.now();

      const candidates = await this.generateCandidates(
        pixelValues,
        generationConfig,
        options,
        signal
      );

      timings.generation = performance.now() - startGeneration;

      if (signal?.aborted) throw new Error("Aborted");

      // 4) Post-process LaTeX
      const processedCandidates = candidates.map((c) => this.postprocess(c));
      timings.total = performance.now() - startTotal;

      if (isDev) {
        console.log(
          `[InferenceEngine] Timing: ` +
          `preprocess=${timings.preprocess?.toFixed(1)}ms, ` +
          `generation=${timings.generation?.toFixed(1)}ms, ` +
          `total=${timings.total?.toFixed(1)}ms`
        );
      }

      return {
        latex: processedCandidates[0] || "",
        candidates: processedCandidates,
        debugImage,
      };
    } finally {
      if (pixelValues) pixelValues.dispose();
    }
  }

  private async generateCandidates(
    pixelValues: Tensor,
    generationConfig: ReturnType<typeof getGenerationConfig>,
    options: SamplingOptions,
    signal?: AbortSignal
  ): Promise<string[]> {
    const isDev = (import.meta as unknown as { env: { DEV: boolean } }).env?.DEV ?? false;
    let candidates: string[] = [];

    const doSample = options.do_sample || false;
    const effectiveNumBeams = options.num_beams || 1;

    const useGenerate = (doSample && effectiveNumBeams > 1) || effectiveNumBeams === 1;

    if (useGenerate) {
      const generateOptions: Record<string, unknown> = {
        inputs: pixelValues,
        max_new_tokens: generationConfig.max_new_tokens,
        decoder_start_token_id: generationConfig.decoder_start_token_id,
      };

      if (doSample && effectiveNumBeams > 1) {
        generateOptions.do_sample = true;
        generateOptions.temperature = options.temperature;
        generateOptions.top_k = options.top_k;
        generateOptions.top_p = options.top_p;

        if (isDev) {
          console.log('[InferenceEngine] Sampling options:', generateOptions);
        }

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
        const outputTokenIds = await this.model!.generate(generateOptions) as Tensor;
        const decoded = this.tokenizer!.batch_decode(outputTokenIds, {
          skip_special_tokens: true,
        });
        candidates = decoded;
      }

      if (signal?.aborted) throw new Error("Aborted");
    } else {
      // Batched beam search for n > 1 (Deterministic)
      candidates = await beamSearch(
        this.model!,
        this.tokenizer!,
        pixelValues,
        effectiveNumBeams,
        signal || new AbortController().signal,
        generationConfig.max_new_tokens,
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

  public async dispose(_force: boolean = false): Promise<void> {
    // Increment generation to invalidate any pending inits
    this.disposalGeneration++;

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
    this.isLoading = false;
  }
}
