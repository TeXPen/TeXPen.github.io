import { AutoTokenizer, AutoModelForVision2Seq, PreTrainedModel, PreTrainedTokenizer, Tensor } from '@huggingface/transformers';
import { removeStyle, addNewlines } from '../latexUtils';
import { preprocess } from './imagePreprocessing';
import { beamSearch } from './beamSearch';
import { isWebGPUAvailable } from '../../utils/env';
import { INFERENCE_CONFIG, getSessionOptions, getGenerationConfig } from './config';
import { InferenceOptions, InferenceResult, VisionEncoderDecoderModel } from './types';

export class InferenceService {
  private model: VisionEncoderDecoderModel | null = null;
  private tokenizer: PreTrainedTokenizer | null = null;
  private static instance: InferenceService;
  private isInferring: boolean = false;
  private dtype: string = INFERENCE_CONFIG.DEFAULT_QUANTIZATION;
  private currentModelId: string = INFERENCE_CONFIG.MODEL_ID;
  private initPromise: Promise<void> | null = null;

  private constructor() { }

  public static getInstance(): InferenceService {
    if (!InferenceService.instance) {
      InferenceService.instance = new InferenceService();
    }
    return InferenceService.instance;
  }

  private disposalGeneration: number = 0;
  private loadingMutex: Promise<void> = Promise.resolve();

  public async init(onProgress?: (status: string, progress?: number) => void, options: InferenceOptions = {}): Promise<void> {
    // If initialization is already in progress, return the existing promise
    if (this.initPromise) {
      return this.initPromise;
    }

    // Append our actual work to the mutex chain
    const work = async () => {
      const localGeneration = this.disposalGeneration;
      // console.log('[DEBUG] Entering work. model:', !!this.model, 'currentModelId:', this.currentModelId, 'optId:', options.modelId);

      if (this.model && this.tokenizer) {
        // If the model is already loaded, but the quantization or device is different, we need to dispose and reload.
        if ((options.dtype && (this.model as any).config.dtype !== options.dtype) ||
          (options.device && (this.model as any).config.device !== options.device) ||
          (options.modelId && this.currentModelId !== options.modelId)) {

          if (this.isInferring) {
            console.warn("Changing model settings while inference is in progress.");
            throw new Error("Cannot change model settings while an inference is in progress.");
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

        // Check if we just refreshed OR a previous load was interrupted
        const UNLOAD_KEY = '__texpen_unloading__';
        const LOADING_KEY = '__texpen_loading__';

        if (typeof sessionStorage !== 'undefined') {
          const unloadTime = sessionStorage.getItem(UNLOAD_KEY);
          const wasLoading = sessionStorage.getItem(LOADING_KEY);

          // If previous page was in the middle of loading, we need a longer delay
          // because the GPU is likely still processing the interrupted load
          if (wasLoading) {
            if (onProgress) onProgress('Waiting for GPU cleanup...');
            await new Promise(resolve => setTimeout(resolve, 3000));
            sessionStorage.removeItem(LOADING_KEY);
          } else if (unloadTime) {
            const elapsed = Date.now() - parseInt(unloadTime, 10);
            if (elapsed < 3000) {
              if (onProgress) onProgress('Cleaning up previous session...');
              const waitTime = Math.min(1500, Math.max(500, 1500 - elapsed));
              await new Promise(resolve => setTimeout(resolve, waitTime));
            }
          }
          sessionStorage.removeItem(UNLOAD_KEY);

          // Mark that we're starting to load
          sessionStorage.setItem(LOADING_KEY, Date.now().toString());
        }

        const webgpuAvailable = await isWebGPUAvailable();
        let device = options.device || (webgpuAvailable ? 'webgpu' : 'wasm');
        let dtype = options.dtype || (webgpuAvailable ? INFERENCE_CONFIG.DEFAULT_QUANTIZATION : 'q8');
        // Update current ID if provided, otherwise keep existing (or default on first run)
        if (options.modelId) {
          this.currentModelId = options.modelId;
        }

        this.dtype = dtype;

        if (onProgress) onProgress(`Loading model ${this.currentModelId} (${device}, ${dtype})...`);

        const sessionOptions = getSessionOptions(device, dtype);

        // Load Tokenizer and Model in parallel
        const [tokenizer, model] = await Promise.all([
          AutoTokenizer.from_pretrained(this.currentModelId),
          (async () => {
            try {
              return await AutoModelForVision2Seq.from_pretrained(this.currentModelId, sessionOptions) as VisionEncoderDecoderModel;
            } catch (loadError: any) {
              // Check if this is a WebGPU buffer size / memory error
              const isWebGPUMemoryError = loadError?.message?.includes('createBuffer') ||
                loadError?.message?.includes('mappedAtCreation') ||
                loadError?.message?.includes('too large for the implementation') ||
                loadError?.message?.includes('GPUDevice');

              if (isWebGPUMemoryError && device === 'webgpu') {
                console.warn('[InferenceService] WebGPU buffer allocation failed, falling back to WASM...');
                if (onProgress) onProgress('WebGPU memory limit hit. Switching to WASM...');

                // Retry with WASM
                device = 'wasm';
                dtype = 'q8';
                this.dtype = dtype;

                const fallbackSessionOptions = getSessionOptions(device, dtype);
                return await AutoModelForVision2Seq.from_pretrained(this.currentModelId, fallbackSessionOptions) as VisionEncoderDecoderModel;
              } else {
                throw loadError;
              }
            }
          })()
        ]);

        // CHECK GENERATION AGAIN
        if (this.disposalGeneration !== localGeneration) {
          if (model && 'dispose' in model) {
            await (model as any).dispose();
          }
          return;
        }

        this.tokenizer = tokenizer;
        this.model = model;

        // Clear loading flag - we're done
        try {
          sessionStorage.removeItem('__texpen_loading__');
        } catch (e) { /* ignore */ }

        if (onProgress) onProgress('Ready');
      } catch (error) {
        console.error('Failed to load model:', error);
        throw error;
      }
    };

    // Update the mutex - chain work onto it
    this.loadingMutex = this.loadingMutex.then(work).catch(err => {
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

  private abortController: AbortController | null = null;
  private currentInferencePromise: Promise<void> | null = null;
  private isProcessingQueue: boolean = false;
  private wakeQueuePromise: ((value: void) => void) | null = null;

  private pendingRequestTimestamp: number = 0;
  private graceTimeoutId: ReturnType<typeof setTimeout> | null = null;

  private pendingRequest: {
    blob: Blob;
    numCandidates: number;
    resolve: (value: InferenceResult | PromiseLike<InferenceResult>) => void;
    reject: (reason?: any) => void;
  } | null = null;

  private static readonly GRACE_PERIOD_MS = 3000;

  public async infer(imageBlob: Blob, numCandidates: number = 1): Promise<InferenceResult> {
    return new Promise((resolve, reject) => {
      if (this.pendingRequest) {
        this.pendingRequest.reject(new Error("Skipped"));
      }

      if (this.graceTimeoutId) {
        clearTimeout(this.graceTimeoutId);
        this.graceTimeoutId = null;
      }

      this.pendingRequest = {
        blob: imageBlob,
        numCandidates,
        resolve,
        reject
      };
      this.pendingRequestTimestamp = Date.now();

      if (this.isInferring && this.abortController) {
        console.log('[InferenceService] New request while inferring. Starting 3s grace period from now...');
        this.graceTimeoutId = setTimeout(() => {
          if (this.isInferring && this.abortController) {
            console.warn('[InferenceService] 3s grace period expired. Aborting current inference.');
            this.abortController.abort();
          }
          this.graceTimeoutId = null;
        }, InferenceService.GRACE_PERIOD_MS);
      }

      if (this.wakeQueuePromise) {
        this.wakeQueuePromise();
        this.wakeQueuePromise = null;
      }

      if (!this.isProcessingQueue) {
        this.processQueue();
      }
    });
  }

  private async processQueue() {
    this.isProcessingQueue = true;

    try {
      while (this.pendingRequest) {
        if (this.currentInferencePromise && this.isInferring) {
          console.log('[InferenceService] Waiting for current inference to finish or abort...');
          try { await this.currentInferencePromise; } catch (e) { /* ignore */ }
        }

        if (!this.pendingRequest) break;

        const req = this.pendingRequest;
        this.pendingRequest = null;
        this.pendingRequestTimestamp = 0;

        if (this.graceTimeoutId) {
          clearTimeout(this.graceTimeoutId);
          this.graceTimeoutId = null;
        }

        this.isInferring = true;
        this.abortController = new AbortController();
        const signal = this.abortController.signal;

        this.currentInferencePromise = (async () => {
          let pixelValues: Tensor | null = null;
          let debugImage: string = '';

          try {
            if (!this.model || !this.tokenizer) {
              await this.init();
            }
            if (signal.aborted) throw new Error("Aborted");

            const { tensor, debugImage: dbgImg } = await preprocess(req.blob);
            pixelValues = tensor;
            debugImage = dbgImg;

            if (signal.aborted) throw new Error("Aborted");

            const generationConfig = getGenerationConfig(this.dtype, this.tokenizer!);
            const repetitionPenalty = generationConfig.repetition_penalty || 1.0;
            const effectiveNumBeams = req.numCandidates;

            let candidates = await beamSearch(
              this.model!,
              this.tokenizer!,
              pixelValues,
              effectiveNumBeams,
              signal,
              generationConfig.max_new_tokens,
              repetitionPenalty
            );

            candidates = candidates.map(c => this.postprocess(c));

            if (signal.aborted) throw new Error("Aborted");

            req.resolve({
              latex: candidates[0] || '',
              candidates,
              debugImage
            });

          } catch (e: any) {
            if (e.message === 'Skipped') {
              req.reject(e);
            } else if (e.message === 'Aborted' || signal.aborted) {
              console.warn('[InferenceService] Inference aborted.');
              req.reject(new Error("Aborted"));
            } else {
              console.error('[InferenceService] Error:', e);
              req.reject(e);
            }
          } finally {
            if (pixelValues) pixelValues.dispose();
            this.isInferring = false;
            this.abortController = null;
            this.currentInferencePromise = null;

            if (this.wakeQueuePromise) {
              this.wakeQueuePromise();
              this.wakeQueuePromise = null;
            }
          }
        })();

        if (this.pendingRequest) {
          continue;
        } else {
          await Promise.race([
            this.currentInferencePromise,
            new Promise<void>(resolve => { this.wakeQueuePromise = resolve; })
          ]);
        }
      }
    } finally {
      this.isProcessingQueue = false;
    }
  }

  private postprocess(latex: string): string {
    let processed = removeStyle(latex);
    processed = addNewlines(processed);
    return processed;
  }

  public async dispose(force: boolean = false): Promise<void> {
    if (this.isInferring && !force) {
      console.warn("Attempting to dispose model while inference is in progress. Ignoring (unless forced).");
      return;
    }

    // Increment generation to invalidate any pending inits
    this.disposalGeneration++;

    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }

    if (this.pendingRequest) {
      this.pendingRequest.reject(new Error("Aborted"));
      this.pendingRequest = null;
    }

    this.isInferring = false;

    if (this.model) {
      if ('dispose' in this.model && typeof (this.model as any).dispose === 'function') {
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

    // Abort any running inference immediately
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }

    // Reject pending requests
    if (this.pendingRequest) {
      try {
        this.pendingRequest.reject(new Error("Aborted"));
      } catch (e) { /* ignore */ }
      this.pendingRequest = null;
    }

    this.isInferring = false;

    // Fire and forget the model disposal - don't await
    if (this.model) {
      const modelRef = this.model;
      this.model = null;
      if ('dispose' in modelRef && typeof (modelRef as any).dispose === 'function') {
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
  if (typeof window !== 'undefined') {
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
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    // Mark that we're unloading so the next init can add a delay
    try {
      sessionStorage.setItem('__texpen_unloading__', Date.now().toString());
    } catch (e) { /* ignore storage errors */ }
    getOrCreateInstance().disposeSync();
  });
}

// HMR Cleanup - dispose the model when this module is hot-reloaded
if ((import.meta as any).hot) {
  (import.meta as any).hot.dispose(() => {
    getOrCreateInstance().dispose(true);
  });
}
