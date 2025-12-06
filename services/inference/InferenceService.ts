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
  private initPromise: Promise<void> | null = null;

  private constructor() { }

  public static getInstance(): InferenceService {
    if (!InferenceService.instance) {
      InferenceService.instance = new InferenceService();
    }
    return InferenceService.instance;
  }

  public async init(onProgress?: (status: string, progress?: number) => void, options: InferenceOptions = {}): Promise<void> {
    // If initialization is already in progress, return the existing promise
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = (async () => {
      if (this.model && this.tokenizer) {
        // If the model is already loaded, but the quantization or device is different, we need to dispose and reload.
        if ((options.dtype && (this.model as any).config.dtype !== options.dtype) ||
          (options.device && (this.model as any).config.device !== options.device)) {
          if (this.isInferring) {
            console.warn("Changing model settings while inference is in progress. Waiting for it to finish or forceful disposal might occur.");
            // Ideally we should wait, but for now we proceed to dispose which checks isInferring
            // For safety in init, we might want to throw or wait. 
            // Current decision: Throw if inferring, same as before, but wrapped in promise.
            if (this.isInferring) {
              throw new Error("Cannot change model settings while an inference is in progress.");
            }
          }
          await this.dispose();
        } else {
          return;
        }
      }

      try {
        if (onProgress) onProgress('Loading tokenizer...');
        this.tokenizer = await AutoTokenizer.from_pretrained(INFERENCE_CONFIG.MODEL_ID);

        const webgpuAvailable = await isWebGPUAvailable();
        let device = options.device || (webgpuAvailable ? 'webgpu' : 'wasm');
        let dtype = options.dtype || (webgpuAvailable ? INFERENCE_CONFIG.DEFAULT_QUANTIZATION : 'q8');
        this.dtype = dtype;

        if (onProgress) onProgress(`Loading model with ${device} (${dtype})...`);

        const sessionOptions = getSessionOptions(device, dtype);

        try {
          this.model = await AutoModelForVision2Seq.from_pretrained(INFERENCE_CONFIG.MODEL_ID, sessionOptions) as VisionEncoderDecoderModel;
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
            dtype = 'q8'; // Use quantized for WASM performance
            this.dtype = dtype;

            const fallbackSessionOptions = getSessionOptions(device, dtype);
            this.model = await AutoModelForVision2Seq.from_pretrained(INFERENCE_CONFIG.MODEL_ID, fallbackSessionOptions) as VisionEncoderDecoderModel;
          } else {
            throw loadError;
          }
        }

        if (onProgress) onProgress('Ready');
      } catch (error) {
        console.error('Failed to load model:', error);
        throw error;
      }
    })();

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

  // Timestamp when the pending request was created - used for grace period calculation
  private pendingRequestTimestamp: number = 0;
  // Grace period timeout ID for the pending request
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
      // 1. If there's already a pending request, reject it (Skipped) - always keep only the latest
      if (this.pendingRequest) {
        this.pendingRequest.reject(new Error("Skipped"));
      }

      // 2. Clear any existing grace timeout since we have a new request
      if (this.graceTimeoutId) {
        clearTimeout(this.graceTimeoutId);
        this.graceTimeoutId = null;
      }

      // 3. Set new pending request with timestamp
      this.pendingRequest = {
        blob: imageBlob,
        numCandidates,
        resolve,
        reject
      };
      this.pendingRequestTimestamp = Date.now();

      // 4. If currently inferring, start the 3-second grace period timer immediately
      //    This timer starts NOW, from when the user finished their stroke
      if (this.isInferring && this.abortController) {
        console.log('[InferenceService] New request while inferring. Starting 3s grace period from now...');
        this.graceTimeoutId = setTimeout(() => {
          // Time's up - abort the current inference if it's still running
          if (this.isInferring && this.abortController) {
            console.warn('[InferenceService] 3s grace period expired. Aborting current inference.');
            this.abortController.abort();
          }
          this.graceTimeoutId = null;
        }, InferenceService.GRACE_PERIOD_MS);
      }

      // 5. Wake up the loop if it's waiting
      if (this.wakeQueuePromise) {
        this.wakeQueuePromise();
        this.wakeQueuePromise = null;
      }

      // 6. Ensure queue processing is running
      if (!this.isProcessingQueue) {
        this.processQueue();
      }
    });
  }

  private async processQueue() {
    this.isProcessingQueue = true;

    try {
      while (this.pendingRequest) {
        // If an inference is currently running, wait for it to complete or be aborted
        // Note: The grace period timer is already running from when infer() was called
        if (this.currentInferencePromise && this.isInferring) {
          console.log('[InferenceService] Waiting for current inference to finish or abort...');
          // Just wait - the grace timeout will handle aborting if needed
          try { await this.currentInferencePromise; } catch (e) { /* ignore */ }
        }

        // Double check pendingRequest still exists
        if (!this.pendingRequest) break;

        // Pop the request - take the LATEST one only
        const req = this.pendingRequest;
        this.pendingRequest = null;
        this.pendingRequestTimestamp = 0;

        // Clear the grace timeout since we're now processing this request
        if (this.graceTimeoutId) {
          clearTimeout(this.graceTimeoutId);
          this.graceTimeoutId = null;
        }

        // Start the inference
        this.isInferring = true;
        this.abortController = new AbortController();
        const signal = this.abortController.signal;

        // Create a promise wrapper for this inference
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

            // Wake up the loop if it was waiting for this inference to complete
            if (this.wakeQueuePromise) {
              this.wakeQueuePromise();
              this.wakeQueuePromise = null;
            }
          }
        })();

        // Wait for this inference to complete OR for a new request to come in
        if (this.pendingRequest) {
          // Immediately loop back - new request already waiting
          continue;
        } else {
          // Wait for completion or new request
          await Promise.race([
            this.currentInferencePromise,
            new Promise<void>(resolve => { this.wakeQueuePromise = resolve; })
          ]);
          // If woke up by new request, loop continues and hits the top 'if (pending)' block
          // If woke up by completion, loop continues, checks pending, if null, breaks.
        }
      }
    } finally {
      this.isProcessingQueue = false;
    }
  }

  private postprocess(latex: string): string {
    // 1. Remove style (bold, italic, etc.) - optional but recommended for cleaner output
    let processed = removeStyle(latex);

    // 2. Add newlines for readability
    processed = addNewlines(processed);

    return processed;
  }

  public async dispose(force: boolean = false): Promise<void> {
    if (this.isInferring && !force) {
      console.warn("Attempting to dispose model while inference is in progress. Ignoring (unless forced).");
      return;
    }

    // Force abort any running inference
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }

    // Reject any pending request
    if (this.pendingRequest) {
      this.pendingRequest.reject(new Error("Aborted"));
      this.pendingRequest = null;
    }

    // If loading is in progress, we can't easily cancel the promise, but we can reset the state.
    // Ideally we should await initPromise, but that might deadlock if dispose is called from within init (reconfig).
    // For now, we assume dispose logic cleans up what it can.

    this.isInferring = false; // Force reset state

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

    // Important: Clear the instance so next getInstance creates a fresh one? 
    // Or just keep the instance but empty?
    // Following existing pattern of clearing instance.
    (InferenceService as any).instance = null;
    this.initPromise = null;
  }
}

export const inferenceService = InferenceService.getInstance();
