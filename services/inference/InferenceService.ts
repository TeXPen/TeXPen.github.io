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
        const device = options.device || (webgpuAvailable ? 'webgpu' : 'wasm');
        const dtype = options.dtype || (webgpuAvailable ? INFERENCE_CONFIG.DEFAULT_QUANTIZATION : 'q8');
        this.dtype = dtype;

        if (onProgress) onProgress(`Loading model with ${device} (${dtype})... (this may take a while)`);

        const sessionOptions = getSessionOptions(device, dtype);

        this.model = await AutoModelForVision2Seq.from_pretrained(INFERENCE_CONFIG.MODEL_ID, sessionOptions) as VisionEncoderDecoderModel;

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

  private pendingRequest: {
    blob: Blob;
    numCandidates: number;
    resolve: (value: InferenceResult | PromiseLike<InferenceResult>) => void;
    reject: (reason?: any) => void;
  } | null = null;

  public async infer(imageBlob: Blob, numCandidates: number = 1): Promise<InferenceResult> {
    return new Promise((resolve, reject) => {
      // 1. If there's already a pending request, reject it (Skipped)
      if (this.pendingRequest) {
        this.pendingRequest.reject(new Error("Skipped"));
      }

      // 2. Set new pending request
      this.pendingRequest = {
        blob: imageBlob,
        numCandidates,
        resolve,
        reject
      };

      // 3. Wake up the loop if it's waiting
      if (this.wakeQueuePromise) {
        this.wakeQueuePromise();
        this.wakeQueuePromise = null;
      }

      // 4. Ensure queue processing is running
      if (!this.isProcessingQueue) {
        this.processQueue();
      }
    });
  }

  private async processQueue() {
    this.isProcessingQueue = true;

    try {
      while (this.pendingRequest) {
        // If an inference is currently running, we need to decide: Wait or Abort?
        if (this.currentInferencePromise && this.isInferring) {
          console.log('[InferenceService] New request pending. Allowing current inference 3s grace period...');

          let timedOut = false;
          const timeoutPromise = new Promise(resolve => setTimeout(() => {
            timedOut = true;
            resolve('timeout');
          }, 3000));

          // Wait for either the inference to finish naturally OR the 3s timer
          await Promise.race([this.currentInferencePromise, timeoutPromise]);

          if (timedOut && this.isInferring) {
            console.warn('[InferenceService] 3s grace period expired. Aborting current inference.');
            this.abortController?.abort();
            // Critical: Must wait for it to actually cleanup before starting next
            try { await this.currentInferencePromise; } catch (e) { /* ignore */ }
          }
        }

        // Double check pendingRequest still exists (it should)
        if (!this.pendingRequest) break;

        // Pop the request
        const req = this.pendingRequest;
        this.pendingRequest = null;

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

            // If pending request exists, wake up the loop if it was stuck expecting more?
            // Actually, since this promise resolves, the loop below (await this.currentInferencePromise)
            // will unblock, allowing the loop to continue.
            if (this.wakeQueuePromise) {
              this.wakeQueuePromise();
              this.wakeQueuePromise = null;
            }
          }
        })();

        // Wait for this inference to complete OR for a new request to come in
        // If a new request comes in, we want to wake up and race it against the timer.
        // We race: currentInferencePromise VS newRequestSignal

        if (this.pendingRequest) {
          // Immediately loop back to check race logic
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
