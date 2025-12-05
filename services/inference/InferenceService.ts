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

  public async infer(imageBlob: Blob, numCandidates: number = 1): Promise<InferenceResult> {
    if (this.isInferring) {
      throw new Error("Another inference is already in progress.");
    }
    this.isInferring = true;

    let pixelValues: Tensor | null = null;
    let debugImage: string = '';

    try {
      if (!this.model || !this.tokenizer) {
        await this.init();
      }

      // 1. Preprocess
      const { tensor, debugImage: dbgImg } = await preprocess(imageBlob);
      pixelValues = tensor;
      debugImage = dbgImg;

      // 2. Generate candidates
      let candidates: string[];
      if (numCandidates <= 1) {
        const generationConfig = getGenerationConfig(this.dtype, this.tokenizer!);

        const outputTokenIds = await this.model!.generate!({
          pixel_values: pixelValues,
          ...generationConfig,
        });

        const generatedText = this.tokenizer!.decode(outputTokenIds[0], {
          skip_special_tokens: true,
        });
        candidates = [this.postprocess(generatedText)];
      } else {
        candidates = await beamSearch(this.model!, this.tokenizer!, pixelValues, numCandidates);
        candidates = candidates.map(c => this.postprocess(c));
      }

      return {
        latex: candidates[0] || '',
        candidates,
        debugImage
      };
    } finally {
      if (pixelValues) {
        pixelValues.dispose();
      }
      this.isInferring = false;
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
