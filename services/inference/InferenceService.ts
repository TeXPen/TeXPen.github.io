import { AutoTokenizer, AutoModelForVision2Seq, PreTrainedModel, PreTrainedTokenizer, Tensor } from '@huggingface/transformers';
import { removeStyle, addNewlines } from '../latexUtils';
import { preprocess } from './imagePreprocessing';
import { beamSearch } from './beamSearch';
import { isWebGPUAvailable } from '../../utils/env';

// Constants
const MODEL_ID = 'onnx-community/TexTeller3-ONNX';

export class InferenceService {
  private model: PreTrainedModel | null = null;
  private tokenizer: PreTrainedTokenizer | null = null;
  private static instance: InferenceService;
  private isInferring: boolean = false;

  private constructor() { }

  public static getInstance(): InferenceService {
    if (!InferenceService.instance) {
      InferenceService.instance = new InferenceService();
    }
    return InferenceService.instance;
  }

  public async init(onProgress?: (status: string) => void, options: { dtype?: string, device?: 'webgpu' | 'wasm' | 'webgl' } = {}): Promise<void> {
    if (this.model && this.tokenizer) {
      // If the model is already loaded, but the quantization or device is different, we need to dispose and reload.
      if ((options.dtype && (this.model as any).config.dtype !== options.dtype) ||
        (options.device && (this.model as any).config.device !== options.device)) {
        if (this.isInferring) {
          throw new Error("Cannot change model settings while an inference is in progress.");
        }
        await this.dispose();
      } else {
        return;
      }
    }

    try {
      if (onProgress) onProgress('Loading tokenizer...');
      this.tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);

      const webgpuAvailable = await isWebGPUAvailable();
      const device = options.device || (webgpuAvailable ? 'webgpu' : 'wasm');
      const dtype = options.dtype || (webgpuAvailable ? 'fp16' : 'q8');

      if (onProgress) onProgress(`Loading model with ${device} (${dtype})... (this may take a while)`);

      // Define file mappings based on dtype
      // Note: We assume 'encoder_model.onnx' is used for fp16 if a specific fp16 encoder doesn't exist.
      // The user provided links for:
      // - decoder_with_past_model_fp16.onnx
      // - encoder_model_int8.onnx
      // - decoder_with_past_model_int8.onnx

      let sessionOptions: any = {
        device: device,
        dtype: dtype,
      };

      if (dtype === 'fp16') {
        sessionOptions = {
          ...sessionOptions,
          // For fp16, we use the default encoder (usually fp32 or mixed) if fp16 encoder isn't available,
          // and the specific fp16 decoder.
          encoder_model_file_name: 'encoder_model.onnx',
          decoder_model_file_name: 'decoder_with_past_model_fp16.onnx',
        };
      } else if (dtype === 'q8') {
        sessionOptions = {
          ...sessionOptions,
          encoder_model_file_name: 'encoder_model_int8.onnx',
          decoder_model_file_name: 'decoder_with_past_model_int8.onnx',
        };
      }
      // For fp32 (default), we rely on standard naming or explicit defaults if needed.

      this.model = await AutoModelForVision2Seq.from_pretrained(MODEL_ID, sessionOptions);

      if (onProgress) onProgress('Ready');
    } catch (error) {
      console.error('Failed to load model:', error);
      throw error;
    }
  }

  public async infer(imageBlob: Blob, numCandidates: number = 1): Promise<{ latex: string; candidates: string[]; debugImage: string }> {
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
        const outputTokenIds = await this.model!.generate({
          pixel_values: pixelValues,
          max_new_tokens: 1024,
          do_sample: false,
          pad_token_id: this.tokenizer!.pad_token_id,
          eos_token_id: this.tokenizer!.eos_token_id,
          bos_token_id: this.tokenizer!.bos_token_id,
          decoder_start_token_id: this.tokenizer!.bos_token_id,
        } as any);

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

  public async dispose(): Promise<void> {
    if (this.isInferring) {
      throw new Error("Cannot dispose model while an inference is in progress.");
    }
    if (this.model) {
      if ('dispose' in this.model && typeof (this.model as any).dispose === 'function') {
        await (this.model as any).dispose();
      }
      this.model = null;
    }
    this.tokenizer = null;
    (InferenceService as any).instance = null;
  }
}

export const inferenceService = InferenceService.getInstance();
