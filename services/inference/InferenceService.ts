import { AutoTokenizer, AutoModelForVision2Seq, PreTrainedModel, PreTrainedTokenizer, Tensor } from '@huggingface/transformers';
import { removeStyle, addNewlines } from '../latexUtils';
import { preprocess } from './imagePreprocessing';
import { beamSearch } from './beamSearch';

// Constants
const MODEL_ID = 'onnx-community/TexTeller3-ONNX';

export class InferenceService {
  private model: PreTrainedModel | null = null;
  private tokenizer: PreTrainedTokenizer | null = null;
  private static instance: InferenceService;

  private constructor() { }

  public static getInstance(): InferenceService {
    if (!InferenceService.instance) {
      InferenceService.instance = new InferenceService();
    }
    return InferenceService.instance;
  }

  public async init(onProgress?: (status: string) => void, options: { device?: string; dtype?: string } = {}): Promise<void> {
    if (this.model && this.tokenizer) return;

    try {
      if (onProgress) onProgress('Loading tokenizer...');
      this.tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);

      if (onProgress) onProgress('Loading model... (this may take a while)');
      // Force browser cache usage and allow remote models
      this.model = await AutoModelForVision2Seq.from_pretrained(MODEL_ID, {
        device: options.device || 'webgpu', // Try WebGPU first, fallback to wasm automatically
        dtype: options.dtype || 'fp32',    // Use fp32 for unquantized model as requested
      } as any);

      if (onProgress) onProgress('Ready');
    } catch (error) {
      console.error('Failed to load model:', error);
      throw error;
    }
  }

  public async infer(imageBlob: Blob, numCandidates: number = 5): Promise<{ latex: string; candidates: string[]; debugImage: string }> {
    if (!this.model || !this.tokenizer) {
      await this.init();
    }

    // 1. Preprocess
    const { tensor: pixelValues, debugImage } = await preprocess(imageBlob);

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
  }

  private postprocess(latex: string): string {
    // 1. Remove style (bold, italic, etc.) - optional but recommended for cleaner output
    let processed = removeStyle(latex);

    // 2. Add newlines for readability
    processed = addNewlines(processed);

    return processed;
  }

  public async dispose(): Promise<void> {
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
