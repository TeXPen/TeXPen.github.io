import {
  ParagraphInferenceResult
} from "./types";
import { InferenceEngine } from "./InferenceEngine";
import { VLMInferenceEngine } from "./VLMInferenceEngine";

export class ParagraphInferenceEngine {
  private vlmEngine: VLMInferenceEngine;

  constructor(private latexRecEngine: InferenceEngine) {
    // We don't use latexRecEngine strictly anymore if VLM handles everything,
    // but we keep the signature for compatibility or fallback.
    this.vlmEngine = new VLMInferenceEngine();
  }

  public async init(onProgress?: (status: string, progress?: number) => void) {
    await this.vlmEngine.init(onProgress);
  }

  public async inferParagraph(
    imageBlob: Blob,
    options?: any,
    signal?: AbortSignal
  ): Promise<ParagraphInferenceResult> {

    // Delegate to VLM Engine
    const vlmResult = await this.vlmEngine.inferVLM(imageBlob);

    return {
      markdown: vlmResult.markdown,
      debugImage: undefined // VLM engine doesn't currently return a debug image for this flow
    };
  }
}
