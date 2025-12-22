
import {
  InferenceResult,
  ParagraphInferenceResult
} from "./types";
import { InferenceEngine } from "./InferenceEngine";
import { InferenceSession, Tensor, env } from "onnxruntime-web";
import { MODEL_CONFIG } from "./config";
import { downloadManager } from "../downloader/DownloadManager";
import { preprocessPaddleVL } from "./utils/paddleVLPreprocess";
import { paddleVLPostprocess } from "./utils/paddleVLPostprocess";
import { AutoTokenizer, PreTrainedTokenizer } from "@huggingface/transformers";

// Helper for argmax
function argmax(data: Float32Array): number {
  let maxVal = -Infinity;
  let maxIdx = -1;
  for (let i = 0; i < data.length; i++) {
    if (data[i] > maxVal) {
      maxVal = data[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

export class ParagraphInferenceEngine {
  private encoderSession: InferenceSession | null = null;
  private decoderSession: InferenceSession | null = null;
  private tokenizer: PreTrainedTokenizer | null = null;

  constructor(private latexRecEngine: InferenceEngine) {
    // We don't use latexRecEngine strictly anymore if VLM handles everything,
    // but we keep the signature for compatibility or fallback.
  }

  public async init(onProgress?: (status: string, progress?: number) => void) {
    if (this.encoderSession && this.tokenizer) return;

    if (onProgress) onProgress("Initializing PaddleOCR-VL...", 0);

    // 1. Load Tokenizer
    try {
      this.tokenizer = await AutoTokenizer.from_pretrained(MODEL_CONFIG.PADDLE_VL_ID);
    } catch (e) {
      console.warn("Failed to load tokenizer from HF Hub, using fallback or caching might be needed", e);
      // If user converted locally, they might need to point to local path.
      // For now assume HF Hub works for tokenizer.json
    }

    // 2. Load Models (Encoder & Decoder)
    const sessionOptions: InferenceSession.SessionOptions = {
      executionProviders: ['wasm'], // WebGPU is preferred if available but WASM is safer fallback
      executionMode: 'sequential',
      graphOptimizationLevel: 'all'
    };

    try {
      // ENCODER
      if (onProgress) onProgress("Downloading Vision Encoder...", 20);
      const encoderUrl = `https://huggingface.co/${MODEL_CONFIG.PADDLE_VL_ONNX_REPO}/resolve/main/${MODEL_CONFIG.PADDLE_VL_ENCODER}`;

      await downloadManager.downloadFile(encoderUrl, (p) => {
        if (onProgress) onProgress(`Downloading Vision Encoder...`, Math.round((p.loaded / p.total) * 100));
      });

      const cache = await caches.open('transformers-cache');
      const encoderResp = await cache.match(encoderUrl);
      if (!encoderResp) throw new Error("Encoder not in cache");
      const encoderBuffer = await (await encoderResp.blob()).arrayBuffer();
      this.encoderSession = await InferenceSession.create(new Uint8Array(encoderBuffer), sessionOptions);

      // DECODER
      if (onProgress) onProgress("Downloading Text Decoder...", 50);
      const decoderUrl = `https://huggingface.co/${MODEL_CONFIG.PADDLE_VL_ONNX_REPO}/resolve/main/${MODEL_CONFIG.PADDLE_VL_DECODER}`;

      await downloadManager.downloadFile(decoderUrl, (p) => {
        if (onProgress) onProgress(`Downloading Text Decoder...`, Math.round((p.loaded / p.total) * 100));
      });

      const decoderResp = await cache.match(decoderUrl);
      if (!decoderResp) throw new Error("Decoder not in cache");
      const decoderBuffer = await (await decoderResp.blob()).arrayBuffer();
      this.decoderSession = await InferenceSession.create(new Uint8Array(decoderBuffer), sessionOptions);

    } catch (e) {
      console.error("Failed to load VLM ONNX models", e);
      throw new Error(`Model loading failed. Ensure models are uploaded to ${MODEL_CONFIG.PADDLE_VL_ONNX_REPO}`);
    }

    if (onProgress) onProgress("Ready", 100);
  }

  public async inferParagraph(
    imageBlob: Blob,
    options?: any, // SamplingOptions generic
    signal?: AbortSignal
  ): Promise<ParagraphInferenceResult> {
    if (!this.encoderSession || !this.decoderSession || !this.tokenizer) {
      throw new Error("Models not initialized");
    }

    // 1. Preprocess
    const pixelValues = await preprocessPaddleVL(imageBlob);

    // 2. Run Encoder
    // Input: pixel_values [1, 3, H, W]
    const encoderFeeds = {
      [this.encoderSession.inputNames[0]]: pixelValues
    };
    const encoderResults = await this.encoderSession.run(encoderFeeds);
    // Usually 'last_hidden_state'
    const encoderHiddenState = encoderResults[this.encoderSession.outputNames[0]];

    // 3. Decoding Loop (Greedy Search)
    const MAX_TOKENS = 512;
    // Start Token: <s> or similar. Check tokenizer or config.
    // ERNIE/Paddle usually 1/2.
    // We'll use tokenizer.bos_token_id if available, else 1.
    const bosToken = this.tokenizer.bos_token_id || 1;
    const eosToken = this.tokenizer.eos_token_id || 2;

    let currentToken = bosToken;
    const generatedIds: number[] = [currentToken];

    // KV Cache (Past Key Values) management is complex in raw ONNX.
    // If we use 'decoder_model_merged.onnx' (with past), we need to handle inputs.
    // For simplicity V1: Use 'decoder_model.onnx' (no past) or pass empty pasts?
    // Using `merged` usually allows both.
    // We will start with a basic loop without KV cache optimization if performance permits,
    // OR just pass generic inputs.

    // Simplest approach: Re-run decoder full sequence (slow but correct for naive impl)
    // input_ids: [1, t2, t3...]

    for (let i = 0; i < MAX_TOKENS; i++) {
      const inputIdsTensor = new Tensor("int64", new BigInt64Array(generatedIds.map(BigInt)), [1, generatedIds.length]);

      // Context: encoder_hidden_states
      // Decoder Inputs: input_ids, encoder_hidden_states
      // (plus attention_mask presumably)

      const feedback = {
        input_ids: inputIdsTensor,
        encoder_hidden_states: encoderHiddenState,
        // Add attention masks if required by model
      } as any;

      // Add dummy attention mask [1, seq_len]
      // const mask = new Tensor("int64", new BigInt64Array(generatedIds.length).fill(1n), [1, generatedIds.length]);
      // feedback['attention_mask'] = mask;

      const results = await this.decoderSession.run(feedback);
      const logits = results.logits || results[this.decoderSession.outputNames[0]];

      // Get last token logits: [1, seq_len, vocab_size] -> last row
      const vocabSize = logits.dims[2];
      const seqLen = logits.dims[1];
      // Pointer to last token's logits
      const startIdx = (seqLen - 1) * vocabSize;
      const lastLogits = logits.data.slice(startIdx, startIdx + vocabSize) as Float32Array;

      const nextId = argmax(lastLogits);

      if (nextId === eosToken) break;

      generatedIds.push(nextId);
      currentToken = nextId;
    }

    // 4. Decode text
    const text = this.tokenizer.decode(generatedIds, { skip_special_tokens: true });

    // 5. Postprocess
    const markdown = paddleVLPostprocess(text);

    return {
      markdown
    };
  }
}
