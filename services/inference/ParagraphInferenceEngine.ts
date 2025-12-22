
import {
  InferenceResult,
  InferenceOptions,
  SamplingOptions,
  ParagraphInferenceResult,
  BBox
} from "./types";
import { InferenceEngine } from "./InferenceEngine";
import { bboxMerge, splitConflict, sortBoxes } from "./utils/boxUtils";
import { maskImg, sliceFromImage } from "./utils/imageUtils";
import { removeStyle, addNewlines } from "../../utils/latex";
import { InferenceSession, env } from "onnxruntime-web";
import { preprocessYolo } from "./utils/yoloPreprocess";
import { yoloPostprocess } from "./utils/yoloPostprocess";
import { downloadManager } from "../downloader/DownloadManager";
import { MODEL_CONFIG } from "./config";
import { preprocessDBNet } from "./utils/dbnetPreprocess";
import { dbnetPostprocess } from "./utils/dbnetPostprocess";

import { modelLoader } from "./ModelLoader";
import { preprocessTrOCR } from "./utils/trocrPreprocess";
import { getSessionOptions } from "./config";
import { recBatchPreprocess } from "./utils/recPreprocess";
import { recBatchPostprocess } from "./utils/recPostprocess";

// Custom filenames for Text Recognition models in the same repo
// const TEXT_ENC_NAME = "onnx/text_recognizer_encoder.onnx";
// const TEXT_DEC_NAME = "onnx/text_recognizer_decoder_with_past.onnx";
// Start with a hardcoded name in the repo, or add to config later.
// "detection.onnx" needs to exist in the HF repo.
// Start with a hardcoded name in the repo, or add to config later.
// "detection.onnx" needs to exist in the HF repo.


export class ParagraphInferenceEngine {
  private latexRecEngine: InferenceEngine;
  private latexDetSession: InferenceSession | null = null;
  private textDetSession: InferenceSession | null = null;
  private textRecSession: InferenceSession | null = null;

  // We need models for:
  // 1. Latex Detection (YOLO/ONNX)
  // 2. Text Detection (DBNet/ONNX)
  // 3. Text Recognition (CRNN/ONNX)

  // For now, we assume these are initialized or passed in. 
  // Since loading 4 models in browser is heavy, we might lazy load them.

  constructor(latexRecEngine: InferenceEngine) {
    this.latexRecEngine = latexRecEngine;
  }

  public async init(onProgress?: (status: string, progress?: number) => void) {
    if (this.latexDetSession) return; // Already init

    if (onProgress) onProgress("Initializing Paragraph Models...", 0);

    // Configure WASM paths to root (where vite-plugin-static-copy puts them)
    env.wasm.wasmPaths = "/";
    // Enable multi-threading for performance
    env.wasm.numThreads = 4;
    // Enable SIMD if possible for performance
    env.wasm.simd = true;

    // Standard Session Options for Stability & Performance
    const sessionOptions: InferenceSession.SessionOptions = {
      executionProviders: ['wasm'], // Use WASM, it's generally more stable for these models
      executionMode: 'sequential',
      intraOpNumThreads: 4,
      interOpNumThreads: 4,
      graphOptimizationLevel: 'all'
    };

    // Load Latex Detection Model (YOLO)
    try {
      const modelId = MODEL_CONFIG.LATEX_DET_ID;
      const fileUrl = `https://huggingface.co/${modelId}/resolve/main/${MODEL_CONFIG.LATEX_DET_MODEL}`;

      if (onProgress) onProgress("Downloading Detection Model...", 10);

      // Ensure cached
      await downloadManager.downloadFile(fileUrl, (p) => {
        if (onProgress) onProgress(`Downloading Detection Model...`, Math.round((p.loaded / p.total) * 100));
      });

      // Load from cache or fetch
      // Since downloadManager puts it in cache, we try to read it back.
      // Note: DownloadManager doesn't expose the response directly in a convenient way for resizing?
      // Actually standard Cache API:
      const cache = await caches.open('transformers-cache');
      const response = await cache.match(fileUrl);

      if (!response) {
        throw new Error("Failed to load model from cache after download");
      }

      const modelBlob = await response.blob();
      const modelBuffer = await modelBlob.arrayBuffer();

      if (onProgress) onProgress("Creating Inference Session...", 80);

      // Create Session
      // We assume webgpu is preferred if available?
      // For now, let's try 'webgpu' then 'wasm' fallback?
      // Or just 'wasm' for detection if it's lighter? YOLOv8s is ~20MB. WebGPU is better.
      try {
        // Try WebGPU first? The error log showed WebGPU failed detection earlier, so maybe just force WASM with safe options.
        // If we want to support WebGPU, we need to pass options there too.
        // For debugging, let's stick to the user's current fallback path which is WASM.
        // Actually, let's try ['webgpu', 'wasm'] but with the options.
        this.latexDetSession = await InferenceSession.create(new Uint8Array(modelBuffer), {
          ...sessionOptions,
          executionProviders: ['webgpu', 'wasm']
        });
      } catch (e) {
        console.warn("WebGPU failed for detection, falling back to wasm", e);
        this.latexDetSession = await InferenceSession.create(new Uint8Array(modelBuffer), {
          ...sessionOptions,
          executionProviders: ['wasm']
        });
      }

      if (onProgress) onProgress("Ready", 100);
    } catch (e) {
      console.error("Failed to init detection model", e);
    } // End Latex Detection Try

    // Load Text Detection Model (DBNet)
    if (!this.textDetSession) {
      try {
        const txtModelId = MODEL_CONFIG.TEXT_DETECTOR_ID;
        const txtFileUrl = `https://huggingface.co/${txtModelId}/resolve/main/${MODEL_CONFIG.TEXT_DET_MODEL}`;

        if (onProgress) onProgress("Downloading Text Detection Model...", 50);

        await downloadManager.downloadFile(txtFileUrl, (p) => {
          if (onProgress) onProgress(`Downloading Text Detection Model...`, Math.round((p.loaded / p.total) * 100));
        });

        const txtCache = await caches.open('transformers-cache');
        const txtResponse = await txtCache.match(txtFileUrl);

        if (txtResponse) {
          const txtBlob = await txtResponse.blob();
          const txtBuffer = await txtBlob.arrayBuffer();

          try {
            // Force WASM with disabled optimization for this specific model to avoid ceil() error
            this.textDetSession = await InferenceSession.create(new Uint8Array(txtBuffer), {
              ...sessionOptions,
              graphOptimizationLevel: 'disabled', // Vital for fixing 'ceil() in shape computation' error
              executionProviders: ['wasm']
            });
          } catch (e) {
            console.warn("Text Detection Init Failed", e);
            throw e;
          }
        } else {
          console.warn("Text detection model fetch failed or empty");
        }
      } catch (e) {
        console.error("Failed to init text detection model", e);
      }
    }

    // Load Text Recognition (CRNN/SVTR)
    if (!this.textRecSession) {
      try {
        const recModelId = MODEL_CONFIG.TEXT_RECOGNIZER_ID;
        // Construct URL for ONNX file
        // e.g. https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/english/rec.onnx
        const recFileUrl = `https://huggingface.co/${recModelId}/resolve/main/${MODEL_CONFIG.TEXT_REC_MODEL}`;

        if (onProgress) onProgress("Downloading Text Recognition Model...", 70);

        await downloadManager.downloadFile(recFileUrl, (p) => {
          if (onProgress) onProgress(`Downloading Text Recognition Model...`, Math.round((p.loaded / p.total) * 100));
        });

        const recCache = await caches.open('transformers-cache');
        const recResponse = await recCache.match(recFileUrl);

        if (recResponse) {
          const recBlob = await recResponse.blob();
          const recBuffer = await recBlob.arrayBuffer();

          try {
            this.textRecSession = await InferenceSession.create(new Uint8Array(recBuffer), {
              ...sessionOptions,
              executionProviders: ['webgpu', 'wasm']
            });
          } catch (e) {
            console.warn("WebGPU Text Rec failed, fallback to WASM", e);
            this.textRecSession = await InferenceSession.create(new Uint8Array(recBuffer), {
              ...sessionOptions,
              executionProviders: ['wasm']
            });
          }
        } else {
          console.warn("Text recognition model fetch failed or empty");
        }
      } catch (e) {
        console.error("Failed to init text recognition model", e);
      }
    }
  }

  public async inferParagraph(
    imageBlob: Blob,
    options: SamplingOptions,
    signal?: AbortSignal
  ): Promise<ParagraphInferenceResult> {
    // Stage 1: Initial Raw Image (Non-blocking)
    this.sendDebugImage(imageBlob, [], [], options);

    // 1. Latex Detection
    // Returns list of BBoxes for formulas
    const latexBBoxes = await this.detectLatex(imageBlob);

    // 2. Mask Image
    // Mask out the formulas to avoid text detector picking them up as text
    const maskedImageBlob = await maskImg(imageBlob, latexBBoxes);

    // 3. Text Detection
    // Returns list of BBoxes for text lines
    let textBBoxes = await this.detectText(maskedImageBlob);

    // 4. Merge/Refine BBoxes
    // "ocr_bboxes = sorted(ocr_bboxes); ocr_bboxes = bbox_merge(ocr_bboxes)"
    // "ocr_bboxes = split_conflict(ocr_bboxes, latex_bboxes)"
    textBBoxes = sortBoxes(textBBoxes);
    textBBoxes = bboxMerge(textBBoxes);
    textBBoxes = splitConflict(textBBoxes, latexBBoxes);

    // Filter out non-text (if splitConflict changed labels or we have garbage)
    textBBoxes = textBBoxes.filter(b => b.label === "text");

    // Stage 2: Detection Result (Boxes only, Non-blocking)
    this.sendDebugImage(imageBlob, textBBoxes, latexBBoxes, options, false);

    // 5. Slice Images
    const textSlices = await sliceFromImage(imageBlob, textBBoxes);
    const latexSlices = await sliceFromImage(imageBlob, latexBBoxes);

    // 6. Recognize Text
    // Run Text Rec Model on each slice
    const textContents = await this.recognizeText(textSlices);
    textBBoxes.forEach((b, i) => b.content = textContents[i]);

    // 7. Recognize Latex
    // Run Latex Rec Model (Formula Rec) on each slice SEQUENTIALLY to avoid session concurrency issues
    const latexResults: InferenceResult[] = [];
    // Clean options for sub-inference to avoid callback recursion/flicker
    const subOptions = { ...options, onPreprocess: undefined };
    for (const slice of latexSlices) {
      latexResults.push(await this.latexRecEngine.infer(slice, subOptions, signal));
    }
    latexResults.forEach((res, i) => {
      latexBBoxes[i].content = res.latex;
    });

    // 8. Combine & Format
    const resultMarkdown = this.combineResults(textBBoxes, latexBBoxes);

    // Stage 3: Final Result (Boxes + Labels, Non-blocking)
    this.sendDebugImage(imageBlob, textBBoxes, latexBBoxes, options, true);

    return {
      markdown: resultMarkdown
    };
  }

  private async sendDebugImage(
    imageBlob: Blob,
    textBBoxes: BBox[],
    latexBBoxes: BBox[],
    options: SamplingOptions,
    showLabels: boolean = true
  ) {
    if (!options.onPreprocess) return;

    try {
      const bitmap = await createImageBitmap(imageBlob);
      const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(bitmap, 0, 0);

        // Draw Text BBoxes (Cyan)
        ctx.strokeStyle = '#00ffff';
        ctx.lineWidth = 3;
        ctx.setLineDash([5, 5]);
        for (const box of textBBoxes) {
          ctx.strokeRect(box.x, box.y, box.w, box.h);
        }

        // Draw Latex BBoxes (Yellow/Gold)
        ctx.strokeStyle = '#ffd700';
        ctx.lineWidth = 4;
        ctx.setLineDash([]);
        for (const box of latexBBoxes) {
          ctx.strokeRect(box.x, box.y, box.w, box.h);
        }

        // Draw content labels for debugging
        if (showLabels) {
          ctx.font = 'bold 16px sans-serif';
          ctx.fillStyle = '#00ffff';
          for (const box of textBBoxes) {
            if (box.content) {
              const label = box.content.substring(0, 15) + (box.content.length > 15 ? '...' : '');
              ctx.fillText(label, box.x, box.y - 5);
            }
          }
        }

        const debugBlob = await canvas.convertToBlob();
        const reader = new FileReader();
        const debugDataUrl = await new Promise<string>((resolve) => {
          reader.onloadend = () => resolve(reader.result as string);
          reader.readAsDataURL(debugBlob);
        });
        options.onPreprocess(debugDataUrl);
      }
    } catch (e) {
      console.error("Failed to generate debug image", e);
    }
  }

  private async detectLatex(image: Blob): Promise<BBox[]> {
    if (!this.latexDetSession) {
      console.warn("Latex Detection session not ready, using mock.");
      return [];
    }

    try {
      const { tensor, inputWidth, inputHeight, originalWidth, originalHeight } = await preprocessYolo(image);

      const feeds: Record<string, import("onnxruntime-web").Tensor> = {};
      // YOLOv8 input name is usually 'images'. Check model metadata if possible, or assume standard.
      // If unknown, we can use session.inputNames[0]
      const inputName = this.latexDetSession.inputNames[0];
      feeds[inputName] = tensor;

      const results = await this.latexDetSession.run(feeds);

      const outputName = this.latexDetSession.outputNames[0];
      const output = results[outputName];

      // Output shape: [1, 5+C, 8400]
      return yoloPostprocess(
        output.data as Float32Array,
        output.dims as number[],
        0.25, // Conf Threshold
        originalWidth,
        originalHeight,
        inputWidth,
        inputHeight
      );

    } catch (e) {
      console.error("Detection Failed", e);
      return [];
    }
  }

  private async detectText(image: Blob): Promise<BBox[]> {
    if (!this.textDetSession) {
      console.warn("Text Detection session not ready, assuming whole image.");
      // Fallback or empty
      return [];
    }

    try {
      const { tensor, inputWidth, inputHeight, originalWidth, originalHeight } = await preprocessDBNet(image);

      const feeds: Record<string, import("onnxruntime-web").Tensor> = {};
      const inputName = this.textDetSession.inputNames[0];
      feeds[inputName] = tensor;

      const results = await this.textDetSession.run(feeds);
      const outputName = this.textDetSession.outputNames[0];
      const output = results[outputName];

      // Output dims: [1, 1, H, W] or [1, H, W] depending on model export
      // DBNet usually outputs probability map

      return dbnetPostprocess(
        output.data as Float32Array,
        output.dims[output.dims.length - 1], // Width
        output.dims[output.dims.length - 2], // Height
        0.3,
        originalWidth,
        originalHeight,
        inputWidth,
        inputHeight
      );
    } catch (e) {
      console.error("Text Detection Failed", e);
      return [];
    }
  }

  private async recognizeText(images: Blob[]): Promise<string[]> {
    if (images.length === 0) return [];
    if (!this.textRecSession) {
      return images.map(() => "Rec Model Not Loaded");
    }

    try {
      // BATCHED Inference for major speed boost and stability
      const tensor = await recBatchPreprocess(images);

      const feeds: Record<string, import("onnxruntime-web").Tensor> = {};
      const inputName = this.textRecSession!.inputNames[0];
      feeds[inputName] = tensor;

      const sessRes = await this.textRecSession!.run(feeds);
      const outputName = this.textRecSession!.outputNames[0];
      const output = sessRes[outputName];

      return recBatchPostprocess(output.data as Float32Array, output.dims as number[]);
    } catch (e) {
      console.error(`Batched Text Rec Error:`, e);
      return images.map(() => "[Error]");
    }
  }

  private combineResults(textBBoxes: BBox[], latexBBoxes: BBox[]): string {
    // Logic from paragraph2md
    // 1. Format Latex content (add $ signs)
    latexBBoxes.forEach(b => {
      // Heuristic: if label is "embedding" (inline) -> $...$
      // if "isolated" -> $$...$$
      // managing this distinction requires the detector to provide labels.
      // Default to isolated $$ for safety if unknown? 
      // TexTeller source: "embedding" vs "isolated".
      // We'll assume isolated for now unless detected.
      const content = b.content || "";
      b.content = ` $${content}$ `; // Simplify to inline for now or add logic
    });

    const allBoxes = [...textBBoxes, ...latexBBoxes];
    const sortedBoxes = sortBoxes(allBoxes);

    if (sortedBoxes.length === 0) return "";

    let md = "";
    let prev: BBox = { x: -1, y: -1, w: -1, h: -1, label: "guard" };

    for (const curr of sortedBoxes) {
      // Logic for adding spaces / newlines
      if (!this.isSameRow(prev, curr)) {
        md += "\n"; // New line
      } else {
        md += " ";
      }
      md += curr.content || "";
      prev = curr;
    }

    return md.trim();
  }

  private isSameRow(a: BBox, b: BBox, tolerance: number = 10): boolean {
    if (a.y === -1) return false; // Guard
    return Math.abs(a.y - b.y) < tolerance;
  }
}
