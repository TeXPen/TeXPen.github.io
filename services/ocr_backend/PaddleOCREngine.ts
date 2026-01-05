import * as ort from 'onnxruntime-web';
import {
  resizeForDetection,
  preprocessImage,
  boxesFromBitmap,
  ctcDecode,
  sortBoxes,
  OcrResult,
  BoundingBox
} from './PaddleUtils';
import { PADDLE_MODEL_CONFIG, DEFAULT_REC_KEYS } from './ModelConfig';

export interface ScanResult {
  layout: LayoutItem[];
  rawText: string;
}

export interface LayoutItem {
  type: string;
  box: [number, number, number, number]; // x, y, w, h
  text?: string;
  subItems?: OcrResult[]; // If it's a Text region, contains text lines
  confidence: number;
}

// Configure ONNX Runtime WASM paths immediately on import
// This avoids race conditions where internal initialization might rely on defaults.
if (typeof self !== 'undefined') {
  console.log(`[PaddleOCREngine] Config - crossOriginIsolated: ${self.crossOriginIsolated}`);

  // Point to the files served by vite-plugin-static-copy at root (or public folder)
  ort.env.wasm.wasmPaths = {
    'ort-wasm-simd-threaded.wasm': '/ort-wasm-simd-threaded.wasm',
    'ort-wasm-simd-threaded.jsep.wasm': '/ort-wasm-simd-threaded.jsep.wasm',
    'ort-wasm-simd-threaded.asyncify.wasm': '/ort-wasm-simd-threaded.asyncify.wasm',
    'ort-wasm-simd.wasm': '/ort-wasm-simd-threaded.wasm',
    'ort-wasm.wasm': '/ort-wasm.wasm',
    'ort-wasm.jsep.wasm': '/ort-wasm-simd-threaded.jsep.wasm', // Fallback/Alias if needed
  } as any;
}

export class PaddleOCREngine {
  private layoutSession: ort.InferenceSession | null = null;
  private detSession: ort.InferenceSession | null = null;
  private recSession: ort.InferenceSession | null = null;

  // Config
  private recKeys: string[] = []; // Chars

  public async init(): Promise<void> {
    // 1. Load Sessions
    const options: ort.InferenceSession.SessionOptions = {
      executionProviders: ['wasm'], // Fallback to wasm if webgpu fails or for stability
      graphOptimizationLevel: 'all'
    };

    try {
      console.log('Loading PaddleOCR Layout Model...');
      this.layoutSession = await ort.InferenceSession.create(PADDLE_MODEL_CONFIG.LAYOUT.MODEL_PATH, options);

      console.log('Loading PaddleOCR Det Model...');
      this.detSession = await ort.InferenceSession.create(PADDLE_MODEL_CONFIG.DET.MODEL_PATH, options);

      console.log('Loading PaddleOCR Rec Model...');
      this.recSession = await ort.InferenceSession.create(PADDLE_MODEL_CONFIG.REC.MODEL_PATH, options);

      // 2. Load Keys
      try {
        const response = await fetch(PADDLE_MODEL_CONFIG.REC.KEY_FILE);
        if (response.ok) {
          const text = await response.text();
          this.recKeys = text.split('\n');
          // ppocr_keys_v1.txt usually separates by newline.
          // We need to ensure we handle the last blank char correctly if needed
          // PaddleOCR convention: 
          // The dictionary file contains mapping: index -> char (0-based or 1-based?)
          // Actually index 0 is first char in file.
          // The model output has N+1 classes. 
          // Last class is 'blank' for CTC.

          // We just load the list. ctcDecode handles checks.
        } else {
          console.warn("Failed to load keys file, using defaults");
          this.recKeys = DEFAULT_REC_KEYS.split('');
        }
      } catch (e) {
        console.error("Error loading keys:", e);
        this.recKeys = DEFAULT_REC_KEYS.split('');
      }

    } catch (e) {
      console.error('Failed to init PaddleOCR:', e);
      throw e;
    }
  }

  public async process(imageBlob: Blob): Promise<ScanResult> {
    if (!this.layoutSession || !this.detSession || !this.recSession) {
      await this.init();
    }

    const imgBitmap = await createImageBitmap(imageBlob);

    // 1. Layout Analysis
    const layoutItems = await this.inferLayout(imgBitmap);

    // 2. Process Regions
    // Sort regions top-down
    layoutItems.sort((a, b) => a.box[1] - b.box[1]);

    const fullTextParts: string[] = [];

    for (const item of layoutItems) {
      const normalizedType = this.normalizeLayoutLabel(item.type);
      // Cut region
      const regionCanvas = new OffscreenCanvas(item.box[2], item.box[3]);
      const ctx = regionCanvas.getContext('2d') as OffscreenCanvasRenderingContext2D;
      ctx.drawImage(imgBitmap, item.box[0], item.box[1], item.box[2], item.box[3], 0, 0, item.box[2], item.box[3]);

      if (this.isTextRegion(normalizedType)) {
        // Run Text Det + Rec on this region
        const ocrResults = await this.detectAndRecognize(regionCanvas);
        item.subItems = ocrResults;

        const regionText = ocrResults.map(r => r.text).join(' ');
        item.text = regionText;
        fullTextParts.push(regionText);
      } else if (normalizedType === 'Equation') {
        // Here we would ideally call TexTeller
        // For now, placeholder or just keep as [Equation]
        item.text = "[Equation]";
        fullTextParts.push("[Equation]");
      } else {
        // Table / Figure
        item.text = `[${normalizedType}]`;
        fullTextParts.push(item.text);
      }
    }

    return {
      layout: layoutItems,
      rawText: fullTextParts.join('\n\n')
    };
  }

  /**
   * Run Layout Analysis (PP-Structure)
   * Model input: [1, 3, 800, 608] (Example, dynamic shape supported? usually fixed in ONNX export)
   * PP-Structurev2 typically uses YOLO/PicoDet style output.
   * 
   * NOTE: This implementation assumes the ONNX model handles the post-processing 
   * OR provides standard detection outputs (boxes, scores, classes).
   * 
   * If using RT-DETR / PicoDet, output processing is complex.
   * 
   * SIMPLIFICATION:
   * Assuming we use a simpler model or mock implementation for this step if layout.onnx is complex.
   * Let's assume standard object detection output: [1, N, 6] (x1, y1, x2, y2, score, class)
   */
  private async inferLayout(image: ImageBitmap): Promise<LayoutItem[]> {
    if (!this.layoutSession) throw new Error("Layout session not init");

    // 1. Resize/Preprocess
    // Layout models (Picodet) usually expect 32-multiple size or 640/800.
    // Let's use 800 (standard for some PP-Structure models) or 640.
    // We reuse resizeForDetection logic but with different limit?
    const { canvas, ratio } = await resizeForDetection(image, 800);
    const w = canvas.width;
    const h = canvas.height;

    // Preprocess (Mean/Std same as Det?) 
    // Picodet often uses: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale=1/255
    const ctx = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D;
    const floatData = preprocessImage(ctx, w, h);
    const tensor = new ort.Tensor('float32', floatData, [1, 3, h, w]);

    // 2. Run Inference
    const feeds = { [this.layoutSession.inputNames[0]]: tensor };
    const results = await this.layoutSession.run(feeds);

    // 3. Post-process
    // Output usually: "multiclass_nms3_0.tmp_0" (or similar) -> Shape [N, 6]
    // [class_id, score, x1, y1, x2, y2]

    // Find the output tensor (usually first one)
    const outputKey = this.layoutSession.outputNames[0];
    const outputNode = results[outputKey];

    if (!outputNode) return [{ type: 'Text', box: [0, 0, image.width, image.height], confidence: 1.0 }];

    const data = outputNode.data as Float32Array;
    // Shape is usually dynamic dim at 0? 
    // but flattened data is just a sequence.
    // We iterate by 6 stride.

    const items: LayoutItem[] = [];
    const numBoxes = data.length / 6;

    // Layout Labels for PicoDet-S_layout_17cls (layout.onnx)
    // Order matters. See ModelConfig for class list.
    const LABELS = PADDLE_MODEL_CONFIG.LAYOUT.LABELS;

    // Scale back to original image
    // 'ratio' from resizeForDetection is roughly target/source.
    // So we divide by ratio.
    // Note: resizing maintained aspect ratio.

    for (let i = 0; i < numBoxes; i++) {
      const offset = i * 6;
      const cls = data[offset];
      const score = data[offset + 1];
      const x1 = data[offset + 2];
      const y1 = data[offset + 3];
      const x2 = data[offset + 4];
      const y2 = data[offset + 5];

      if (score < 0.5) continue; // Threshold
      if (x2 - x1 < 5 || y2 - y1 < 5) continue; // Too small

      const label = LABELS[Math.round(cls)] || 'Text';

      // Map back to original
      const origX = Math.max(0, x1 / ratio);
      const origY = Math.max(0, y1 / ratio);
      const origW = Math.min((x2 - x1) / ratio, image.width - origX);
      const origH = Math.min((y2 - y1) / ratio, image.height - origY);

      items.push({
        type: label,
        box: [origX, origY, origW, origH],
        confidence: score
      });
    }

    // Fallback if no regions found
    if (items.length === 0) {
      items.push({
        type: 'Text',
        box: [0, 0, image.width, image.height],
        confidence: 1.0
      });
    }

    return items;
  }

  /**
   * Run Text Detection + Recognition on a canvas (region or full image)
   */
  public async detectAndRecognize(image: OffscreenCanvas): Promise<OcrResult[]> {
    // 1. Detection
    const { boxes } = await this.inferDet(image);
    const sortedBoxes = sortBoxes(boxes);

    // 2. Recognition Loop
    const results: OcrResult[] = [];

    for (const box of sortedBoxes) {
      // Crop box from image
      const crop = this.cropPolygon(image, box.points);

      // Recognize
      const { text, confidence } = await this.inferRec(crop);

      if (confidence > 0.5) {
        results.push({
          box,
          text,
          confidence
        });
      }
    }

    return results;
  }

  private async inferDet(image: OffscreenCanvas): Promise<{ boxes: BoundingBox[] }> {
    // 1. Resize
    const { canvas: resizedCanvas } = await resizeForDetection(image, PADDLE_MODEL_CONFIG.DET.LIMIT_SIDE_LEN);
    const w = resizedCanvas.width;
    const h = resizedCanvas.height;

    // 2. Preprocess
    // Cast to OffscreenCanvasRenderingContext2D because resizeForDetection guarantees OffscreenCanvas in our impl
    const ctx = resizedCanvas.getContext('2d') as OffscreenCanvasRenderingContext2D;
    const floatData = preprocessImage(ctx, w, h);

    const tensor = new ort.Tensor('float32', floatData, [1, 3, h, w]);

    // 3. Run
    const feeds = { [this.detSession!.inputNames[0]]: tensor };
    const results = await this.detSession!.run(feeds);

    const output = results[this.detSession!.outputNames[0]]; // usually 'sigmoid_0.tmp_0'
    // Shape: [1, 1, h, w]

    // 4. Post-process
    const boxes = boxesFromBitmap(
      output.data as Float32Array,
      output.dims[2], // h
      output.dims[3], // w
      image.width,    // original w
      image.height    // original h
    );

    return { boxes };
  }

  private async inferRec(image: OffscreenCanvas): Promise<{ text: string, confidence: number }> {
    // 1. Resize to (3, 48, 320) - maintain aspect ratio, pad
    // Standard PP-OCR: H=48, W=320 (or dynamic W, but ONNX might be fixed)

    const H = PADDLE_MODEL_CONFIG.REC.IMG_H; // 48
    const W = PADDLE_MODEL_CONFIG.REC.IMG_W; // 320

    // Resize logic: scale to H=48, w=scaled. If w > 320, resize to 320?
    // Actually usually we just resize to fixed H, preserve ratio, pad W to 320.

    const ratio = image.width / image.height;
    let newW = Math.round(H * ratio);
    if (newW > W) newW = W;

    const inputCanvas = new OffscreenCanvas(W, H);
    const ctx = inputCanvas.getContext('2d') as OffscreenCanvasRenderingContext2D;

    // Fill gray? or black/white?
    ctx.fillStyle = '#000000'; // Black padding
    ctx.fillRect(0, 0, W, H);

    // Draw image centered or left-aligned? Usually left-aligned 0,0
    ctx.drawImage(image, 0, 0, image.width, image.height, 0, 0, newW, H);

    // Preprocess
    // Rec model expects Values in [-0.5, 0.5] usually, or [0, 1] normalized with mean=0.5, std=0.5
    // Standard PP-OCR: mean=0.5, std=0.5
    const mean = [0.5, 0.5, 0.5];
    const std = [0.5, 0.5, 0.5];
    const floatData = preprocessImage(ctx, W, H, mean, std);

    const tensor = new ort.Tensor('float32', floatData, [1, 3, H, W]);

    // Run
    const feeds = { [this.recSession!.inputNames[0]]: tensor };
    const results = await this.recSession!.run(feeds);
    const output = results[this.recSession!.outputNames[0]]; // 'softmax_0.tmp_0'

    // Shape: [1, 40, 97] usually (Batch, Time, Classes)
    const dims = output.dims; // [Batch, Time, Vocab]
    const vocabSize = dims[dims.length - 1];

    return ctcDecode(output.data as Float32Array, this.recKeys, vocabSize);
  }

  private cropPolygon(image: OffscreenCanvas, points: [number, number][]): OffscreenCanvas {
    // Basic crop (bounding rect) + warp perspective if needed
    // For simple rects, just crop. 
    // For general quadrilaterals, should use warpPerspective (OpenCV).

    // Calculate bounding rect of points
    const xs = points.map(p => p[0]);
    const ys = points.map(p => p[1]);
    const minX = Math.min(...xs);
    const minY = Math.min(...ys);
    const maxX = Math.max(...xs);
    const maxY = Math.max(...ys);
    const w = maxX - minX;
    const h = maxY - minY;

    if (w <= 0 || h <= 0) return new OffscreenCanvas(1, 1);

    const crop = new OffscreenCanvas(w, h);
    const ctx = crop.getContext('2d') as OffscreenCanvasRenderingContext2D;

    // Only draw the part
    ctx.drawImage(image, minX, minY, w, h, 0, 0, w, h);

    return crop;
  }

  private normalizeLayoutLabel(label: string): string {
    const key = label.toLowerCase().replace(/\s+/g, '_');
    switch (key) {
      case 'paragraph_title':
      case 'document_title':
        return 'Title';
      case 'text':
      case 'content':
      case 'abstract':
      case 'number':
      case 'figure_caption':
      case 'table_caption':
      case 'algorithm':
        return 'Text';
      case 'header':
        return 'Header';
      case 'footer':
      case 'footnote':
        return 'Footer';
      case 'references':
      case 'reference':
        return 'Reference';
      case 'formula':
        return 'Equation';
      case 'table':
        return 'Table';
      case 'image':
      case 'seal':
        return 'Figure';
      default:
        return label;
    }
  }

  private isTextRegion(normalizedType: string): boolean {
    return ['Text', 'Title', 'Header', 'Footer', 'Reference'].includes(normalizedType);
  }
}
