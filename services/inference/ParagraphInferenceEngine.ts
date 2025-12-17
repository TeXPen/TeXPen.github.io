
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
import { removeStyle, addNewlines } from "../../utils/latex"; // Assuming these exist or will use existing ones

// Placeholder for external ONNX models if we don't have full wrappers yet
// We might need to extend InferenceEngine or create new ones for Det/Rec.
// For this port, we will define the structure and allow injection of model runners.

export class ParagraphInferenceEngine {
  private latexRecEngine: InferenceEngine;

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
    // Init all models
    if (onProgress) onProgress("Initializing Paragraph Models...", 0);
    // TODO: Load Detection Models
  }

  public async inferParagraph(
    imageBlob: Blob,
    options: SamplingOptions,
    signal?: AbortSignal
  ): Promise<ParagraphInferenceResult> {

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

    // 5. Slice Images
    const textSlices = await sliceFromImage(imageBlob, textBBoxes);
    const latexSlices = await sliceFromImage(imageBlob, latexBBoxes);

    // 6. Recognize Text
    // Run Text Rec Model on each slice
    const textContents = await this.recognizeText(textSlices);
    textBBoxes.forEach((b, i) => b.content = textContents[i]);

    // 7. Recognize Latex
    // Run Latex Rec Model (Formula Rec) on each slice
    // We can use the existing 'infer' method of InferenceEngine, but we need batching or sequential
    const latexContents = [];
    for (const slice of latexSlices) {
      const res = await this.latexRecEngine.infer(slice, options, signal);
      latexContents.push(res.latex);
    }
    latexBBoxes.forEach((b, i) => b.content = latexContents[i]);

    // 8. Combine & Format
    const resultMarkdown = this.combineResults(textBBoxes, latexBBoxes);

    return {
      markdown: resultMarkdown
    };
  }

  private async detectLatex(image: Blob): Promise<BBox[]> {
    // TODO: Implement ONNX run for Latex Detection
    // Return mock for now or throw "Not Implemented"
    console.warn("Latex Detection not implemented, returning empty");
    return [];
  }

  private async detectText(image: Blob): Promise<BBox[]> {
    // TODO: Implement ONNX run for Text Detection (DBNet)
    // Return mock for now
    console.warn("Text Detection not implemented, assuming whole image is text if no latex?");
    // If we really fallback, we might just return one box for the whole image? 
    // But for paragraph mode, we expect structure.
    return [{
      x: 0,
      y: 0,
      w: 100, // Dummy
      h: 100,
      label: "text"
    }];
  }

  private async recognizeText(images: Blob[]): Promise<string[]> {
    // TODO: Implement ONNX run for Text Recognition
    return images.map(() => "Detected Text Mock");
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
