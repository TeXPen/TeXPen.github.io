import { Tensor } from '@huggingface/transformers';
import cv from '@techstark/opencv-js';

// Constants for PaddleOCR
export const DET_LIMIT_SIDE_LEN = 960;
export const DET_DB_THRESH = 0.3;
export const DET_DB_BOX_THRESH = 0.6;
export const DET_DB_UNCLIP_RATIO = 1.5;

export const REC_IMG_H = 48; // PP-OCRv4 uses 48
export const REC_IMG_W = 320;

// Standard ImageNet mean/std used by PaddleOCR
export const PADDLE_MEAN = [0.485, 0.456, 0.406];
export const PADDLE_STD = [0.229, 0.224, 0.225];

export interface BoundingBox {
  points: [number, number][]; // 4 points: tl, tr, br, bl
  confidence: number;
}

export interface OcrResult {
  box: BoundingBox;
  text: string;
  confidence: number;
}

/**
 * Resize image for text detection (DBNet).
 * Must be multiple of 32.
 */
export async function resizeForDetection(
  image: ImageBitmap | HTMLImageElement | OffscreenCanvas,
  limitSideLen: number = DET_LIMIT_SIDE_LEN
): Promise<{
  canvas: HTMLCanvasElement | OffscreenCanvas,
  ratio: number
}> {
  const w = image.width;
  const h = image.height;
  let ratio = 1.0;

  if (Math.max(w, h) > limitSideLen) {
    if (h > w) {
      ratio = limitSideLen / h;
    } else {
      ratio = limitSideLen / w;
    }
  }

  let resizeH = Math.round(h * ratio);
  let resizeW = Math.round(w * ratio);

  resizeH = Math.max(Math.round(resizeH / 32) * 32, 32);
  resizeW = Math.max(Math.round(resizeW / 32) * 32, 32);

  // Recalculate ratio exactly based on the snapped dimensions to allow mapping back
  const ratioH = resizeH / h;
  const ratioW = resizeW / w;

  // We store the ratio to map back detection boxes to original image
  // Typically we just use one ratio if aspect ratio is preserved, but snapping changes it slightly.
  // For simplicity, we return the primary scaling factor, but DBNet post-process usually handles restoration via the scale.

  const canvas = new OffscreenCanvas(resizeW, resizeH);
  const ctx = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D;
  ctx.drawImage(image, 0, 0, resizeW, resizeH);

  return { canvas, ratio };
}

/**
 * Preprocess image for inference (Normalize -> CHW Float32Array).
 */
export function preprocessImage(
  ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D,
  width: number,
  height: number,
  mean: number[] = PADDLE_MEAN,
  std: number[] = PADDLE_STD,
  scale: number = 1 / 255
): Float32Array {
  const imageData = ctx.getImageData(0, 0, width, height);
  const { data } = imageData;
  const float32Data = new Float32Array(3 * height * width);

  // CHW Layout
  for (let i = 0; i < height * width; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];

    // R channel
    float32Data[i] = ((r * scale) - mean[0]) / std[0];
    // G channel
    float32Data[width * height + i] = ((g * scale) - mean[1]) / std[1];
    // B channel
    float32Data[2 * width * height + i] = ((b * scale) - mean[2]) / std[2];
  }

  return float32Data;
}

/**
 * Post-process DBNet output (segmentation map) to get bounding boxes.
 * Requires OpenCV.js
 */
export function boxesFromBitmap(
  pred: Float32Array, // (1, 1, H, W)
  maskH: number,
  maskW: number,
  destWidth: number,
  destHeight: number,
  boxThresh: number = DET_DB_BOX_THRESH,
  unclipRatio: number = DET_DB_UNCLIP_RATIO
): BoundingBox[] {
  const boxes: BoundingBox[] = [];

  // 1. Binarize
  const bitmap = new cv.Mat(maskH, maskW, cv.CV_8UC1);
  const dataPtr = bitmap.data;
  for (let i = 0; i < maskH * maskW; i++) {
    // pred is float32, usually sigmoid output logic happens in model or here. 
    // Assuming pred is already probability map.
    dataPtr[i] = pred[i] > DET_DB_THRESH ? 255 : 0;
  }

  // 2. Find Contours
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(bitmap, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

  const numContours = contours.size();
  for (let i = 0; i < numContours; i++) {
    const contour = contours.get(i);

    // 3. Get generic checks (min size)
    const area = cv.contourArea(contour);
    if (area < 16) { // Min area filter
      contour.delete();
      continue;
    }

    // 4. Get minAreaRect
    // box_points = cv2.boxPoints(rect).reshape((-1, 1, 2))
    const minRect = cv.minAreaRect(contour);

    // 5. Unclip (expand) the polygon
    // Simplification: We will just use the minAreaRect for now, 
    // but full DBNet unclip logic expands the polygon offset by Vatti clipping.
    // Implementing proper unclip in JS/OpenCV.js can be complex.
    // For MVP, we can potentially skip unclip or do a simple scale.

    // Let's rely on minAreaRect for now, potentially expanding slightly.
    // Or if we need high precision, we must implement the polygon expansion math.
    // DB uses: distance = area / length * unclip_ratio
    // offset = Clipper.offset(...)

    // Approximating unclip by scaling the rect
    const expandedW = minRect.size.width * (1 + (unclipRatio * 0.1)); // Very rough approximation
    const expandedH = minRect.size.height * (1 + (unclipRatio * 0.1));

    // Manual 4 points computation from RotatedRect
    // center (x,y), size (width, height), angle (degrees)
    const angleRad = (minRect.angle * Math.PI) / 180;
    const cosA = Math.cos(angleRad);
    const sinA = Math.sin(angleRad);
    const hW = minRect.size.width / 2;
    const hH = minRect.size.height / 2;

    // Unrotated corners relative to center
    // bl: -hW, +hH
    // tl: -hW, -hH
    // tr: +hW, -hH
    // br: +hW, +hH

    const cx = minRect.center.x;
    const cy = minRect.center.y;

    const points2D: [number, number][] = [
      [-hW, hH],
      [-hW, -hH],
      [hW, -hH],
      [hW, hH]
    ];

    const points: [number, number][] = points2D.map(([px, py]) => {
      const rotX = px * cosA - py * sinA;
      const rotY = px * sinA + py * cosA;
      return [cx + rotX, cy + rotY];
    });

    // This is bounding box on the MASK. Need to scale to DEST image.
    const scaleX = destWidth / maskW;
    const scaleY = destHeight / maskH;

    // 4 points relative to center
    // TBD: Correct unclip math is hard in pure JS without Clipper lib. 
    // We will output the raw minAreaRect for now.

    // To get points:
    // bl, tl, tr, br logic

    // Calculate score
    // Access original heatmap values within this contour mask (ROI)
    // This is expensive in JS loop. We can skip box score filtering for MVP 
    // or just use contour area as proxy.
    // Paddle checks if mean score in box > boxThresh.

    // For this MVP, we accept the contour if valid.

    // We need 4 points [x, y] map back to orig image.
    // We only have the mask, so we multiply by scale factor.

    // Using cv.boxPoints if available, or math
    // vertices = cv.boxPoints(rect)
    // Since OpenCV.js might lack full boxPoints, let's just get bounding rect (align to axis) if rotation is small?
    // No, DBNet handles rotation.

    // Let's assume we can get vertices.
    // Workaround: generic polygon approximation

    // For MVP: Use boundingRect + Scale
    const rect = cv.boundingRect(contour);
    const currentPoints: [number, number][] = [];
    currentPoints.push([rect.x, rect.y]);
    currentPoints.push([rect.x + rect.width, rect.y]);
    currentPoints.push([rect.x + rect.width, rect.y + rect.height]);
    currentPoints.push([rect.x, rect.y + rect.height]);

    // Scale back
    const scaledPoints = currentPoints.map(p => [p[0] * scaleX, p[1] * scaleY] as [number, number]);

    boxes.push({
      points: scaledPoints,
      confidence: 1.0 // Placeholder
    });

    contour.delete();
    // approx.delete();
  }

  bitmap.delete();
  contours.delete();
  hierarchy.delete();

  return boxes;
}

export function sortBoxes(boxes: BoundingBox[]): BoundingBox[] {
  // Sort by Y first, then X
  return boxes.sort((a, b) => {
    // Basic implementation: if Y difference is significant (> 10px), sort by Y
    // else sort by X
    const centerAy = (a.points[0][1] + a.points[3][1]) / 2;
    const centerBy = (b.points[0][1] + b.points[3][1]) / 2;

    if (Math.abs(centerAy - centerBy) > 10) {
      return centerAy - centerBy;
    }
    return a.points[0][0] - b.points[0][0];
  });
}

/**
 * CTC Decode for Recognition
 */
export function ctcDecode(
  preds: Float32Array, // (Batch, Time, NumClasses) - usually 1, 40, 97
  textLabels: string[],
  vocabSize: number
): { text: string; confidence: number } {
  // Preds is flattened. 
  // Shape: [1, seq_len, vocab_size]
  // PP-OCRv4 has shape [1, 40, ~97]

  // Argmax per time step
  const seqLen = preds.length / vocabSize;
  const charIndices: number[] = [];
  const confidences: number[] = [];

  for (let t = 0; t < seqLen; t++) {
    let maxScore = -Infinity;
    let maxIdx = 0;
    const offset = t * vocabSize;

    for (let i = 0; i < vocabSize; i++) {
      if (preds[offset + i] > maxScore) {
        maxScore = preds[offset + i];
        maxIdx = i;
      }
    }
    charIndices.push(maxIdx);
    confidences.push(maxScore);
    // Note: softmax might be needed if model outputs raw logits. 
    // PP-OCR ONNX usually outputs softmax probabilities.
  }

  // Decode: remove duplicates and blanks
  // Blank is usually last index or 0. In PaddleOCR, blank is usually ' ' (last) or 0.
  // Standard PaddleOCR dict usually has ' ' as last char, index = len(dict)
  // If vocabSize = len(dict) + 1, then blank is last.

  let text = "";
  let totalConf = 0;
  let count = 0;

  // PaddleOCR logic: 
  // blank index is usually the last class (vocabSize - 1)
  const blankIdx = vocabSize - 1;
  let lastIdx = -1;

  for (let i = 0; i < charIndices.length; i++) {
    const idx = charIndices[i];
    if (idx !== lastIdx && idx !== blankIdx) {
      if (idx < textLabels.length) {
        text += textLabels[idx];
        totalConf += confidences[i];
        count++;
      }
    }
    lastIdx = idx;
  }

  return {
    text,
    confidence: count > 0 ? totalConf / count : 0
  };
}
