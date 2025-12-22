
import { BBox } from '../types';

/**
 * Basic NMS implementation.
 * @param boxes Array of boxes [x, y, w, h] with combined confidence in label or separate
 * @param scores Array of confidence scores corresponding to boxes
 * @param iouThreshold Intersection over Union threshold
 */
export function nonMaxSuppression(
  boxes: BBox[],
  iouThreshold: number
): BBox[] {
  // Sort by confidence (descending)
  const sorted = [...boxes].sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
  const results: BBox[] = [];

  while (sorted.length > 0) {
    const current = sorted.shift()!;
    results.push(current);

    // Filter out boxes with high IOU
    for (let i = sorted.length - 1; i >= 0; i--) {
      const other = sorted[i];
      if (computeIOU(current, other) > iouThreshold) {
        sorted.splice(i, 1);
      }
    }
  }

  return results;
}

function computeIOU(a: BBox, b: BBox): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);

  const intersectionW = Math.max(0, x2 - x1);
  const intersectionH = Math.max(0, y2 - y1);
  const intersectionArea = intersectionW * intersectionH;

  const areaA = a.w * a.h;
  const areaB = b.w * b.h;
  const unionArea = areaA + areaB - intersectionArea;

  if (unionArea === 0) return 0;
  return intersectionArea / unionArea;
}

/**
 * Post-processes YOLO output for multi-class detection.
 * Pix2Text MFD has 2 classes: isolated (index 0), embedding (index 1)
 * Assumes output shape [1, 4+C, N] where C = number of classes
 */
export function yoloPostprocess(
  output: Float32Array,
  dims: number[],
  confThreshold: number,
  scale: number,
  padX: number,
  padY: number
): BBox[] {
  // Check dims. Usually [1, 4+C, N] where C is num classes
  // For Pix2Text MFD: [1, 6, N] = 4 bbox coords + 2 classes
  // For single class: [1, 5, N] = 4 bbox coords + 1 conf

  const numChannels = dims[1]; // e.g. 6 (4 bbox + 2 classes) or 5 (4 bbox + 1 conf)
  const numAnchors = dims[2]; // e.g. 8400

  const numClasses = numChannels - 4; // Number of class scores
  const classNames = numClasses === 2
    ? ['isolated', 'embedding']
    : ['latex']; // Fallback for single-class models

  const boxes: BBox[] = [];

  // Loop through anchors
  for (let i = 0; i < numAnchors; i++) {
    // Find best class and its score
    let bestClassIdx = 0;
    let bestScore = 0;

    for (let c = 0; c < numClasses; c++) {
      const score = output[(4 + c) * numAnchors + i];
      if (score > bestScore) {
        bestScore = score;
        bestClassIdx = c;
      }
    }

    if (bestScore > confThreshold) {
      const cx = output[0 * numAnchors + i];
      const cy = output[1 * numAnchors + i];
      const w = output[2 * numAnchors + i];
      const h = output[3 * numAnchors + i];

      // Convert center-wh to top-left-wh
      // And unpad/unscale:
      // x_orig = (x_input - padX) / scale

      const xInput = cx - w / 2;
      const yInput = cy - h / 2;

      const x = (xInput - padX) / scale;
      const y = (yInput - padY) / scale;
      const width = w / scale;
      const height = h / scale;

      boxes.push({
        x,
        y,
        w: width,
        h: height,
        confidence: bestScore,
        label: classNames[bestClassIdx]
      });
    }
  }

  return nonMaxSuppression(boxes, 0.45); // Standard IOU threshold
}
