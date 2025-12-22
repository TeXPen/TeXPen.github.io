
import { BBox } from '../types';

/**
 * Connected Components Post-processing for DBNet
 * 1. Binarize
 * 2. Find Connected Components
 * 3. Extract Boxes
 * 4. Scale back to original
 */
export function dbnetPostprocess(
  probMap: Float32Array,
  width: number, // map width
  height: number,
  threshold: number = 0.3,
  originalWidth: number,
  originalHeight: number,
  inputWidth: number, // Size passed to model
  inputHeight: number
): BBox[] {
  const visited = new Uint8Array(width * height);
  const boxes: BBox[] = [];

  const scaleX = originalWidth / width;
  // If map is smaller than inputWidth (e.g. 1/4), width < inputWidth.
  // We want to scale to original image. 
  // Map -> Scale -> Original. 
  // mapW -> origW implies scale = origW / mapW. Correct.

  const scaleY = originalHeight / height;

  const stack: number[] = []; // For DFS

  for (let i = 0; i < width * height; i++) {
    if (probMap[i] > threshold && visited[i] === 0) {
      // Start Component
      let minX = width, maxX = 0, minY = height, maxY = 0;
      let pixelCount = 0;

      stack.push(i);
      visited[i] = 1;

      while (stack.length > 0) {
        const idx = stack.pop()!;
        const y = Math.floor(idx / width);
        const x = idx % width;

        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        pixelCount++;

        // Neighbors (4-connectivity)
        // Up
        if (y > 0) {
          const nIdx = idx - width;
          if (visited[nIdx] === 0 && probMap[nIdx] > threshold) {
            visited[nIdx] = 1;
            stack.push(nIdx);
          }
        }
        // Down
        if (y < height - 1) {
          const nIdx = idx + width;
          if (visited[nIdx] === 0 && probMap[nIdx] > threshold) {
            visited[nIdx] = 1;
            stack.push(nIdx);
          }
        }
        // Left
        if (x > 0) {
          const nIdx = idx - 1;
          if (visited[nIdx] === 0 && probMap[nIdx] > threshold) {
            visited[nIdx] = 1;
            stack.push(nIdx);
          }
        }
        // Right
        if (x < width - 1) {
          const nIdx = idx + 1;
          if (visited[nIdx] === 0 && probMap[nIdx] > threshold) {
            visited[nIdx] = 1;
            stack.push(nIdx);
          }
        }
      }

      // Filter small blobs
      if (pixelCount < 50) continue; // Noise filter

      // Add Box
      const w = maxX - minX + 1;
      const h = maxY - minY + 1;

      // Basic heuristic: extremely thin or small boxes might be noise
      if (w < 5 || h < 5) continue;

      // Scale to original
      let boxX = minX * scaleX;
      let boxY = minY * scaleY;
      let boxW = w * scaleX;
      let boxH = h * scaleY;

      // Unclip / Expand
      // DBNet predicts a shrunk kernel (0.4 ratio). We need to expand it.
      // Offset = (Area * unclip_ratio) / Perimeter
      // Polygon offset formula, approx for rect:
      const area = boxW * boxH;
      const perimeter = 2 * (boxW + boxH);
      const unclipRatio = 1.5; // Standard DBNet/PaddleOCR unclip ratio
      const offset = (area * unclipRatio) / perimeter;

      // Apply expansion
      boxX = Math.max(0, boxX - offset);
      boxY = Math.max(0, boxY - offset);
      boxW = boxW + 2 * offset;
      boxH = boxH + 2 * offset;

      // Clamp to image boundaries
      if (boxX + boxW > originalWidth) boxW = originalWidth - boxX;
      if (boxY + boxH > originalHeight) boxH = originalHeight - boxY;

      boxes.push({
        x: boxX,
        y: boxY,
        w: boxW,
        h: boxH,
        label: 'text',
        confidence: 1.0 // Prob map gives confidence but aggregated
      });
    }
  }

  return boxes;
}
