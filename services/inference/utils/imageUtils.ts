
import { BBox } from '../types';

// We assume we are running in a worker where we can use OffscreenCanvas or similar.
// Since this is for inference preprocessing, we deal with ImageData or simple array manipulation.
// However, for performance and compatibility with the rest of the app, we might check what `opencv-js` offers or use Canvas API.
// Given strict strict TS environment and potential lack of full cv2, we'll strive for Canvas API where possible.

/**
 * Masks the given regions in the image with a background color.
 * Logic:
 *   mask_img = img.copy()
 *   for bbox in bboxes:
 *       mask_img[y:y+h, x:x+w] = bg_color
 */
export async function maskImg(
  imageBlob: Blob,
  bboxes: BBox[],
  bgColor: [number, number, number] = [255, 255, 255]
): Promise<Blob> {
  const bitmap = await createImageBitmap(imageBlob);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error("Could not get 2d context");

  ctx.drawImage(bitmap, 0, 0);

  const fillStyle = `rgb(${bgColor[0]}, ${bgColor[1]}, ${bgColor[2]})`;
  ctx.fillStyle = fillStyle;

  for (const bbox of bboxes) {
    ctx.fillRect(bbox.x, bbox.y, bbox.w, bbox.h);
  }

  // Return buffer
  const blob = await canvas.convertToBlob({ type: 'image/png' });
  return blob;
}

/**
 * Extract sub-images based on bounding boxes.
 */
export async function sliceFromImage(imageBlob: Blob, bboxes: BBox[]): Promise<Blob[]> {
  const bitmap = await createImageBitmap(imageBlob);
  const slicedBlobs: Blob[] = [];

  // Optimization: Reuse canvas?
  // Given slicing is distinct per bbox, we create new small canvases or just extract ImageData
  // But we need Blobs for generic inference input down the line.

  // We can use a shared source canvas
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error("Could not get 2d context");

  ctx.drawImage(bitmap, 0, 0);

  for (const bbox of bboxes) {
    // Ensure bounds and add padding
    // Increased padding to 12 to prevent cutting off edge characters (e.g. "test" -> "st")
    const padding = 12;
    const x = Math.max(0, bbox.x - padding);
    const y = Math.max(0, bbox.y - padding);
    const w = Math.min(bitmap.width - x, bbox.w + padding * 2);
    const h = Math.min(bitmap.height - y, bbox.h + padding * 2);

    if (w <= 0 || h <= 0) {
      // Handle empty slice?
      console.warn("Empty slice detected", bbox);
      // Create empty white blob?
      const empty = new OffscreenCanvas(1, 1);
      slicedBlobs.push(await empty.convertToBlob());
      continue;
    }

    // Get Region
    const region = ctx.getImageData(x, y, w, h);

    // Check for dark background (white text on black)
    // Heuristic: Check mean brightness
    let totalBrightness = 0;
    const data = region.data;
    const len = data.length;

    // Sample every 4th pixel for speed
    let samples = 0;
    for (let i = 0; i < len; i += 16) { // 4 channels * 4 pixels stride
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      // Brightness = 0.299R + 0.587G + 0.114B
      totalBrightness += (0.299 * r + 0.587 * g + 0.114 * b);
      samples++;
    }

    const meanBrightness = samples > 0 ? totalBrightness / samples : 255;

    // If background is dark (mean < 128), invert colors
    // We assume text is likely lighter than background if mean is low.
    if (meanBrightness < 100) { // Threshold 100 to be safe
      for (let i = 0; i < len; i += 4) {
        data[i] = 255 - data[i];     // R
        data[i + 1] = 255 - data[i + 1]; // G
        data[i + 2] = 255 - data[i + 2]; // B
        // Alpha unchanged
      }
    }

    // Put on new canvas to blobify
    const sliceCanvas = new OffscreenCanvas(w, h);
    const sliceCtx = sliceCanvas.getContext('2d', { willReadFrequently: true });
    if (!sliceCtx) continue;

    // Ensure white background
    sliceCtx.fillStyle = "#ffffff";
    sliceCtx.fillRect(0, 0, w, h);
    sliceCtx.putImageData(region, 0, 0);
    slicedBlobs.push(await sliceCanvas.convertToBlob());
  }

  return slicedBlobs;
}
