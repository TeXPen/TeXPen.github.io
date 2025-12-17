
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
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error("Could not get 2d context");

  ctx.drawImage(bitmap, 0, 0);

  for (const bbox of bboxes) {
    // Ensure bounds
    const x = Math.max(0, bbox.x);
    const y = Math.max(0, bbox.y);
    const w = Math.min(bitmap.width - x, bbox.w);
    const h = Math.min(bitmap.height - y, bbox.h);

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

    // Put on new canvas to blobify
    const sliceCanvas = new OffscreenCanvas(w, h);
    const sliceCtx = sliceCanvas.getContext('2d');
    if (!sliceCtx) continue;

    sliceCtx.putImageData(region, 0, 0);
    slicedBlobs.push(await sliceCanvas.convertToBlob());
  }

  return slicedBlobs;
}
