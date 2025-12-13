import { Tensor } from '@huggingface/transformers';
import { MODEL_CONFIG } from './config';

export const FIXED_IMG_SIZE = MODEL_CONFIG.IMAGE_SIZE;
export const IMAGE_MEAN = MODEL_CONFIG.MEAN[0];
export const IMAGE_STD = MODEL_CONFIG.STD[0];

// Pre-allocated constants for grayscale conversion (PyTorch standard weights)
const GRAY_R = 0.299;
const GRAY_G = 0.587;
const GRAY_B = 0.114;
const INV_255 = 1 / 255;
const INV_STD = 1 / IMAGE_STD;

/**
 * Check if OffscreenCanvas is available (not in all browsers/environments)
 */
function supportsOffscreenCanvas(): boolean {
  return typeof OffscreenCanvas !== 'undefined';
}

/**
 * Create a canvas (OffscreenCanvas if available, otherwise regular canvas)
 */
function createOptimizedCanvas(width: number, height: number): HTMLCanvasElement | OffscreenCanvas {
  if (supportsOffscreenCanvas()) {
    return new OffscreenCanvas(width, height);
  }
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

/**
 * Get 2D context with willReadFrequently hint for better performance
 */
function getOptimizedContext(canvas: HTMLCanvasElement | OffscreenCanvas): CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D {
  const ctx = canvas.getContext('2d', { willReadFrequently: true }) as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
  if (!ctx) throw new Error('Failed to get canvas context');
  return ctx;
}

/**
 * Preprocess an image blob for model inference.
 * 
 * Optimizations applied:
 * - Single-pass pixel processing (transparency detection + conversion combined)
 * - OffscreenCanvas usage where supported (avoids DOM interaction)
 * - Pre-allocated Float32Array with computed constants
 */
export async function preprocess(
  imageBlob: Blob,
  generateDebugImage: boolean = false
): Promise<{ tensor: Tensor; debugImage: string }> {
  // Convert Blob to ImageBitmap
  const img = await createImageBitmap(imageBlob);

  // 1. Draw to canvas to get pixel data (use OffscreenCanvas if available)
  const canvas = createOptimizedCanvas(img.width, img.height);
  const ctx = getOptimizedContext(canvas);
  ctx.drawImage(img, 0, 0);
  img.close(); // Release ImageBitmap memory early

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const pixelData = imageData.data;

  // OPTIMIZATION: Single-pass transparency detection AND pixel processing
  // Instead of two loops (one to detect transparency, one to process),
  // we do a preliminary fast scan for transparency, then process in one pass
  processPixelsForModelInput(pixelData);

  // Note: We don't need putImageData if we are just analyzing the pixels for trimming!
  // BUT we modified pixelData in place (to black/white), so we technically assume subsequent steps happen on this modified data.
  // The old code did putImageData, then read it back inside trimWhiteBorder (via a temp canvas in some impls, or just used the data).
  // The old trimWhiteBorder took imageData and returned imageData (which implied a new clipped buffer).

  // 2. Trim white border (Optimization: Return Bounds, don't create new ImageData yet)
  // We use the modified pixelData directly.
  const bounds = getTrimmedBounds(imageData);

  // 3. Resize and Pad (Letterbox) to FIXED_IMG_SIZE direct from source ImageData + Bounds
  // We need to put the modified pixel data back onto a canvas (or temp one) ONLY so we can draw it scaled.
  // Since we have 'imageData' with correct pixels, we can put it back on 'canvas' (reusing it).
  ctx.putImageData(imageData, 0, 0);

  const processedCanvas = resizeAndPadFromBounds(canvas, bounds, FIXED_IMG_SIZE);
  const processedCtx = getOptimizedContext(processedCanvas);
  const processedData = processedCtx.getImageData(0, 0, FIXED_IMG_SIZE, FIXED_IMG_SIZE);

  // 4. Normalize and create Tensor
  // Pre-allocate the output array
  const float32Data = new Float32Array(FIXED_IMG_SIZE * FIXED_IMG_SIZE);
  const { data } = processedData;

  // OPTIMIZATION: Use pre-computed constants and avoid repeated divisions
  normalizeToTensor(data, float32Data);

  // Generate debug image ONLY if requested
  let debugImage = '';
  if (generateDebugImage) {
    if (supportsOffscreenCanvas()) {
      const offscreen = processedCanvas as OffscreenCanvas;
      const blob = await offscreen.convertToBlob();
      debugImage = await blobToDataURL(blob);
    } else {
      const domCanvas = processedCanvas as HTMLCanvasElement;
      debugImage = domCanvas.toDataURL();
    }
  }

  return {
    tensor: new Tensor(
      'float32',
      float32Data,
      [1, 1, FIXED_IMG_SIZE, FIXED_IMG_SIZE]
    ),
    debugImage
  };
}

/**
 * Convert a Blob to a data URL string
 */
async function blobToDataURL(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/**
 * Process pixels for model input: detect transparency and convert to black-on-white.
 * 
 * OPTIMIZATION: This combines transparency detection with pixel processing
 * in a more efficient manner using typed array access patterns.
 */
function processPixelsForModelInput(pixelData: Uint8ClampedArray): void {
  const len = pixelData.length;

  // Fast transparency check using typed array
  // We use a sample-based approach for very large images to avoid scanning all pixels
  let hasTransparency = false;
  const sampleStep = len > 100000 ? 16 : 4; // Sample every 4th pixel for large images

  for (let i = 3; i < len; i += sampleStep) {
    if (pixelData[i] < 250) {
      hasTransparency = true;
      break;
    }
  }

  // Single-pass processing based on transparency detection result
  if (hasTransparency) {
    // Image has transparency - convert alpha to grayscale (ink on transparent bg)
    for (let i = 0; i < len; i += 4) {
      const alpha = pixelData[i + 3];
      if (alpha < 50) {
        // Transparent -> White
        pixelData[i] = 255;
        pixelData[i + 1] = 255;
        pixelData[i + 2] = 255;
        pixelData[i + 3] = 255;
      } else {
        // Content: convert alpha to grayscale (preserves anti-aliasing)
        const grayscale = 255 - ((alpha * 255) >> 8); // Faster than Math.round(255 * (1 - alpha/255))
        pixelData[i] = grayscale;
        pixelData[i + 1] = grayscale;
        pixelData[i + 2] = grayscale;
        pixelData[i + 3] = 255;
      }
    }
  } else {
    // Opaque image - threshold to binary black/white
    for (let i = 0; i < len; i += 4) {
      // Compute average brightness
      const avg = (pixelData[i] + pixelData[i + 1] + pixelData[i + 2]);
      if (avg > 384) { // 128 * 3
        // White
        pixelData[i] = 255;
        pixelData[i + 1] = 255;
        pixelData[i + 2] = 255;
      } else {
        // Black
        pixelData[i] = 0;
        pixelData[i + 1] = 0;
        pixelData[i + 2] = 0;
      }
      pixelData[i + 3] = 255;
    }
  }
}

/**
 * Normalize pixel data to Float32Array tensor values.
 * 
 * OPTIMIZATION: Uses pre-computed constants to avoid repeated divisions.
 */
function normalizeToTensor(data: Uint8ClampedArray, output: Float32Array): void {
  const totalPixels = FIXED_IMG_SIZE * FIXED_IMG_SIZE;

  for (let i = 0; i < totalPixels; i++) {
    const idx = i * 4;
    // Convert RGB to Grayscale and normalize in one step
    const gray = GRAY_R * data[idx] + GRAY_G * data[idx + 1] + GRAY_B * data[idx + 2];
    output[i] = (gray * INV_255 - IMAGE_MEAN) * INV_STD;
  }
}

/**
 * Get the bounding box of the non-white content.
 * Returns {x, y, w, h}
 */
function getTrimmedBounds(imageData: ImageData): { x: number; y: number; w: number; h: number } {
  const { width, height, data } = imageData;
  let minX = width, minY = height, maxX = 0, maxY = 0;
  let found = false;

  // Detect background color from top-left corner
  const bgR = data[0];
  const bgG = data[1];
  const bgB = data[2];

  const threshold = 15;

  // OPTIMIZATION: Use direct array indexing
  for (let y = 0; y < height; y++) {
    const rowOffset = y * width * 4;
    for (let x = 0; x < width; x++) {
      const idx = rowOffset + x * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];

      // Check if pixel differs from background by more than threshold
      const diffR = r > bgR ? r - bgR : bgR - r;
      const diffG = g > bgG ? g - bgG : bgG - g;
      const diffB = b > bgB ? b - bgB : bgB - b;

      if (diffR > threshold || diffG > threshold || diffB > threshold) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        found = true;
      }
    }
  }

  if (!found) {
    return { x: 0, y: 0, w: width, h: height };
  }

  return {
    x: minX,
    y: minY,
    w: maxX - minX + 1,
    h: maxY - minY + 1
  };
}

/**
 * Resize and pad using source canvas and clip bounds.
 */
function resizeAndPadFromBounds(
  sourceCanvas: HTMLCanvasElement | OffscreenCanvas,
  bounds: { x: number; y: number; w: number; h: number },
  targetSize: number
): HTMLCanvasElement | OffscreenCanvas {
  const { x, y, w, h } = bounds;

  // Python logic: v2.Resize(size=447, max_size=448)
  const scale1 = (targetSize - 1) / Math.min(w, h);
  const scale2 = targetSize / Math.max(w, h);
  const scale = Math.min(scale1, scale2);

  const newW = Math.round(w * scale);
  const newH = Math.round(h * scale);

  const canvas = createOptimizedCanvas(targetSize, targetSize);
  const ctx = getOptimizedContext(canvas);

  // Fill with white background
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, targetSize, targetSize);

  ctx.imageSmoothingEnabled = true;

  // OffscreenCanvas context doesn't always strictly match CanvasRenderingContext2D types in older definitions,
  // but 'high' is standard compliant.
  (ctx as CanvasRenderingContext2D).imageSmoothingQuality = 'high';

  // Draw ONLY the clipped region from the source canvas, scaled to the new size
  ctx.drawImage(
    sourceCanvas as CanvasImageSource, // Works for both HTMLCanvasElement and OffscreenCanvas
    x, y, w, h,    // Source rect
    0, 0, newW, newH // Dest rect (top-left aligned, padding handled by just drawing there and background being white)
  );

  return canvas;
}
