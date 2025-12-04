import { Tensor } from '@huggingface/transformers';

export const FIXED_IMG_SIZE = 448;
export const IMAGE_MEAN = 0.9545467;
export const IMAGE_STD = 0.15394445;

export async function preprocess(imageBlob: Blob): Promise<{ tensor: Tensor; debugImage: string }> {
    // Convert Blob to ImageBitmap
    const img = await createImageBitmap(imageBlob);

    // 1. Draw to canvas to get pixel data
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('Failed to get canvas context');
    ctx.drawImage(img, 0, 0);
    let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // 1.5 Handle Transparency & Theme: Force Black on White
    // The model expects Black text on White background.
    // Our input might be White text on Transparent (Dark Mode) or Black on Transparent (Light Mode).
    const pixelData = imageData.data;
    for (let i = 0; i < pixelData.length; i += 4) {
      const alpha = pixelData[i + 3];
      if (alpha < 50) {
        // Transparent -> White
        pixelData[i] = 255;     // R
        pixelData[i + 1] = 255; // G
        pixelData[i + 2] = 255; // B
        pixelData[i + 3] = 255; // Alpha
      } else {
        // Content -> Black
        pixelData[i] = 0;       // R
        pixelData[i + 1] = 0;   // G
        pixelData[i + 2] = 0;   // B
        pixelData[i + 3] = 255; // Alpha
      }
    }
    ctx.putImageData(imageData, 0, 0);

    // 2. Trim white border
    imageData = trimWhiteBorder(imageData);

    // 3. Resize and Pad (Letterbox) to FIXED_IMG_SIZE
    const processedCanvas = resizeAndPad(imageData, FIXED_IMG_SIZE);
    const processedCtx = processedCanvas.getContext('2d');
    const processedData = processedCtx!.getImageData(0, 0, FIXED_IMG_SIZE, FIXED_IMG_SIZE);

    // 4. Normalize and create Tensor
    // transformers.js expects [batch_size, channels, height, width]
    // We need to flatten it to [1, 1, 448, 448] (grayscale)

    // DEBUG: Log the preprocessed image
    const debugImage = canvas.toDataURL();
    // console.log('[DEBUG] Preprocessed Input Image:', debugImage);

    const float32Data = new Float32Array(FIXED_IMG_SIZE * FIXED_IMG_SIZE);
    const { data } = processedData;

    let minVal = Infinity, maxVal = -Infinity, sumVal = 0;

    for (let i = 0; i < FIXED_IMG_SIZE * FIXED_IMG_SIZE; i++) {
      // Convert RGB to Grayscale using PyTorch standard weights: 0.299*R + 0.587*G + 0.114*B
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      const gray = 0.299 * r + 0.587 * g + 0.114 * b;

      // Normalize: (pixel/255 - mean) / std
      const normalized = ((gray / 255.0) - IMAGE_MEAN) / IMAGE_STD;
      float32Data[i] = normalized;

      // Stats for debugging
      if (normalized < minVal) minVal = normalized;
      if (normalized > maxVal) maxVal = normalized;
      sumVal += normalized;
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

function trimWhiteBorder(imageData: ImageData): ImageData {
    const { width, height, data } = imageData;
    let minX = width, minY = height, maxX = 0, maxY = 0;
    let found = false;

    // Detect background color from corners (like Python implementation)
    const getCornerColor = (x: number, y: number) => {
      const idx = (y * width + x) * 4;
      return [data[idx], data[idx + 1], data[idx + 2]];
    };

    const corners = [
      getCornerColor(0, 0),           // top-left
      getCornerColor(width - 1, 0),   // top-right
      getCornerColor(0, height - 1),  // bottom-left
      getCornerColor(width - 1, height - 1), // bottom-right
    ];

    // Find most common corner color as background
    const bgColor = corners[0]; // Simple: just use top-left as bg color

    // Use threshold of 15 (matching Python implementation)
    const threshold = 15;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];

        // Check if pixel differs from background by more than threshold
        if (Math.abs(r - bgColor[0]) > threshold ||
          Math.abs(g - bgColor[1]) > threshold ||
          Math.abs(b - bgColor[2]) > threshold) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          found = true;
        }
      }
    }

    if (!found) return imageData; // Return original if empty

    const w = maxX - minX + 1;
    const h = maxY - minY + 1;

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return imageData;

    // Draw the cropped region
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return imageData;
    tempCtx.putImageData(imageData, 0, 0);

    ctx.drawImage(tempCanvas, minX, minY, w, h, 0, 0, w, h);
    return ctx.getImageData(0, 0, w, h);
  }

function resizeAndPad(imageData: ImageData, targetSize: number): HTMLCanvasElement {
    const { width, height } = imageData;

    // Python logic: v2.Resize(size=447, max_size=448)
    // scale1 = 447 / min(w, h)
    // scale2 = 448 / max(w, h)
    // scale = min(scale1, scale2)

    const scale1 = (targetSize - 1) / Math.min(width, height);
    const scale2 = targetSize / Math.max(width, height);
    const scale = Math.min(scale1, scale2);

    const newW = Math.round(width * scale);
    const newH = Math.round(height * scale);

    const canvas = document.createElement('canvas');
    canvas.width = targetSize;
    canvas.height = targetSize;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Failed to get canvas context');

    // Fill with white background
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, targetSize, targetSize);

    // Draw resized image at top-left (0, 0)
    // This matches the Python implementation: padding=[0, 0, right, bottom]
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx!.putImageData(imageData, 0, 0);

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(tempCanvas, 0, 0, width, height, 0, 0, newW, newH);

    return canvas;
  }
