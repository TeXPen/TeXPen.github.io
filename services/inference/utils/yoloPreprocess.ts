
import { Tensor } from 'onnxruntime-web';

/**
 * Preprocess image for YOLOv8/v11.
 * Resize to 640x640 (preserving aspect ratio with padding).
 * Normalize 0-1.
 * Return Tensor [1, 3, 640, 640].
 */
export async function preprocessYolo(
  imageBlob: Blob,
  targetSize: number = 640
): Promise<{ tensor: Tensor; inputWidth: number; inputHeight: number; originalWidth: number; originalHeight; scale: number; padX: number; padY: number }> {
  // 1. Load image
  const bitmap = await createImageBitmap(imageBlob);
  const { width: w, height: h } = bitmap;
  const originalWidth = w;
  const originalHeight = h;

  // 2. Scale
  const scale = Math.min(targetSize / w, targetSize / h);
  const newW = Math.round(w * scale);
  const newH = Math.round(h * scale);

  // 3. Draw to canvas with gray padding
  const canvas = new OffscreenCanvas(targetSize, targetSize);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error("Context failed");

  // YOLO Usually uses 114 or 128 as grey padding, or just 0 (black).
  // TeX-Teller / Ultralytics default: 114
  ctx.fillStyle = 'rgb(114, 114, 114)';
  ctx.fillRect(0, 0, targetSize, targetSize);

  // Draw centered or top-left? Ultralytics defaults to centered padding in some modes, top-left in others?
  // Let's assume centered for standard letterbox.
  const dw = (targetSize - newW) / 2;
  const dh = (targetSize - newH) / 2;

  ctx.drawImage(bitmap, dw, dh, newW, newH);

  const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
  const { data } = imageData;

  // 4. Create Tensor [1, 3, 640, 640]
  const floatData = new Float32Array(3 * targetSize * targetSize);

  // Data is RGBA.
  // We need R RR... G GG... B BB...
  // And normalize 0-255 -> 0.0-1.0

  for (let i = 0; i < targetSize * targetSize; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];

    // standard normalization 0-1 for YOLOv8/11
    floatData[i] = r / 255.0; // R
    floatData[targetSize * targetSize + i] = g / 255.0; // G
    floatData[2 * targetSize * targetSize + i] = b / 255.0; // B
  }

  const tensor = new Tensor('float32', floatData, [1, 3, targetSize, targetSize]);

  return {
    tensor,
    inputWidth: targetSize,
    inputHeight: targetSize,
    originalWidth,
    originalHeight,
    scale,
    padX: dw,
    padY: dh
  };
}
