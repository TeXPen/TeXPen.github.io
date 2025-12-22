import { Tensor } from "onnxruntime-web";

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const SIZE = 1024; // Use high-res for Paragraphs

export async function preprocessPaddleVL(imageBlob: Blob): Promise<Tensor> {
  const bitmap = await createImageBitmap(imageBlob);
  const { width, height } = bitmap;

  // 1. Calculate Scale to fit within SIZE x SIZE while preserving aspect ratio
  const scale = Math.min(SIZE / width, SIZE / height);
  const newW = Math.round(width * scale);
  const newH = Math.round(height * scale);

  // 2. Draw resized image onto a square canvas (padded with white)
  const canvas = new OffscreenCanvas(SIZE, SIZE);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error("Context failure");

  // Fill white (padding)
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, SIZE, SIZE);

  // Draw image centered or top-left?
  // Usually VLM prefers top-left or center. Let's do top-left for simplicity matching standard resize.
  ctx.drawImage(bitmap, 0, 0, width, height, 0, 0, newW, newH);

  // 3. Get Data & Normalize
  const imageData = ctx.getImageData(0, 0, SIZE, SIZE);
  const { data } = imageData;
  const float32Data = new Float32Array(SIZE * SIZE * 3);

  // Loop pixels
  for (let i = 0; i < SIZE * SIZE; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];

    // Normalize: (val/255 - mean) / std
    // CHW or HWC? ONNX usually expects NCHW [1, 3, H, W]

    // R
    float32Data[i] = (r / 255.0 - MEAN[0]) / STD[0];
    // G
    float32Data[i + SIZE * SIZE] = (g / 255.0 - MEAN[1]) / STD[1];
    // B
    float32Data[i + 2 * SIZE * SIZE] = (b / 255.0 - MEAN[2]) / STD[2];
  }

  return new Tensor("float32", float32Data, [1, 3, SIZE, SIZE]);
}
