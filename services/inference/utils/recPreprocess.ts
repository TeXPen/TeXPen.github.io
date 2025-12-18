import { Tensor } from "onnxruntime-web";

/**
 * Preprocess image for PaddleOCR Text Recognition (SVTR/CRNN)
 * Expected Input: Image Blob
 * Expected Output: Tensor [1, 3, 48, 320]
 */
export async function recPreprocess(imageBlob: Blob): Promise<Tensor> {
  const imgBitmap = await createImageBitmap(imageBlob);
  const targetH = 48;
  const targetW = 320;

  const canvas = new OffscreenCanvas(targetW, targetH);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas context init failed");

  // Fill with gray/pad color if needed, or just black/white.
  // PaddleOCR usually pads with 0 or 127. Let's assume standard resize-pad behavior.
  // We will resize preserving aspect ratio up to targetW, then pad right.
  // Or simple resize for now if we assume crops are roughly text line shaped.
  // Simple resize to target dimensions is often "good enough" for fixed shape models,
  // but aspect ratio preservation is better.

  const scale = targetH / imgBitmap.height;
  let newW = imgBitmap.width * scale;
  if (newW > targetW) newW = targetW;

  // Draw background (padding)
  ctx.fillStyle = "#000000"; // Black padding? Or Match standard.
  // PaddleOCR normalization usually implies 0.5 mean, 0.5 std -> 0..1 input.
  // Let's draw image.
  ctx.drawImage(imgBitmap, 0, 0, newW, targetH);

  const imageData = ctx.getImageData(0, 0, targetW, targetH);
  const { data } = imageData;

  // HWC -> CHW and Normalize
  // Mean: 0.5, Std: 0.5 (Common for Paddle)
  const mean = 0.5;
  const std = 0.5;

  const float32Data = new Float32Array(3 * targetH * targetW);

  for (let i = 0; i < targetH * targetW; i++) {
    const r = data[i * 4] / 255.0;
    const g = data[i * 4 + 1] / 255.0;
    const b = data[i * 4 + 2] / 255.0;

    // (Val - Mean) / Std
    float32Data[i] = (r - mean) / std; // R
    float32Data[i + targetH * targetW] = (g - mean) / std; // G
    float32Data[i + 2 * targetH * targetW] = (b - mean) / std; // B
  }

  return new Tensor("float32", float32Data, [1, 3, targetH, targetW]);
}
