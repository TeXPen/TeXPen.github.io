import { Tensor } from "onnxruntime-web";
import { MODEL_CONFIG } from "../config";

const MEAN = MODEL_CONFIG.VLM_MEAN;
const STD = MODEL_CONFIG.VLM_STD;
const PATCH_SIZE = MODEL_CONFIG.VLM_PATCH_SIZE;
const MERGE_SIZE = MODEL_CONFIG.VLM_MERGE_SIZE;
const MIN_PIXELS = MODEL_CONFIG.VLM_MIN_PIXELS;
const MAX_PIXELS = MODEL_CONFIG.VLM_MAX_PIXELS;
const MAX_TOKENS = MODEL_CONFIG.VLM_MAX_TOKENS;

const RESIZE_FACTOR = PATCH_SIZE * MERGE_SIZE;

function smartResize(height: number, width: number): { height: number; width: number } {
  let h = height;
  let w = width;

  if (h < RESIZE_FACTOR) {
    w = Math.round((w * RESIZE_FACTOR) / h);
    h = RESIZE_FACTOR;
  }

  if (w < RESIZE_FACTOR) {
    h = Math.round((h * RESIZE_FACTOR) / w);
    w = RESIZE_FACTOR;
  }

  const aspect = Math.max(h, w) / Math.min(h, w);
  if (aspect > 200) {
    throw new Error(`VLM image aspect ratio too large: ${aspect}`);
  }

  let hBar = Math.round(h / RESIZE_FACTOR) * RESIZE_FACTOR;
  let wBar = Math.round(w / RESIZE_FACTOR) * RESIZE_FACTOR;

  if (hBar * wBar > MAX_PIXELS) {
    const beta = Math.sqrt((h * w) / MAX_PIXELS);
    hBar = Math.floor(h / beta / RESIZE_FACTOR) * RESIZE_FACTOR;
    wBar = Math.floor(w / beta / RESIZE_FACTOR) * RESIZE_FACTOR;
  } else if (hBar * wBar < MIN_PIXELS) {
    const beta = Math.sqrt(MIN_PIXELS / (h * w));
    hBar = Math.ceil(h * beta / RESIZE_FACTOR) * RESIZE_FACTOR;
    wBar = Math.ceil(w * beta / RESIZE_FACTOR) * RESIZE_FACTOR;
  }

  return { height: hBar, width: wBar };
}

function clampToMaxTokens(height: number, width: number): { height: number; width: number } {
  const gridH = Math.floor(height / PATCH_SIZE);
  const gridW = Math.floor(width / PATCH_SIZE);
  const tokens = gridH * gridW;
  if (tokens <= MAX_TOKENS) {
    return { height, width };
  }

  const scale = Math.sqrt(MAX_TOKENS / tokens);
  const scaledH = Math.max(RESIZE_FACTOR, Math.floor((height * scale) / RESIZE_FACTOR) * RESIZE_FACTOR);
  const scaledW = Math.max(RESIZE_FACTOR, Math.floor((width * scale) / RESIZE_FACTOR) * RESIZE_FACTOR);
  return { height: scaledH, width: scaledW };
}

export async function preprocessPaddleVL(imageBlob: Blob): Promise<{
  pixelValues: Tensor;
  grid: { t: number; h: number; w: number };
}> {
  const bitmap = await createImageBitmap(imageBlob);
  const { width, height } = bitmap;

  const { height: resizedH0, width: resizedW0 } = smartResize(height, width);
  const { height: resizedH, width: resizedW } = clampToMaxTokens(resizedH0, resizedW0);

  const canvas = new OffscreenCanvas(resizedW, resizedH);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error("Context failure");

  ctx.drawImage(bitmap, 0, 0, width, height, 0, 0, resizedW, resizedH);
  bitmap.close();

  const imageData = ctx.getImageData(0, 0, resizedW, resizedH);
  const { data } = imageData;
  const pixelCount = resizedW * resizedH;
  const float32Data = new Float32Array(pixelCount * 3);

  for (let i = 0; i < pixelCount; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];

    float32Data[i] = (r / 255.0 - MEAN[0]) / STD[0];
    float32Data[i + pixelCount] = (g / 255.0 - MEAN[1]) / STD[1];
    float32Data[i + 2 * pixelCount] = (b / 255.0 - MEAN[2]) / STD[2];
  }

  const gridH = Math.floor(resizedH / PATCH_SIZE);
  const gridW = Math.floor(resizedW / PATCH_SIZE);

  return {
    pixelValues: new Tensor("float32", float32Data, [1, 3, resizedH, resizedW]),
    grid: { t: 1, h: gridH, w: gridW }
  };
}
